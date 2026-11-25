"""Sintetizador de música de fondo y efectos de sonido (SFX) para los reels.

Todo se genera con numpy, sin bibliotecas de audio externas, así que no hay
problemas de derechos de autor.
"""
import numpy as np
from scipy import signal

SR = 48000


def _t(dur):
    return np.arange(int(dur * SR)) / SR


def _norm(x, peak=0.9):
    m = np.max(np.abs(x)) or 1.0
    return x * (peak / m)


def _reverb(x, seconds=1.8, mix=0.25, seed=1):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 7, n))
    ir = signal.lfilter(*signal.butter(1, 5000 / (SR / 2)), ir)
    wet = signal.fftconvolve(x, ir)[: len(x)]
    wet = wet / (np.max(np.abs(wet)) or 1) * np.max(np.abs(x))
    return (1 - mix) * x + mix * wet


def _hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


# ---------------------------------------------------------------- SFX

def whoosh(dur=0.55, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    block = 256
    zi = None
    for i in range(0, n, block):
        p = i / n
        fc = 300 + 4500 * np.sin(np.pi * p) ** 2
        b, a = signal.butter(2, [max(fc * 0.5, 60) / (SR / 2), min(fc * 1.6, SR / 2 - 100) / (SR / 2)], "band")
        if zi is None:
            zi = signal.lfilter_zi(b, a) * 0
        seg, zi = signal.lfilter(b, a, noise[i:i + block], zi=zi)
        out[i:i + block] = seg
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5
    out = _norm(out * env, 0.8)
    pan = np.linspace(0.2, 0.8, n)
    return np.stack([out * (1 - pan) * 1.4, out * pan * 1.4], 1)


def pop(seed=0):
    t = _t(0.12)
    f = 500 + 700 * np.exp(-t * 60)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-t * 35)
    x[:30] += np.random.default_rng(seed).standard_normal(30) * 0.3
    x = _norm(x, 0.7)
    return np.stack([x, x], 1)


def impact():
    t = _t(1.6)
    f = 30 + 60 * np.exp(-t * 8)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.6)
    rng = np.random.default_rng(3)
    nz = rng.standard_normal(len(t)) * np.exp(-t * 18)
    nz = signal.lfilter(*signal.butter(2, 2500 / (SR / 2)), nz)
    x = _reverb(boom + 0.5 * nz, 2.0, 0.3)
    x = _norm(x, 0.95)
    return np.stack([x, x], 1)


def riser(dur=2.0):
    t = _t(dur)
    rng = np.random.default_rng(4)
    nz = rng.standard_normal(len(t))
    nz = signal.lfilter(*signal.butter(2, 1500 / (SR / 2), "high"), nz)
    f = 180 * (2 ** (np.linspace(0, 2.6, len(t))))
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.4 * np.sin(4 * np.pi * np.cumsum(f) / SR)
    env = np.linspace(0, 1, len(t)) ** 2.2
    x = _norm((0.6 * nz + 0.5 * tone) * env, 0.8)
    return np.stack([x, x], 1)


def ding():
    t = _t(1.4)
    x = sum(a * np.sin(2 * np.pi * 1318 * r * t) * np.exp(-t * d)
            for r, a, d in [(1, 1, 3), (2.76, 0.4, 5), (5.4, 0.2, 8)])
    x = _norm(_reverb(x, 1.5, 0.3), 0.6)
    return np.stack([x, x], 1)


# ---------------------------------------------------------------- Música

def music(total, drums_from, break_from, cta_from, bpm=96):
    """Base inspiracional: Am - F - C - G.

    drums_from: segundo donde entra la batería (después del hook).
    break_from: segundo donde se corta la batería (antes del CTA).
    cta_from:   segundo donde vuelve todo, más fuerte (CTA).
    """
    n = int(total * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    beat = 60 / bpm
    bar = 4 * beat
    chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]
    roots = [45, 41, 36, 43]
    rng = np.random.default_rng(7)

    def add(x, start, gl=1.0, gr=1.0):
        i = int(start * SR)
        if i >= n:
            return
        x = x[: n - i]
        L[i:i + len(x)] += x * gl
        R[i:i + len(x)] += x * gr

    nbars = int(np.ceil(total / bar)) + 1
    for b in range(nbars):
        t0 = b * bar
        ch = chords[b % 4]
        # Pad
        tt = _t(bar + 1.0)
        env = np.minimum(1, tt / 0.5) * np.where(tt > bar, np.exp(-(tt - bar) * 4), 1)
        pad = np.zeros(len(tt))
        for m in ch + [ch[0] + 12]:
            for det in (-0.004, 0.004):
                f = _hz(m) * (1 + det)
                pad += sum(np.sin(2 * np.pi * f * h * tt) / h ** 1.6 for h in range(1, 5))
        pad = pad * env * 0.05
        add(pad, t0, 1.0, 0.9)
        # Bajo
        tb = _t(bar)
        bass = np.sin(2 * np.pi * _hz(roots[b % 4]) * tb) * np.minimum(1, tb / 0.02) * np.exp(-tb * 0.6)
        add(bass * 0.22, t0)
        # Arpegio
        arp = ch + [ch[1] + 12, ch[2] + 12, ch[0] + 12, ch[2], ch[1]]
        for k in range(8):
            ta = _t(0.6)
            f = _hz(arp[k % len(arp)] + 12)
            x = (np.sin(2 * np.pi * f * ta) + 0.3 * np.sin(4 * np.pi * f * ta)) * np.exp(-ta * 7) * 0.06
            pan = 0.35 + 0.3 * (k % 2)
            add(x, t0 + k * beat / 2, 1 - pan + 0.3, pan + 0.3)
    # Reverb
    L = _reverb(L, 2.2, 0.35, 11)
    R = _reverb(R, 2.2, 0.35, 12)

    # Batería
    def kick():
        tk = _t(0.4)
        f = 45 + 90 * np.exp(-tk * 30)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tk * 9) * 0.5

    def clap():
        tc = _t(0.25)
        x = rng.standard_normal(len(tc)) * np.exp(-tc * 22)
        return signal.lfilter(*signal.butter(2, [900 / (SR / 2), 3500 / (SR / 2)], "band"), x) * 0.25

    def hat():
        th = _t(0.05)
        x = rng.standard_normal(len(th)) * np.exp(-th * 90)
        return signal.lfilter(*signal.butter(2, 7000 / (SR / 2), "high"), x) * 0.07

    k_, c_, h_ = kick(), clap(), hat()
    t = drums_from
    i = 0
    while t < total:
        in_break = break_from <= t < cta_from
        if not in_break:
            if i % 4 in (0, 2):
                add(k_, t)
            if i % 4 in (1, 3):
                add(c_, t, 0.9, 1.0)
            add(h_, t + beat / 2, 0.8, 1.0)
        t += beat
        i += 1

    out = np.stack([L, R], 1)
    # Volumen por sección: hook suave, cuerpo normal, CTA más fuerte
    tt = np.arange(n) / SR
    g = np.where(tt < drums_from, 0.6, 1.0)
    g = np.where((tt >= break_from) & (tt < cta_from), 0.75, g)
    g = np.where(tt >= cta_from, 1.35, g)
    b_, a_ = [0.0005], [1, -0.9995]
    g, _ = signal.lfilter(b_, a_, g, zi=signal.lfilter_zi(b_, a_) * g[0])
    out *= g[:, None]
    # Fade final
    fo = int(1.2 * SR)
    out[-fo:] *= np.linspace(1, 0, fo)[:, None]
    return _norm(out, 0.9)
