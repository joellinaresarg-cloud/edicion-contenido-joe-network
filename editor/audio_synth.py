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

STYLES = {
    # chords: notas MIDI por compás; arp: subdivisiones por compás (0 = sin arpegio)
    # kick/clap/hat: posiciones en semicorcheas (0-15) dentro del compás
    "inspira": dict(bpm=96, chords=[[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]], arp=8, wave="sine",
                    kick=[0, 8], clap=[4, 12], hat=[2, 6, 10, 14], pulse=False, lp=None, pad=0.05),
    "tech": dict(bpm=118, chords=[[57, 60, 64], [57, 60, 64], [53, 57, 60], [55, 59, 62]], arp=16, wave="square",
                 kick=[0, 4, 8, 12], clap=[4, 12], hat=[2, 6, 10, 14, 3, 7, 11, 15], pulse=True, lp=None, pad=0.035),
    "epica": dict(bpm=84, chords=[[50, 53, 57], [46, 50, 53], [53, 57, 60], [48, 52, 55]], arp=0, wave="sine",
                  kick=[0, 8, 14], clap=[8], hat=[], pulse=False, lp=None, pad=0.08, tom=[12, 14]),
    "tension": dict(bpm=92, chords=[[52, 55, 59], [52, 55, 59], [48, 52, 55], [47, 51, 54]], arp=0, wave="sine",
                    kick=[0], clap=[], hat=list(range(0, 16, 2)), pulse=True, lp=None, pad=0.06),
    "oscuro": dict(bpm=96, chords=[[48, 51, 55], [44, 48, 51], [46, 50, 53], [43, 47, 50]], arp=0, wave="sine",
                   kick=[0, 6, 8], clap=[12], hat=[4, 10, 14], pulse=True, lp=1800, pad=0.07, tom=[]),
    "lofi": dict(bpm=76, chords=[[48, 52, 55, 59], [45, 48, 52, 55], [50, 53, 57, 60], [43, 47, 50, 53]], arp=4,
                 wave="sine", kick=[0, 7, 10], clap=[4, 12], hat=[2, 6, 10, 14], pulse=False, lp=2800, pad=0.05),
}


def music(total, drums_from, break_from, cta_from, style="inspira"):
    """Música de fondo.

    drums_from: segundo donde entra la batería (después del hook).
    break_from: segundo donde se corta la batería (antes del CTA).
    cta_from:   segundo donde vuelve todo, más fuerte (CTA).
    """
    P = STYLES[style]
    n = int(total * SR)
    L = np.zeros(n)
    R = np.zeros(n)
    beat = 60 / P["bpm"]
    bar = 4 * beat
    six = beat / 4
    rng = np.random.default_rng(7)

    def add(x, start, gl=1.0, gr=1.0):
        i = int(start * SR)
        if i >= n:
            return
        x = x[: n - i]
        L[i:i + len(x)] += x * gl
        R[i:i + len(x)] += x * gr

    def osc(f, tt):
        if P["wave"] == "square":
            return sum(np.sin(2 * np.pi * f * h * tt) / h for h in (1, 3, 5, 7))
        return np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)

    nbars = int(np.ceil(total / bar)) + 1
    for b in range(nbars):
        t0 = b * bar
        ch = P["chords"][b % len(P["chords"])]
        tt = _t(bar + 1.0)
        env = np.minimum(1, tt / 0.5) * np.where(tt > bar, np.exp(-(tt - bar) * 4), 1)
        pad = np.zeros(len(tt))
        for m in ch + [ch[0] + 12]:
            for det in (-0.004, 0.004):
                f = _hz(m) * (1 + det)
                pad += sum(np.sin(2 * np.pi * f * h * tt) / h ** 1.6 for h in range(1, 5))
        add(pad * env * P["pad"], t0, 1.0, 0.9)
        root = _hz(ch[0] - 12)
        if P["pulse"]:
            for k in range(8):
                tb = _t(beat / 2)
                x = np.sin(2 * np.pi * root * tb) * np.exp(-tb * 6) * 0.2
                add(x, t0 + k * beat / 2)
        else:
            tb = _t(bar)
            bass = np.sin(2 * np.pi * root * tb) * np.minimum(1, tb / 0.02) * np.exp(-tb * 0.6)
            add(bass * 0.22, t0)
        if P["arp"]:
            notes = ch + [ch[1] + 12, ch[2] + 12, ch[0] + 12, ch[2], ch[1]]
            step = bar / P["arp"]
            for k in range(P["arp"]):
                ta = _t(min(0.6, step * 3))
                f = _hz(notes[k % len(notes)] + 12)
                x = osc(f, ta) * np.exp(-ta * 7) * (0.035 if P["wave"] == "square" else 0.06)
                pan = 0.35 + 0.3 * (k % 2)
                add(x, t0 + k * step, 1 - pan + 0.3, pan + 0.3)
    L = _reverb(L, 2.2, 0.35, 11)
    R = _reverb(R, 2.2, 0.35, 12)

    tk = _t(0.4)
    kick = np.sin(2 * np.pi * np.cumsum(45 + 90 * np.exp(-tk * 30)) / SR) * np.exp(-tk * 9) * 0.5
    tc = _t(0.25)
    clap = signal.lfilter(*signal.butter(2, [900 / (SR / 2), 3500 / (SR / 2)], "band"),
                          rng.standard_normal(len(tc)) * np.exp(-tc * 22)) * 0.25
    th = _t(0.05)
    hat = signal.lfilter(*signal.butter(2, 7000 / (SR / 2), "high"),
                         rng.standard_normal(len(th)) * np.exp(-th * 90)) * 0.07
    tm = _t(0.5)
    tom = np.sin(2 * np.pi * np.cumsum(70 + 60 * np.exp(-tm * 12)) / SR) * np.exp(-tm * 6) * 0.35

    b = 0
    while b * bar < total:
        t0 = b * bar
        for pos in range(16):
            t = t0 + pos * six
            if t < drums_from or (break_from <= t < cta_from):
                continue
            if pos in P["kick"]:
                add(kick, t)
            if pos in P["clap"]:
                add(clap, t, 0.9, 1.0)
            if pos in P["hat"]:
                add(hat, t, 0.8, 1.0)
            if pos in P.get("tom", []):
                add(tom, t, 1.0, 0.8)
        b += 1

    out = np.stack([L, R], 1)
    if P["lp"]:
        out = signal.lfilter(*signal.butter(2, P["lp"] / (SR / 2)), out, axis=0)
    tt = np.arange(n) / SR
    g = np.where(tt < drums_from, 0.6, 1.0)
    g = np.where((tt >= break_from) & (tt < cta_from), 0.75, g)
    g = np.where(tt >= cta_from, 1.35, g)
    b_, a_ = [0.0005], [1, -0.9995]
    g, _ = signal.lfilter(b_, a_, g, zi=signal.lfilter_zi(b_, a_) * g[0])
    out *= g[:, None]
    fo = int(1.2 * SR)
    out[-fo:] *= np.linspace(1, 0, fo)[:, None]
    return _norm(out, 0.9)
