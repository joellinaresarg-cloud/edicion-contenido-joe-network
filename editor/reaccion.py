"""Editor de reels en formato REACCIÓN (pantalla dividida) con el estilo joenetwork.arg.

Uso:
    python3 editor/reaccion.py CARPETA_MEDIA CARPETA_SALIDA FONTS_DIR v1 [v2 ...]

CARPETA_MEDIA debe tener los videos y sus transcripciones (faster-whisper con
tiempos por palabra) con los nombres definidos en SOURCES y BROLL.

Los clips se guardan en caché: si dos versiones usan el mismo bloque, se
renderiza una sola vez.
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy import signal as sps

sys.path.insert(0, os.path.dirname(__file__))
import audio_synth as syn  # noqa: E402

MEDIA, OUT, FONTS = sys.argv[1:4]
VERSIONS = sys.argv[4:]
CACHE = f"{OUT}/cache"
os.makedirs(CACHE, exist_ok=True)
SR, FPS = syn.SR, 30
GOLD, WHITE = "&H0000FFD4&", "&H00FFFFFF&"

# ------------------------------------------------------------------ Material
SOURCES = {  # video de la persona hablando: (archivo, transcripción, posición vertical del recorte)
    "L": ("robot1.mp4", "largo.json", 0.30),
    "Y3": ("yo3.mp4", "yo3.json", 0.28),
    "Y4": ("yo4.mp4", "yo4.json", 0.28),
}
BROLL = {  # material de robots: (archivo, desde, hasta, posición vertical del recorte)
    "tienda": ("robot2.mp4", 6.0, 22.9, 0.45),
    "fabrica": ("robot3.mp4", 0.0, 12.9, 0.10),
}

# Bloques: fuente, rangos (por inicio de palabra), layout y gráficos.
# Gráfico: (src_t_inicio, src_t_fin | None, estilo, texto, x, y)
BLOCKS = {
    "tesla": dict(src="L", ranges=[(0.0, 8.82)], layout="full", gfx=[
        (0.0, 8.82, "BoxGold", "TESLA YA LLEGÓ A ARGENTINA", 540, 330)]),
    "maneja": dict(src="L", ranges=[(24.24, 28.18)], layout="full", gfx=[]),
    "shanghai": dict(src="L", ranges=[(33.02, 40.78)], layout="split", broll=("tienda", 0.0), gfx=[
        (33.02, 40.78, "Tag", "SHANGHÁI · CHINA", 540, 190),
        (36.46, 40.78, "BoxGold", "6 ROBOTS", 540, 700),
        (40.26, 40.78, "BoxWhite", "+24 HORAS", 540, 830)]),
    "resultado": dict(src="L", ranges=[(41.82, 46.72), (48.44, 53.18)], layout="split", broll=("tienda", 7.5), gfx=[
        (43.14, 46.72, "BoxGold", "4 APROBARON", 540, 700),
        (44.92, 46.72, "BoxWhite", "2 A REVISIÓN", 540, 830),
        (48.44, 53.18, "Tag", "CAFÉ · ATENCIÓN · REPOSICIÓN", 540, 190)]),
    "seguridad": dict(src="Y3", ranges=[(2.36, 5.92), (12.44, 18.28), (18.82, 25.48)], layout="split",
                      broll=("fabrica", 0.0), gfx=[
        (2.36, 18.28, "Tag", "ROBOTS DE SEGURIDAD", 540, 190),
        (14.82, 18.28, "BoxGold", "SIN DESCANSAR", 540, 760),
        (22.44, 25.48, "BoxWhite", "VS. UN GUARDIA DORMIDO", 540, 760)]),
    "pasar": dict(src="L", ranges=[(82.74, 96.04)], layout="full", gfx=[
        (87.16, 88.14, "Big", "¿CUÁNDO?", 540, 520),
        (89.60, 91.12, "BoxGold", "1 · 2 · 5 AÑOS", 540, 520)]),
    "pasar_corto": dict(src="L", ranges=[(89.60, 96.04)], layout="full", gfx=[
        (89.60, 91.12, "BoxGold", "1 · 2 · 5 AÑOS", 540, 520)]),
    "libro": dict(src="L", ranges=[(96.58, 101.06), (101.98, 113.46), (117.74, 121.76), (122.97, 124.22)],
                  layout="full", gfx=[
        (96.58, 101.06, "Card", "SÁLVESE QUIEN PUEDA", 540, 420),
        (97.22, 101.06, "Tag", "ANDRÉS OPPENHEIMER · 2018", 540, 540),
        (122.97, 124.22, "Big", "ESE FUTURO ES HOY", 540, 520)]),
    "habilidades": dict(src="L", ranges=[(124.90, 136.60)], layout="full", gfx=[
        (129.00, 136.60, "List", "CREATIVIDAD", 540, 300),
        (130.18, 136.60, "List", "COMUNICACIÓN", 540, 400),
        (131.80, 136.60, "List", "INTELIGENCIA EMOCIONAL", 540, 500),
        (133.64, 136.60, "List", "ADAPTACIÓN", 540, 600),
        (136.08, 136.60, "List", "APRENDIZAJE CONTINUO", 540, 700)]),
    "versus": dict(src="Y4", ranges=[(0.0, 4.94)], layout="split", broll=("tienda", 13.0), gfx=[]),
    "cta_comentarios": dict(src="L", ranges=[(226.52, 233.28), (235.74, 236.98)], layout="full", gfx=[
        (230.82, 236.98, "Tag", "COMENTÁ", 540, 300),
        (233.28, 236.98, "BoxGold", "ÉXITO", 540, 420),
        (235.74, 236.98, "BoxGold", "VENTA", 540, 545),
        (236.98, 236.98, "BoxGold", "COMUNICACIÓN", 540, 670)]),
    "cta_compartir": dict(src="L", ranges=[(261.80, 271.08)], layout="full", gfx=[
        (262.54, 265.98, "BoxGold", "UBER · COMERCIO · PEDIDOSYA", 540, 420),
        (265.32, 265.98, "Tag", "COMPARTÍ ESTE VIDEO", 540, 300)]),
    "cta_serie": dict(src="L", ranges=[(243.24, 248.52)], layout="full", gfx=[]),
}

# Correcciones de subtítulos: (fuente, tiempo de la palabra) -> texto ("" = ocultar)
FIX = {("L", 8.90): "", ("L", 15.0): "", ("L", 46.72): "CIRCUITOS", ("L", 99.40): "SÁLVESE",
       ("L", 100.80): "OPPENHEIMER", ("L", 101.06): "", ("L", 98.10): "", ("L", 98.22): "LEÍ",
       ("L", 116.10): "", ("L", 114.40): "", ("L", 115.54): "", ("L", 261.80): "SI",
       ("L", 262.18): "CONOCÉS", ("L", 264.32): "PEDIDOSYA", ("L", 264.70): "",
       ("L", 265.32): "COMPARTÍ", ("Y3", 23.08): "GUARDIA", ("Y3", 9.56): "FÁBRICAS",
       ("Y4", 0.0): "UN", ("Y4", 1.26): "UN", ("Y4", 3.50): "VA", ("Y4", 4.16): "NO",
       ("Y4", 4.94): "SE ENFERMA", ("L", 237.88): ""}
KEYWORDS = {"ROBOTS", "ROBOT", "ROBOTITOS", "SHANGHAI", "CHINA", "24", "HORAS", "APROBARON", "CAFÉ",
            "SEGURIDAD", "DORMIDO", "PASAR", "TIEMPO", "PREPARARNOS", "2018", "SÁLVESE", "OPPENHEIMER",
            "DESAPARECER", "ADAPTARNOS", "HABILIDADES", "FUTURO", "HOY", "CREATIVIDAD", "COMUNICACIÓN",
            "EMOCIONAL", "ADAPTACIÓN", "CONTINUO", "ÉXITO", "VENTA", "USTEDES", "COMENTARIOS", "TESLA",
            "ARGENTINA", "MANEJANDO", "UBER", "COMPARTÍ", "BAÑO", "ENFERMA", "SEIS", "CUATRO", "SERIE",
            "LIBRO", "MUSK", "TRANSPORTE", "SIGNIFICAR", "DESCANSAR"}

BODY = ["resultado", "pasar", "libro", "habilidades"]
BODY_FULL = ["shanghai"] + BODY

V = {
    "v1": dict(nombre="curiosidad", hook=["shanghai"], headline=("ROBOTS TRABAJANDO", "24 HORAS SEGUIDAS"),
               body=BODY, cta=["cta_comentarios"], music="tech",
               end=("¿CUÁL QUERÉS PRIMERO?", "ÉXITO · VENTA · COMUNICACIÓN")),
    "v2": dict(nombre="polemica", hook=["hook_pasar"], headline=("¿TU TRABAJO", "ESTÁ EN PELIGRO?"),
               body=["shanghai", "resultado", "pasar_corto", "libro", "habilidades"],
               cta=["cta_serie"], music="epica", end=("NUEVA SERIE", "SEGUIME PARA NO PERDÉRTELA")),
    "v3": dict(nombre="advertencia", hook=["maneja"], headline=("SI TRABAJÁS DE UBER", "MIRÁ ESTO"),
               body=BODY_FULL, cta=["cta_compartir"], music="tension",
               end=("COMPARTILO", "CON ALGUIEN QUE LO NECESITE")),
    "v4": dict(nombre="comparacion", hook=["versus"], headline=("UN EMPLEADO", "VS. UN ROBOT"),
               body=BODY_FULL, cta=["cta_comentarios"], music="lofi",
               end=("GUARDALO", "Y COMPARTILO")),
    "v5": dict(nombre="noticia", hook=["tesla"], headline=None,
               body=BODY_FULL, cta=["cta_compartir"], music="inspira",
               end=("ESCRIBIME AL PRIVADO", "Y EMPEZAMOS A PREPARARNOS")),
}
BLOCKS["hook_pasar"] = dict(src="L", ranges=[(83.42, 88.14)], layout="full", gfx=[])
BLOCKS["libro_corto"] = dict(BLOCKS["libro"], ranges=[(96.58, 101.06), (101.98, 113.46), (122.97, 124.22)])
BLOCKS["seguridad_corta"] = dict(BLOCKS["seguridad"], ranges=[(2.36, 5.92), (18.82, 25.48)], gfx=[
    (2.36, 5.92, "Tag", "ROBOTS DE SEGURIDAD", 540, 190),
    (22.44, 25.48, "BoxWhite", "VS. UN GUARDIA DORMIDO", 540, 760)])
CORTO = ["shanghai", "resultado", "pasar", "libro", "habilidades"]
for _k in ("v3", "v4", "v5"):
    V[_k]["body"] = CORTO
V["v2"]["body"] = ["shanghai", "resultado", "pasar_corto", "libro", "habilidades"]

GAP, PRE, POST = 0.25, 0.06, 0.10
XF = 0.30
END_HOLD = 2.3


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(" ".join(cmd)[:500])
        print(r.stderr[-3000:])
        raise SystemExit(1)


WORDS, AUDIO = {}, {}
for k, (vid, tr, _) in SOURCES.items():
    ws = [dict(t=w["w"].strip(), s=w["s"], e=w["e"]) for seg in json.load(open(f"{MEDIA}/{tr}")) for w in seg["words"]]
    for i, w in enumerate(ws):
        w["next"] = ws[i + 1]["s"] if i + 1 < len(ws) else 1e9
        w["prev"] = ws[i - 1]["e"] if i else -1e9
    WORDS[k] = ws
    wav = f"{CACHE}/voz_{k}.wav"
    if not os.path.exists(wav):
        run(["ffmpeg", "-y", "-v", "error", "-i", f"{MEDIA}/{vid}", "-vn", "-ac", "1", "-ar", str(SR), "-af",
             "highpass=f=90,afftdn=nf=-25,acompressor=threshold=-22dB:ratio=3:attack=5:release=90:makeup=5,"
             "equalizer=f=3000:t=q:w=1:g=2", wav])
    AUDIO[k], _ = sf.read(wav)


def subclips(src, ranges):
    groups = []
    for lo, hi in ranges:
        ws = [w for w in WORDS[src] if lo - 0.01 <= w["s"] <= hi + 0.01]
        g = [ws[0]]
        for w in ws[1:]:
            if w["s"] - g[-1]["e"] > GAP:
                groups.append(g)
                g = [w]
            else:
                g.append(w)
        groups.append(g)
    clips = []
    for g in groups:
        a = max(g[0]["s"] - PRE, g[0]["prev"] + 0.02 if g[0]["prev"] > 0 else 0.0)
        a = min(a, g[0]["s"] - 0.02) if g[0]["s"] > 0.02 else 0.0
        b = min(g[-1]["e"] + POST, g[-1]["next"] - 0.02)
        clips.append(dict(a=a, dur=max(round((b - a) * FPS), 3) / FPS, words=g))
    return clips


GRADE = "eq=contrast=1.06:saturation=1.12,unsharp=5:5:0.6"


def render_clip(src, a, dur, layout, zoom, broll=None):
    key = hashlib.md5(json.dumps([src, round(a, 3), dur, layout, zoom, broll, SOURCES[src],
                                  BROLL.get(broll[0]) if broll else None, GRADE]).encode()).hexdigest()[:16]
    path = f"{CACHE}/c_{key}.mp4"
    if os.path.exists(path):
        return path
    vid, _, cy = SOURCES[src]
    frames = int(round(dur * FPS))
    if layout == "full":
        W, H = int(np.ceil(1080 * zoom / 2) * 2), int(np.ceil(1920 * zoom / 2) * 2)
        vf = (f"[0:v]scale={W}:{H}:flags=lanczos,setsar=1,crop=1080:1920:(iw-1080)/2:(ih-1920)*0.3,"
              f"{GRADE},fps={FPS},format=yuv420p[v]")
        inputs = ["-ss", f"{a:.3f}", "-i", f"{MEDIA}/{vid}"]
    else:
        bname, boff = broll
        bfile, b0, b1, by = BROLL[bname]
        span = b1 - b0
        bs = b0 + (boff % span)
        if bs + dur > b1:
            bs = b0
        W, H = int(np.ceil(1080 * zoom / 2) * 2), int(np.ceil(1920 * zoom / 2) * 2)
        vf = (f"[0:v]scale={W}:{H}:flags=lanczos,setsar=1,crop=1080:960:(iw-1080)/2:ih*{cy}[me];"
              f"[1:v]scale=1080:1920:flags=lanczos,setsar=1,crop=1080:960:0:(ih-960)*{by},fps={FPS}[br];"
              f"[br][me]vstack,{GRADE},drawbox=x=0:y=955:w=1080:h=10:color=0xD4FF00:t=fill,"
              f"fps={FPS},format=yuv420p[v]")
        inputs = ["-ss", f"{a:.3f}", "-i", f"{MEDIA}/{vid}", "-ss", f"{bs:.3f}", "-i", f"{MEDIA}/{bfile}"]
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", vf, "-map", "[v]", "-an",
         "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-r", str(FPS), path])
    return path


def voice(src, a, dur):
    i, n = int(a * SR), int(round(dur * SR))
    x = AUDIO[src][i:i + n].copy()
    x = np.pad(x, (0, n - len(x)))
    f = int(0.008 * SR)
    x[:f] *= np.linspace(0, 1, f)
    x[-f:] *= np.linspace(1, 0, f)
    return x


def ass_t(x):
    x = max(x, 0)
    return f"{int(x // 3600)}:{int(x % 3600 // 60):02d}:{x % 60:05.2f}"


def clean(w):
    return w.strip().strip("¿?¡!.,;:").upper()


POP = r"{\fscx130\fscy130\t(0,120,\fscx100\fscy100)}"

HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Barlow Condensed,123,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,800,-1,0,0,100,100,2,0,1,7,4,5,60,60,0,1
Style: CapSplit,Barlow Condensed,104,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,800,-1,0,0,100,100,2,0,1,7,3,5,60,60,0,1
Style: Logo,Barlow Condensed,55,&H00FFFFFF,&H00FFFFFF,&H0000FFD4,&H00000000,800,-1,0,0,100,100,3,0,1,0,2,8,40,40,40,1
Style: LogoBig,Barlow Condensed,106,&H00FFFFFF,&H00FFFFFF,&H0000FFD4,&H00000000,800,-1,0,0,100,100,3,0,1,0,0,5,40,40,0,1
Style: Headline,Barlow Condensed,110,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,800,-1,0,0,100,100,2,0,1,7,4,5,40,40,0,1
Style: BoxGold,Barlow Condensed,88,&H00101410,&H00101410,&H0000FFD4,&H00000000,800,-1,0,0,100,100,2,0,3,16,0,5,40,40,0,1
Style: BoxWhite,Barlow Condensed,88,&H00101410,&H00101410,&H00FFFFFF,&H00000000,800,-1,0,0,100,100,2,0,3,16,0,5,40,40,0,1
Style: Tag,Barlow Condensed,58,{GOLD},{GOLD},&H00000000,&H90000000,800,-1,0,0,100,100,5,0,1,5,3,5,40,40,0,1
Style: Big,Barlow Condensed,167,{GOLD},{GOLD},&H00000000,&H90000000,800,-1,0,0,100,100,3,0,1,8,5,5,40,40,0,1
Style: Card,Barlow Condensed,106,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,800,-1,0,0,100,100,3,0,1,7,4,5,40,40,0,1
Style: List,Barlow Condensed,81,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,800,-1,0,0,100,100,3,0,1,6,3,5,40,40,0,1
Style: EndBig,Barlow Condensed,132,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,800,-1,0,0,100,100,2,0,1,7,4,5,40,40,0,1
Style: Dim,Arial,20,&H60000000,&H60000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build(vname):
    cfg = V[vname]
    seq = cfg["hook"] + cfg["body"] + cfg["cta"]
    n_hook = len(cfg["hook"])
    parts, audio, words_tl, gfx_ev, cuts = [], [], [], [], []
    t = 0.0
    zi = 0
    for bi, bname in enumerate(seq):
        B = BLOCKS[bname]
        src = B["src"]
        clips = subclips(src, B["ranges"])
        boff = B.get("broll", (None, 0.0))[1]
        block_start = t
        if bi == n_hook:
            hook_end = t
        cuts.append((t, bname))
        tmap = []
        for k, c in enumerate(clips):
            extra = 0.0
            if bi == n_hook - 1 and k == len(clips) - 1:
                extra = XF + 0.05
            dur = c["dur"] + extra
            zoom = [1.0, 1.15][zi % 2] if B["layout"] == "full" else [1.0, 1.1][zi % 2]
            if bi < n_hook:
                zoom += 0.12
            zi += 1
            br = (B["broll"][0], round(boff, 2)) if B["layout"] == "split" else None
            parts.append(render_clip(src, c["a"], dur, B["layout"], zoom, br))
            a = voice(src, c["a"], c["dur"])
            audio.append(np.pad(a, (0, int(round(dur * FPS)) * SR // FPS - len(a))))
            off = t if bi < n_hook else t - XF
            for w in c["words"]:
                words_tl.append((off + w["s"] - c["a"], off + w["e"] - c["a"], w["t"], src, w["s"], B["layout"]))
            tmap.append((c["a"], c["a"] + c["dur"], off))
            boff += dur
            t += int(round(dur * FPS)) / FPS
        # gráficos del bloque
        for s0, s1, style, text, x, y in B["gfx"]:
            def m(st):
                for a0, a1, off in tmap:
                    if a0 - 0.05 <= st <= a1 + 0.05:
                        return off + st - a0
                cand = [(abs(st - a0), off + max(0, min(st, a1) - a0)) for a0, a1, off in tmap]
                return min(cand)[1]
            gs, ge = m(s0), m(s1) + 0.5
            if ge - gs < 0.6:
                ge = gs + 1.2
            gfx_ev.append((gs, ge, style, text, x, y))
    hook_len = hook_end
    # Unir: hook + resto con transición
    hook_mp4, rest_mp4 = f"{OUT}/{vname}_hook.mp4", f"{OUT}/{vname}_rest.mp4"
    nh = sum(1 for _ in parts)  # placeholder
    hook_parts, rest_parts, acc = [], [], 0.0
    for p in parts:
        d = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                           "csv=p=0", p]).decode())
        (hook_parts if acc < hook_len - 0.01 else rest_parts).append(p)
        acc += d
    for lst, out in [(hook_parts, hook_mp4), (rest_parts, rest_mp4)]:
        with open(out + ".txt", "w") as f:
            f.writelines(f"file '{p}'\n" for p in lst)
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", out + ".txt", "-c", "copy", out])
    total = t - XF
    joined = f"{OUT}/{vname}_joined.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-i", hook_mp4, "-i", rest_mp4, "-filter_complex",
         f"[0][1]xfade=transition=zoomin:duration={XF}:offset={hook_len - XF:.3f},"
         f"tpad=stop_mode=clone:stop_duration={END_HOLD}[v]",
         "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", joined])
    TOTAL = total + END_HOLD
    cta_start = [c for c in cuts if c[1] == cfg["cta"][0]][0][0] - XF

    # ---------------- audio
    N = int(TOTAL * SR) + SR
    vox = np.zeros(N)
    pos = 0
    hook_samples = int(round(hook_len * SR))
    for a in audio:
        i = pos if pos < hook_samples else pos - int(XF * SR)
        vox[i:i + len(a)] += a[: max(0, N - i)]
        pos += len(a)
    vox = vox / (np.max(np.abs(vox)) or 1) * 0.85
    mus = syn.music(N / SR, drums_from=hook_len - XF, break_from=cta_start - 2.4, cta_from=cta_start,
                    style=cfg["music"])
    env = np.convolve(np.abs(vox), np.ones(int(0.05 * SR)) / int(0.05 * SR), "same")
    att = sps.lfilter([0.002], [1, -0.998], (env > 0.02).astype(float))
    mix = mus * (0.30 - 0.18 * np.clip(att * 1.5, 0, 1))[:, None] + vox[:, None]
    sfx = np.zeros((N, 2))

    def fx(x, tt, g):
        i = int(max(tt, 0) * SR)
        x = x[: max(0, N - i)]
        sfx[i:i + len(x)] += x * g

    fx(syn.impact(), 0, 0.4)
    fx(syn.whoosh(0.6, 1), hook_len - XF - 0.3, 0.55)
    prev_layout = None
    for c_t, bname in cuts[1:]:
        lay = BLOCKS[bname]["layout"]
        if lay != prev_layout and bname not in cfg["cta"]:
            fx(syn.whoosh(0.4, int(c_t * 10)), c_t - XF - 0.25, 0.25)
        prev_layout = lay
    for k, (gs, ge, style, *_r) in enumerate(gfx_ev):
        fx(syn.pop(k), gs, 0.45 if style != "Tag" else 0.25)
    fx(syn.riser(2.0), cta_start - 2.0, 0.4)
    fx(syn.impact(), cta_start, 0.6)
    fx(syn.ding(), total + 0.2, 0.4)
    print(f"  bloques: {[(round(c[0],1), c[1]) for c in cuts]}")
    mix = mix + sfx
    mix = mix / (np.max(np.abs(mix)) or 1) * 0.95
    sf.write(f"{OUT}/{vname}_mix.wav", mix[: int(TOTAL * SR)], SR)

    # ---------------- subtítulos y gráficos
    ev = []
    ev.append(f"Dialogue: 3,{ass_t(0)},{ass_t(total)},Logo,,0,0,0,,"
              r"JOE{\c&H0000FFD4&}.NETWORK")
    if cfg["headline"]:
        l1, l2 = cfg["headline"]
        y1 = 300 if BLOCKS[cfg["hook"][0]]["layout"] == "full" else 640
        ev.append(f"Dialogue: 4,{ass_t(0.1)},{ass_t(hook_len - XF)},Headline,,0,0,0,,{POP}{{\\pos(540,{y1})}}{l1}")
        ev.append(f"Dialogue: 4,{ass_t(0.4)},{ass_t(hook_len - XF)},BoxGold,,0,0,0,,{POP}{{\\pos(540,{y1 + 140})}}{l2}")
    for gs, ge, style, text, x, y in gfx_ev:
        if cfg["headline"] and gs < hook_len - XF and y > 250:
            continue  # en el hook manda el titular
        ev.append(f"Dialogue: 2,{ass_t(gs)},{ass_t(ge)},{style},,0,0,0,,{POP}{{\\pos({x},{y})}}{text}")
    # final
    e1, e2 = cfg["end"]
    ev.append(f"Dialogue: 0,{ass_t(total)},{ass_t(TOTAL)},Dim,,0,0,0,,"
              r"{\fad(250,0)\p1}m 0 0 l 1080 0 1080 1920 0 1920{\p0}")
    ev.append(f"Dialogue: 5,{ass_t(total + 0.05)},{ass_t(TOTAL)},EndBig,,0,0,0,,{POP}{{\\pos(540,760)}}{e1}")
    ev.append(f"Dialogue: 5,{ass_t(total + 0.35)},{ass_t(TOTAL)},BoxGold,,0,0,0,,{POP}{{\\pos(540,920)}}{e2}")
    ev.append(f"Dialogue: 5,{ass_t(total + 0.8)},{ass_t(TOTAL)},LogoBig,,0,0,0,,{POP}"
              r"{\pos(540,1120)}JOE{\c&H0000FFD4&}.NETWORK")
    # subtítulos
    items = []
    for s, e, txt, src, st, lay in sorted(words_tl):
        fx_ = FIX.get((src, round(st, 2)))
        if fx_ == "":
            continue
        w = fx_ or clean(txt)
        if w:
            items.append([s, e, w, txt.strip()[-1:] in ".,?!", lay])
    chunks, cur = [], []
    for it in items:
        if cur and (len(cur) >= 3 or len(" ".join(c[2] for c in cur + [it])) > 16 or it[0] - cur[-1][1] > 0.4
                    or cur[-1][3] or it[4] != cur[-1][4]):
            chunks.append(cur)
            cur = []
        cur.append(it)
    chunks.append(cur)
    for k, ch in enumerate(chunks):
        st, en = ch[0][0], ch[-1][1] + 0.25
        if k + 1 < len(chunks):
            en = min(en, chunks[k + 1][0][0])
        en = min(en, total)
        txt = " ".join((r"{\c" + GOLD + "}" + c[2] + r"{\c" + WHITE + "}") if c[2] in KEYWORDS else c[2] for c in ch)
        if ch[0][4] == "split":
            ev.append(f"Dialogue: 1,{ass_t(st)},{ass_t(en)},CapSplit,,0,0,0,,"
                      r"{\pos(540,1010)\fscx112\fscy112\t(0,90,\fscx100\fscy100)}" + txt)
        else:
            ev.append(f"Dialogue: 1,{ass_t(st)},{ass_t(en)},Cap,,0,0,0,,"
                      r"{\pos(540,1330)\fscx112\fscy112\t(0,90,\fscx100\fscy100)}" + txt)
    ass = f"{OUT}/{vname}.ass"
    with open(ass, "w") as f:
        f.write(HEADER + "\n".join(ev) + "\n")

    final = f"{OUT}/reaccion_{vname}_{cfg['nombre']}.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-i", joined, "-i", f"{OUT}/{vname}_mix.wav",
         "-f", "lavfi", "-i", f"color=c=0xD4FF00:s=1080x12:r={FPS}:d={TOTAL:.3f}",
         "-filter_complex",
         f"[0][2]overlay=x='-W+W*t/{TOTAL:.3f}':y=H-12:shortest=1[b];[b]subtitles={ass}:fontsdir={FONTS}[v];"
         "[1]loudnorm=I=-14:TP=-1.5:LRA=11[a]",
         "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-t", f"{TOTAL:.3f}", final])
    print(f"OK {vname}: {final}  duración={TOTAL:.1f}s hook={hook_len:.1f}s cta={cta_start:.1f}s", flush=True)


for v in VERSIONS:
    build(v)
