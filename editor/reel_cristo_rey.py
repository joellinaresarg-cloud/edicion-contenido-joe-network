"""Edición del reel "Cristo Rey" con el estilo de joenetwork.arg.

Uso:
    python3 editor/reel_cristo_rey.py VIDEO.mp4 TRANSCRIPCION.json CARPETA_SALIDA FONTS_DIR

La transcripción es la salida de faster-whisper con tiempos por palabra
(lista de segmentos, cada uno con "words": [{"w", "s", "e"}]).
"""
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import audio_synth as syn  # noqa: E402

SRC, TRANS, OUT, FONTS = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
SR = syn.SR
FPS = 30
GOLD = "&H0000FFD4&"
WHITE = "&H00FFFFFF&"

# ------------------------------------------------------------------ EDL
# Rangos en segundos del video original (por inicio de palabra).
HOOK = [(64.58, 68.02), (69.62, 69.92)]          # "Cumplir los sueños no es para cualquiera. No es fácil."
BROLL_VIDEO_FROM = 0.3                            # toma del Cristo sin persona
BROLL_VO = [(5.70, 6.32), (8.16, 9.10)]           # "Estamos acá en León. Eso es Cristo Rey."
BODY = [
    (9.76, 14.72),   # Venimos 30 minutos subiendo... me pregunto
    (16.30, 23.34),  # ¿Qué cosas puede uno vivir...? Primero... segundo...
    (24.04, 30.32),  # Hace siete años... México, Las Vegas, Chile...
    (31.52, 32.68), (34.22, 37.74),  # Sin embargo, algo... vino a través de un audio, de una oportunidad
    (39.46, 44.18),  # ¿Y qué tan importante darle valor? ... esa es la vida
    (45.14, 46.38),  # Entonces, te propongo algo
    (47.30, 56.08),  # Primero, tomar tu sueño en serio...
    (56.96, 62.48),  # Segundo, estar dispuesto...
    (63.44, 68.02), (69.62, 69.92),  # Cumplir los sueños no es para cualquiera. No es fácil.
    (70.94, 76.20),  # Lo fácil es quedarse...
    (77.44, 79.06),  # Ser disruptivo...
    (82.44, 95.24),  # Así como yo hace siete años... ojalá ustedes también lo puedan contar
]
CTA_FROM, CTA_TO = 95.7, 104.0

# Correcciones de subtítulos (por tiempo de inicio de la palabra). "" = no mostrar.
FIX = {8.90: "", 22.76: "HACER", 22.86: "", 57.82: "ESTAR", 54.68: "SOÑÁS",
       78.46: "COMO", 92.84: "HOY", 93.18: "LO", 43.40: ""}
KEYWORDS = {"SUEÑOS", "SUEÑO", "CUALQUIERA", "FÁCIL", "LEÓN", "CRISTO", "REY", "VIVIR", "DISPUESTO",
            "IMAGINARSE", "SIETE", "OFICINA", "MÉXICO", "VEGAS", "CHILE", "OPORTUNIDAD", "VALOR",
            "VIDA", "PROPONGO", "PRIMERO", "SEGUNDO", "SERIEDAD", "OPCIÓN", "SALGA", "SISTEMA",
            "DISRUPTIVO", "DIFERENTE", "MUNDO", "CONTAR", "HACER", "TIEMPO", "AUDIO", "SOÑÁS"}

GAP = 0.30        # silencios más largos que esto se cortan
PRE, POST = 0.06, 0.10


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:])
        raise SystemExit(1)


def clean(w):
    return w.strip().strip("¿?¡!.,;:").upper()


words = [dict(t=w["w"].strip(), s=w["s"], e=w["e"]) for seg in json.load(open(TRANS)) for w in seg["words"]]
for i, w in enumerate(words):
    w["next"] = words[i + 1]["s"] if i + 1 < len(words) else 1e9
    w["prev"] = words[i - 1]["e"] if i else -1e9


def subclips(ranges):
    out = []
    for lo, hi in ranges:
        ws = [w for w in words if lo - 0.01 <= w["s"] <= hi + 0.01]
        group = [ws[0]]
        for w in ws[1:]:
            if w["s"] - group[-1]["e"] > GAP:
                out.append(group)
                group = [w]
            else:
                group.append(w)
        out.append(group)
    clips = []
    for g in out:
        a = max(g[0]["s"] - PRE, g[0]["prev"] + 0.02 if g[0]["prev"] > 0 else 0)
        a = min(a, g[0]["s"] - 0.02)
        b = min(g[-1]["e"] + POST, g[-1]["next"] - 0.02)
        dur = max(round((b - a) * FPS), 3) / FPS
        clips.append(dict(a=a, dur=dur, words=g))
    return clips


# ------------------------------------------------------------------ Audio de voz
voice_wav = f"{OUT}/voice.wav"
run(["ffmpeg", "-y", "-v", "error", "-i", SRC, "-vn", "-ac", "1", "-ar", str(SR), "-af",
     "highpass=f=90,afftdn=nf=-25,acompressor=threshold=-22dB:ratio=3:attack=5:release=90:makeup=5,"
     "equalizer=f=3000:t=q:w=1:g=2", voice_wav])
VOICE, _ = sf.read(voice_wav)


def vo(a, dur):
    i, n = int(a * SR), int(round(dur * SR))
    x = VOICE[i:i + n].copy()
    x = np.pad(x, (0, n - len(x)))
    f = int(0.008 * SR)
    x[:f] *= np.linspace(0, 1, f)
    x[-f:] *= np.linspace(1, 0, f)
    return x


# ------------------------------------------------------------------ Video
GRADE = "eq=contrast=1.06:saturation=1.15:brightness=0.01,unsharp=5:5:0.7"


def render(a, dur, path, zoom=1.0, push=False):
    frames = int(round(dur * FPS))
    if push:
        vf = (f"scale=1080:1920:flags=lanczos,setsar=1,{GRADE},"
              f"zoompan=z='1+0.0009*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1080x1920:fps={FPS}")
    else:
        W, H = int(np.ceil(1080 * zoom / 2) * 2), int(np.ceil(1920 * zoom / 2) * 2)
        vf = (f"scale={W}:{H}:flags=lanczos,setsar=1,crop=1080:1920:(iw-1080)/2:(ih-1920)*0.30,"
              f"{GRADE},fps={FPS}")
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{a:.3f}", "-i", SRC, "-an", "-vf", vf + ",format=yuv420p",
         "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-r", str(FPS), path])
    return frames / FPS


def concat(paths, out):
    lst = out + ".txt"
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in paths)
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out])


timeline_words = []   # (t_inicio, t_fin, palabra, tiempo_original)


def build_part(name, clips, zooms, tail=0.0, offset=0.0):
    paths, audio, t = [], [], 0.0
    for k, c in enumerate(clips):
        extra = tail if k == len(clips) - 1 else 0.0
        p = f"{OUT}/{name}_{k:03d}.mp4"
        d = render(c["a"], c["dur"] + extra, p, zoom=zooms[k % len(zooms)])
        a = vo(c["a"], c["dur"])
        audio.append(np.pad(a, (0, int(round(d * SR)) - len(a))))
        for w in c["words"]:
            timeline_words.append((offset + t + w["s"] - c["a"], offset + t + w["e"] - c["a"], w["t"], w["s"]))
        paths.append(p)
        t += d
    out = f"{OUT}/{name}.mp4"
    concat(paths, out)
    return out, np.concatenate(audio), t


XF1, XF2 = 0.30, 0.40
print("Hook…")
hook_clips = subclips(HOOK)
hook_mp4, hook_audio, Lh = build_part("hook", hook_clips, [1.32, 1.18], tail=0.35)

print("Cuerpo…")
body_start = Lh - XF1
# B-roll con voz en off
vo_clips = subclips(BROLL_VO)
vo_audio = [np.zeros(int(0.35 * SR))]
t = 0.35
for c in vo_clips:
    for w in c["words"]:
        timeline_words.append((body_start + t + w["s"] - c["a"], body_start + t + w["e"] - c["a"], w["t"], w["s"]))
    vo_audio.append(vo(c["a"], c["dur"]))
    t += c["dur"]
broll_len = render(BROLL_VIDEO_FROM, t + 0.35, f"{OUT}/broll.mp4", push=True)
vo_audio.append(np.zeros(int(round(broll_len * SR)) - sum(len(x) for x in vo_audio)))
broll_audio = np.concatenate(vo_audio)

body_clips = subclips(BODY)
talk_mp4, talk_audio, Lt = build_part("talk", body_clips, [1.0, 1.14], tail=0.45, offset=body_start + broll_len)
body_mp4 = f"{OUT}/body.mp4"
concat([f"{OUT}/broll.mp4", talk_mp4], body_mp4)
body_audio = np.concatenate([broll_audio, talk_audio])
Lb = broll_len + Lt

print("CTA…")
cta_start = body_start + Lb - XF2
Lc = render(CTA_FROM, CTA_TO - CTA_FROM, f"{OUT}/cta.mp4", push=True)
TOTAL = cta_start + Lc

print("Uniendo con transiciones…")
joined = f"{OUT}/joined.mp4"
run(["ffmpeg", "-y", "-v", "error", "-i", hook_mp4, "-i", body_mp4, "-i", f"{OUT}/cta.mp4", "-filter_complex",
     f"[0][1]xfade=transition=zoomin:duration={XF1}:offset={Lh - XF1:.3f}[v1];"
     f"[v1][2]xfade=transition=fadewhite:duration={XF2}:offset={cta_start:.3f}[v]",
     "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", joined])

# ------------------------------------------------------------------ Mezcla de audio
N = int(TOTAL * SR) + SR
voice = np.zeros(N)


def place(buf, x, t):
    i = int(t * SR)
    x = x[: max(0, len(buf) - i)]
    buf[i:i + len(x)] += x


place(voice, hook_audio, 0)
place(voice, body_audio, body_start)
voice = voice / (np.max(np.abs(voice)) or 1) * 0.85

mus = syn.music(N / SR, drums_from=body_start, break_from=cta_start - 2.6, cta_from=cta_start)
# Ducking: baja la música cuando hablás
env = np.convolve(np.abs(voice), np.ones(int(0.05 * SR)) / int(0.05 * SR), "same")
active = (env > 0.02).astype(float)
from scipy import signal as _s  # noqa: E402
att = _s.lfilter([0.002], [1, -0.998], active)
duck = 0.30 - 0.18 * np.clip(att * 1.5, 0, 1)
mix = mus * duck[:, None] + voice[:, None]

sfx = np.zeros((N, 2))


def fx(x, t, g=1.0):
    i = int(max(t, 0) * SR)
    x = x[: max(0, N - i)]
    sfx[i:i + len(x)] += x * g


fx(syn.impact(), 0.0, 0.45)
fx(syn.whoosh(0.6, 1), Lh - XF1 - 0.3, 0.55)


def tw(src_time):
    """Tiempo final de la palabra que empieza en src_time (primera aparición después del hook)."""
    c = [w for w in timeline_words if abs(w[3] - src_time) < 0.02 and w[0] > Lh]
    return c[0][0] if c else None


graphics = []   # eventos ASS adicionales


def ass_t(x):
    x = max(x, 0)
    return f"{int(x // 3600)}:{int(x % 3600 // 60):02d}:{x % 60:05.2f}"


def ev(start, end, style, text, layer=2):
    graphics.append(f"Dialogue: {layer},{ass_t(start)},{ass_t(end)},{style},,0,0,0,,{text}")


POP = r"{\fscx130\fscy130\t(0,120,\fscx100\fscy100)}"

# Hook: titular arriba
ev(0.15, Lh - XF1, "Headline", POP + r"{\pos(540,330)}DE LA OFICINA")
ev(0.45, Lh - XF1, "BoxGold", POP + r"{\pos(540,470)}A RECORRER EL MUNDO")

# B-roll: ubicación
ev(body_start + 0.25, body_start + broll_len, "Place", r"{\fad(200,150)\pos(540,760)\fscx120\t(0,300,\fscx100)}CRISTO REY")
ev(body_start + 0.55, body_start + broll_len, "PlaceSub", r"{\fad(200,150)\pos(540,870)}LEÓN · GUANAJUATO · MÉXICO")
fx(syn.pop(), body_start + 0.25, 0.5)

# 7 años
t7 = tw(24.32)
if t7:
    ev(t7, t7 + 1.8, "Big", POP + r"{\pos(540,560)}7 AÑOS")
    fx(syn.pop(1), t7, 0.6)

# Países
tend = tw(30.32)
for k, (st, txt) in enumerate([(28.06, "MÉXICO"), (28.74, "LAS VEGAS"), (29.26, "CHILE")]):
    tt = tw(st)
    if tt:
        ev(tt, (tend or tt) + 1.0, "BoxGold", POP + rf"{{\pos(540,{420 + k * 135})}}{txt}")
        fx(syn.pop(k + 2), tt, 0.6)

# Primero / Segundo
for st, until, num, txt in [(47.30, 56.08, "1", "TOMÁ TU SUEÑO EN SERIO"),
                            (56.96, 62.48, "2", "HACÉ LO QUE HAY QUE HACER")]:
    a = tw(st)
    b = tw(until)
    if a and b:
        ev(a, b + 0.5, "Num", POP + r"{\pos(540,470)}" + num)
        ev(a + 0.15, b + 0.5, "NumSub", r"{\fad(150,0)\pos(540,700)}" + txt)
        fx(syn.whoosh(0.45, 5), a - 0.2, 0.45)
        fx(syn.ding(), a + 0.05, 0.35)

# Whooshes suaves entre ideas
for st in [16.30, 39.46, 45.14, 63.44, 70.94, 82.44]:
    a = tw(st)
    if a:
        fx(syn.whoosh(0.4, int(st)), a - 0.25, 0.22)

# CTA
fx(syn.riser(2.2), cta_start - 2.2, 0.45)
fx(syn.impact(), cta_start, 0.8)
ev(cta_start, TOTAL, "Dim", r"{\fad(400,0)\p1}m 0 0 l 1080 0 1080 1920 0 1920{\p0}", layer=0)
ev(cta_start + 0.25, TOTAL, "CtaBig", POP + r"{\pos(540,640)}¿NUNCA LA PROBASTE?")
ev(cta_start + 1.6, TOTAL, "BoxGold", POP + r"{\pos(540,820)}ESCRIBIME AL PRIVADO")
ev(cta_start + 2.8, TOTAL, "PlaceSub", POP + r"{\pos(540,960)}ESPECIALMENTE SI ESTÁS EN")
ev(cta_start + 3.2, TOTAL, "CtaCountry", POP + r"{\pos(540,1080)}MÉXICO · COLOMBIA")
ev(cta_start + 4.3, TOTAL, "LogoBig",
   POP + r"{\pos(540,1290)}JOE{\c&H0000FFD4&}.NETWORK")
for d, g in [(0.25, 0.5), (1.6, 0.6), (3.2, 0.5)]:
    fx(syn.pop(int(d * 10)), cta_start + d, g)
fx(syn.ding(), cta_start + 4.3, 0.4)

mix = mix + sfx
mix = mix / (np.max(np.abs(mix)) or 1) * 0.95
mix = mix[: int(TOTAL * SR)]
sf.write(f"{OUT}/mix.wav", mix, SR)

# ------------------------------------------------------------------ Subtítulos
caps = []
tw_sorted = sorted([w for w in timeline_words], key=lambda x: x[0])
items = []
for s, e, txt, src in tw_sorted:
    key = round(src, 2)
    fixed = FIX.get(key, None)
    if fixed == "":
        continue
    word = fixed if fixed else clean(txt)
    if word:
        items.append([s, e, word, txt.strip()[-1:] in ".,?!"])

chunks, cur = [], []
for it in items:
    if cur and (len(cur) >= 3 or len(" ".join(c[2] for c in cur + [it])) > 16
                or it[0] - cur[-1][1] > 0.4 or cur[-1][3]):
        chunks.append(cur)
        cur = []
    cur.append(it)
if cur:
    chunks.append(cur)

for k, ch in enumerate(chunks):
    st = ch[0][0]
    en = ch[-1][1] + 0.25
    if k + 1 < len(chunks):
        en = min(en, chunks[k + 1][0][0])
    if st < cta_start:
        en = min(en, cta_start)
    txt = " ".join((r"{\c" + GOLD + "}" + c[2] + r"{\c" + WHITE + "}") if c[2] in KEYWORDS else c[2] for c in ch)
    style = "CapHook" if st < Lh else "Cap"
    caps.append(f"Dialogue: 1,{ass_t(st)},{ass_t(en)},{style},,0,0,0,,"
                r"{\fscx112\fscy112\t(0,90,\fscx100\fscy100)}" + txt)

header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Barlow Condensed,123,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,800,-1,0,0,100,100,2,0,1,7,4,2,70,70,560,1
Style: CapHook,Barlow Condensed,145,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,800,-1,0,0,100,100,2,0,1,8,4,2,60,60,560,1
Style: Logo,Barlow Condensed,55,&H00FFFFFF,&H00FFFFFF,&H0000FFD4,&H00000000,800,-1,0,0,100,100,3,0,1,0,2,8,40,40,95,1
Style: LogoBig,Barlow Condensed,106,&H00FFFFFF,&H00FFFFFF,&H0000FFD4,&H00000000,800,-1,0,0,100,100,3,0,1,0,0,5,40,40,0,1
Style: Headline,Barlow Condensed,114,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,800,-1,0,0,100,100,2,0,1,7,4,5,40,40,0,1
Style: BoxGold,Barlow Condensed,92,&H00101410,&H00101410,&H0000FFD4,&H00000000,800,-1,0,0,100,100,2,0,3,18,0,5,40,40,0,1
Style: Place,Barlow Condensed,167,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,800,-1,0,0,100,100,6,0,1,6,5,5,40,40,0,1
Style: PlaceSub,Barlow Condensed,56,{GOLD},{GOLD},&H00000000,&H90000000,800,-1,0,0,100,100,5,0,1,4,3,5,40,40,0,1
Style: Big,Barlow Condensed,185,{GOLD},{GOLD},&H00000000,&H90000000,800,-1,0,0,100,100,3,0,1,8,5,5,40,40,0,1
Style: Num,Barlow Condensed,334,{GOLD},{GOLD},&H00000000,&H90000000,800,-1,0,0,100,100,0,0,1,10,6,5,40,40,0,1
Style: NumSub,Barlow Condensed,81,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,800,-1,0,0,100,100,2,0,1,6,3,5,40,40,0,1
Style: CtaBig,Barlow Condensed,132,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,800,-1,0,0,100,100,2,0,1,7,4,5,40,40,0,1
Style: CtaCountry,Barlow Condensed,106,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,800,-1,0,0,100,100,4,0,1,6,4,5,40,40,0,1
Style: Dim,Arial,20,&H70000000,&H70000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
logo = (f"Dialogue: 3,{ass_t(0)},{ass_t(cta_start)},Logo,,0,0,0,,"
        r"JOE{\c&H0000FFD4&}.NETWORK")
with open(f"{OUT}/subs.ass", "w") as f:
    f.write(header + "\n".join([logo] + graphics + caps) + "\n")

# ------------------------------------------------------------------ Render final
print("Render final…")
final = f"{OUT}/reel_cristo_rey_joenetwork.mp4"
run(["ffmpeg", "-y", "-v", "error", "-i", joined, "-i", f"{OUT}/mix.wav",
     "-f", "lavfi", "-i", f"color=c=0xD4FF00:s=1080x12:r={FPS}:d={TOTAL:.3f}",
     "-filter_complex",
     f"[0][2]overlay=x='-W+W*t/{TOTAL:.3f}':y=H-12:shortest=1[b];[b]subtitles={OUT}/subs.ass:fontsdir={FONTS}[v];"
     "[1]loudnorm=I=-14:TP=-1.5:LRA=11[a]",
     "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
     "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-t", f"{TOTAL:.3f}", final])
print(f"OK {final}  duración={TOTAL:.1f}s  hook={Lh:.1f}s  cta_inicio={cta_start:.1f}s")
