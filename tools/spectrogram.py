import wave
import numpy as np
import zlib, struct

SR = 24000
p = r"C:\Users\oxsta\Desktop\custom voicechanger\out\clip\a2.wav"
with wave.open(p, "rb") as w:
    data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0

t0, t1 = 3.60, 9.45
seg = data[int(t0 * SR): int(t1 * SR)]

NFFT, HOP = 512, 128
win = np.hanning(NFFT).astype(np.float32)
n = 1 + (len(seg) - NFFT) // HOP
idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
S = np.abs(np.fft.rfft(seg[idx] * win, axis=1))
lo, hi = 1, 257  # 0 .. 12 kHz in 256 steps
S = S[:, lo:hi].T                       # (freq, time)
db = 20 * np.log10(S + 1e-7)
db -= db.max()
img = np.clip((db + 60.0) / 60.0 * 255, 0, 255).astype(np.uint8)
img = np.flipud(img)                    # low freq at bottom
H, W = img.shape
SCALE = 4
img = np.repeat(np.repeat(img, SCALE, axis=0), SCALE, axis=1)
H, W = img.shape
print("image", W, "x", H, "covering", t0, "-", t1, "s ;", SR, "Hz")

# grid: 1 s vertical lines, 2 kHz horizontal lines
for s in range(int(t0) + 1, int(t1) + 1):
    x = int((s - t0) * SCALE * SR / HOP)
    if 0 <= x < W:
        img[:, x] = 255
for khz in range(2, 12, 2):
    y = H - 1 - int(khz * 1000 / (SR / 2) * 256 * SCALE)
    if 0 <= y < H:
        img[y, :] = 0

def write_png(path, gray):
    h, w = gray.shape
    raw = b"".join(b"\x00" + gray[y].tobytes() for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    out = b"\x89PNG\r\n\x1a\n"
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    out += chunk(b"IDAT", zlib.compress(raw, 6))
    out += chunk(b"IEND", b"")
    open(path, "wb").write(out)

write_png(r"C:\Users\oxsta\Desktop\custom voicechanger\out\clip\a2_spec_grid.png", img)

# per-second band energies, to see where the energy actually is
print("\nper 0.5s band energy (dB rel. full scale):")
bands = [(0, 300), (300, 1000), (1000, 3000), (3000, 6000), (6000, 12000)]
f = np.fft.rfftfreq(NFFT, 1 / SR)
print("  time   " + "".join(f"{a/1000:>4.0f}-{b/1000:<3.0f}k" for a, b in bands) + "   rms")
for k in range(int((t1 - t0) * 2)):
    a, b = int(k * 0.5 * SR), int((k + 0.5) * 0.5 * SR)
    x = seg[a:b]
    if len(x) < NFFT:
        continue
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    ff = np.fft.rfftfreq(len(x), 1 / SR)
    row = ""
    for lo_b, hi_b in bands:
        e = np.sqrt((X[(ff >= lo_b) & (ff < hi_b)] ** 2).mean())
        row += f"{20*np.log10(e + 1e-9):8.1f}"
    print(f"  {t0 + k*0.5:5.2f}s {row}   {np.sqrt((x**2).mean()):.4f}")
