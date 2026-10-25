#!/usr/bin/env python3
"""Cross-validates ipf against independent implementations (Pillow, numpy, scipy).

Usage: python run_crossval.py <path/to/ipf_crossval[.exe]> [section ...]

Each check runs one ipf operation through the ipf_crossval tool and compares the
result with a reference computed in Python. Exit status is non-zero if any check fails.
"""
import io
import os
import struct
import subprocess
import sys
import tempfile
import zlib
import colorsys

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageEnhance, ImageOps

TOOL = os.path.abspath(sys.argv[1])
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SECTIONS = sys.argv[2:]
TMP = tempfile.mkdtemp(prefix="ipf_cv_")
RESULTS = []


def run(*args):
    p = subprocess.run([TOOL, *map(str, args)], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ipf_crossval {' '.join(map(str, args))} failed: {p.stderr.strip()}")


def tmp(name):
    return os.path.join(TMP, name)


def report(section, name, ok, detail=""):
    RESULTS.append((section, name, bool(ok), detail))
    print(f"  [{'ok' if ok else 'FAIL'}] {section}: {name} {detail}")


def natural_image(h, w, c, seed=0):
    """Smooth gradients + texture + hard edges + mild noise: exercises codecs and filters."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    chans = []
    for k in range(c):
        v = 128 + 70 * np.sin(x / (7 + 3 * k) + k) * np.cos(y / (9 + 2 * k))
        v += 40 * ((x // 11 + y // 13 + k) % 2)  # checker edges
        v += rng.normal(0, 6, size=(h, w))
        chans.append(v)
    a = np.clip(np.stack(chans, -1), 0, 255).round().astype(np.uint8)
    return a[..., 0] if c == 1 else a


def ipf_decode(data_or_path, name, sixteen=False):
    if isinstance(data_or_path, (bytes, bytearray)):
        p = tmp(name)
        with open(p, "wb") as f:
            f.write(data_or_path)
    else:
        p = data_or_path
    out = tmp(name + ".npy")
    run("decode16" if sixteen else "decode", p, out)
    return np.load(out)


def pil_bytes(img, fmt, **kw):
    b = io.BytesIO()
    img.save(b, fmt, **kw)
    return b.getvalue()


def squeeze(a):
    return a[..., 0] if a.ndim == 3 and a.shape[2] == 1 else a


def compare_exact(section, name, got, want):
    got, want = squeeze(np.asarray(got)), squeeze(np.asarray(want))
    if got.shape != want.shape:
        report(section, name, False, f"shape {got.shape} != {want.shape}")
        return
    diff = np.abs(got.astype(np.int64) - want.astype(np.int64))
    report(section, name, diff.max() == 0, f"(max diff {diff.max()})")


def compare_close(section, name, got, want, max_tol, mean_tol):
    got, want = squeeze(np.asarray(got)), squeeze(np.asarray(want))
    if got.shape != want.shape:
        report(section, name, False, f"shape {got.shape} != {want.shape}")
        return
    diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
    ok = diff.max() <= max_tol and diff.mean() <= mean_tol
    report(section, name, ok, f"(max {diff.max():.4g} <= {max_tol}, mean {diff.mean():.4g} <= {mean_tol})")


# ---------------------------------------------------------------------------
# Independent PNG writer (for interlaced / low bit depth / 16-bit color files)
# ---------------------------------------------------------------------------

def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _filter(row, prev, ftype, bpp):
    out = bytearray(len(row))
    for i in range(len(row)):
        a = row[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        pred = [0, a, b, (a + b) >> 1, _paeth(a, b, c)][ftype]
        out[i] = (row[i] - pred) & 0xFF
    return bytes(out)


def _pack(samples, depth):
    samples = np.asarray(samples).ravel()
    if depth == 8:
        return samples.astype(np.uint8).tobytes()
    if depth == 16:
        return samples.astype(">u2").tobytes()
    per = 8 // depth
    out = bytearray((len(samples) + per - 1) // per)
    for i, s in enumerate(samples):
        out[i // per] |= int(s) << (8 - depth - (i % per) * depth)
    return bytes(out)


def write_png(arr, depth, ctype, interlace=False, plte=None, trns=None):
    """arr: (h, w, spp) native sample values."""
    h, w, spp = arr.shape
    bpp = max(1, spp * depth // 8)
    passes = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
    if not interlace:
        passes = [(0, 0, 1, 1)]
    raw = bytearray()
    ftype = 0
    for x0, y0, dx, dy in passes:
        sub = arr[y0::dy, x0::dx]
        if sub.size == 0:
            continue
        prev = bytes(len(_pack(sub[0], depth)))
        for row in sub:
            packed = _pack(row, depth)
            raw.append(ftype)
            raw += _filter(packed, prev, ftype, bpp)
            prev = packed
            ftype = (ftype + 1) % 5  # cycle through all filter types

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, int(interlace)))
    if plte is not None:
        out += chunk(b"PLTE", bytes(np.asarray(plte, np.uint8).ravel()))
    if trns is not None:
        out += chunk(b"tRNS", trns)
    z = zlib.compress(bytes(raw), 9)
    out += chunk(b"IDAT", z[: len(z) // 2]) + chunk(b"IDAT", z[len(z) // 2:])  # split IDAT
    return out + chunk(b"IEND", b"")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_zlib():
    s = "zlib"
    rng = np.random.default_rng(1)
    data = [b"", b"a", bytes(rng.integers(0, 256, 70000, dtype=np.uint8)),
            b"".join(b"token%d " % (i % 97) for i in range(20000)), bytes(200000)]
    for i, d in enumerate(data):
        for level in (0, 1, 6, 9):
            for strategy in (zlib.Z_DEFAULT_STRATEGY, zlib.Z_FIXED, zlib.Z_HUFFMAN_ONLY, zlib.Z_RLE):
                co = zlib.compressobj(level, zlib.DEFLATED, 15, 9, strategy)
                z = co.compress(d) + co.flush()
                with open(tmp("z.in"), "wb") as f:
                    f.write(z)
                run("zinflate", tmp("z.in"), tmp("z.out"))
                ok = open(tmp("z.out"), "rb").read() == d
                if not ok or (level == 6 and strategy == zlib.Z_DEFAULT_STRATEGY):
                    report(s, f"inflate python stream #{i} level {level} strategy {strategy}", ok)
        for level in (0, 1, 6, 9):
            with open(tmp("d.in"), "wb") as f:
                f.write(d)
            run("zdeflate", tmp("d.in"), tmp("d.out"), level)
            z = open(tmp("d.out"), "rb").read()
            ok = zlib.decompress(z) == d
            ratio = len(z) / max(1, len(zlib.compress(d, level)))
            report(s, f"deflate #{i} level {level} -> python zlib", ok, f"(size vs zlib: {ratio:.2f}x)")


def section_png():
    s = "png"
    rgb = natural_image(29, 37, 3, 1)
    gray = natural_image(29, 37, 1, 2)
    alpha = natural_image(29, 37, 1, 3)
    # Pillow-written files
    cases = {
        "L": Image.fromarray(gray, "L"),
        "LA": Image.fromarray(np.dstack([gray, alpha]), "LA"),
        "RGB": Image.fromarray(rgb, "RGB"),
        "RGBA": Image.fromarray(np.dstack([rgb, alpha]), "RGBA"),
        "1": Image.fromarray(gray, "L").convert("1"),
        "P": Image.fromarray(rgb, "RGB").quantize(64),
        "P4bit": Image.fromarray(rgb, "RGB").quantize(16),
    }
    for name, im in cases.items():
        for opt in (False, True):
            data = pil_bytes(im, "PNG", optimize=opt)
            want_mode = {"1": "L", "P": "RGB", "P4bit": "RGB"}.get(name, name)
            compare_exact(s, f"decode Pillow {name} optimize={opt}", ipf_decode(data, "p.png"),
                          np.array(Image.open(io.BytesIO(data)).convert(want_mode)))
    # palette with transparency -> RGBA
    p = Image.fromarray(rgb, "RGB").quantize(32)
    trns = bytes([(i * 37) % 256 for i in range(32)])
    data = pil_bytes(p, "PNG", transparency=trns)
    compare_exact(s, "decode palette + tRNS", ipf_decode(data, "p.png"),
                  np.array(Image.open(io.BytesIO(data)).convert("RGBA")))
    # 16-bit gray (Pillow I;16)
    g16 = (natural_image(29, 37, 1, 4).astype(np.uint16) * 257 + np.arange(37, dtype=np.uint16)).astype(np.uint16)
    data = pil_bytes(Image.fromarray(g16), "PNG")
    compare_exact(s, "decode16 Pillow I;16", ipf_decode(data, "p16.png", sixteen=True), g16)
    # Independent writer: interlace, bit depths, 16-bit color, color keys, all filters
    rng = np.random.default_rng(5)
    for depth in (1, 2, 4, 8, 16):
        for interlace in (False, True):
            maxv = (1 << depth) - 1
            arr = rng.integers(0, maxv + 1, (23, 19, 1))
            want = (arr[..., 0] * (255 // maxv)) if depth < 8 else (arr[..., 0] if depth == 8 else arr[..., 0] >> 8)
            got = ipf_decode(write_png(arr, depth, 0, interlace), "g.png")
            if depth == 16:  # ipf rounds 16->8 bit: (v * 255 + 32767) / 65535
                want = np.floor(arr[..., 0] * 255.0 / 65535.0 + 0.5)
            compare_exact(s, f"decode gray depth={depth} interlace={interlace}", got, want)
            if depth == 16:
                compare_exact(s, f"decode16 gray interlace={interlace}",
                              ipf_decode(write_png(arr, 16, 0, interlace), "g16.png", True), arr[..., 0])
    for depth in (1, 2, 4, 8):
        pal = rng.integers(0, 256, (1 << depth, 3))
        idx = rng.integers(0, 1 << depth, (21, 26, 1))
        for interlace in (False, True):
            compare_exact(s, f"decode palette depth={depth} interlace={interlace}",
                          ipf_decode(write_png(idx, depth, 3, interlace, plte=pal), "pal.png"), pal[idx[..., 0]])
    for ctype, spp in ((2, 3), (4, 2), (6, 4)):
        for depth in (8, 16):
            arr = rng.integers(0, (1 << depth), (17, 22, spp))
            for interlace in (False, True):
                got = ipf_decode(write_png(arr, depth, ctype, interlace), "c.png", sixteen=(depth == 16))
                compare_exact(s, f"decode ctype={ctype} depth={depth} interlace={interlace}", got, arr)
    # color-key transparency
    arr = rng.integers(0, 4, (9, 11, 3)) * 80
    key = struct.pack(">HHH", 80, 160, 0)
    got = ipf_decode(write_png(arr, 8, 2, trns=key), "k.png")
    want = np.dstack([arr, np.where((arr == [80, 160, 0]).all(-1), 0, 255)])
    compare_exact(s, "decode RGB + tRNS color key", got, want)
    # ipf encoder -> Pillow
    for c, mode in ((1, "L"), (2, "LA"), (3, "RGB"), (4, "RGBA")):
        a = natural_image(31, 45, c, 10 + c)
        np.save(tmp("e.npy"), a)
        run("encode", tmp("e.npy"), tmp("e.png"))
        im = Image.open(tmp("e.png"))
        compare_exact(s, f"Pillow decodes ipf {mode} PNG", np.array(im.convert(mode)), a)
    a16 = rng.integers(0, 65536, (13, 17)).astype(np.uint16)
    np.save(tmp("e16.npy"), a16)
    run("encode", tmp("e16.npy"), tmp("e16.png"))
    compare_exact(s, "Pillow decodes ipf 16-bit PNG", np.array(Image.open(tmp("e16.png"))), a16)
    big = natural_image(256, 256, 3, 7)
    np.save(tmp("big.npy"), big)
    run("encode", tmp("big.npy"), tmp("big.png"))
    ours = os.path.getsize(tmp("big.png"))
    theirs = len(pil_bytes(Image.fromarray(big), "PNG"))
    report(s, "compression vs Pillow", ours < theirs * 1.25, f"(ipf {ours} B, Pillow {theirs} B)")


def section_jpeg():
    s = "jpeg"
    rgb = natural_image(61, 83, 3, 21)
    gray = natural_image(61, 83, 1, 22)
    variants = [
        ("q75 4:2:0", dict(quality=75, subsampling=2)),
        ("q95 4:4:4", dict(quality=95, subsampling=0)),
        ("q90 4:2:2", dict(quality=90, subsampling=1)),
        ("progressive 4:2:0", dict(quality=85, progressive=True)),
        ("progressive 4:4:4", dict(quality=90, progressive=True, subsampling=0)),
        ("optimized huffman", dict(quality=80, optimize=True)),
        ("restart markers", dict(quality=80, restart_marker_blocks=3)),
        ("progressive + restart", dict(quality=80, progressive=True, restart_marker_rows=1)),
    ]
    for name, kw in variants:
        data = pil_bytes(Image.fromarray(rgb), "JPEG", **kw)
        compare_close(s, f"decode Pillow {name}", ipf_decode(data, "j.jpg"),
                      np.array(Image.open(io.BytesIO(data)).convert("RGB")), max_tol=4, mean_tol=0.6)
    for prog in (False, True):
        data = pil_bytes(Image.fromarray(gray, "L"), "JPEG", quality=90, progressive=prog)
        compare_close(s, f"decode Pillow gray progressive={prog}", ipf_decode(data, "j.jpg"),
                      np.array(Image.open(io.BytesIO(data))), max_tol=2, mean_tol=0.3)
    cmyk = Image.fromarray(rgb).convert("CMYK")
    data = pil_bytes(cmyk, "JPEG", quality=95)
    compare_close(s, "decode Pillow CMYK (Adobe)", ipf_decode(data, "j.jpg"),
                  np.array(Image.open(io.BytesIO(data)).convert("RGB")), max_tol=6, mean_tol=1.0)
    odd = natural_image(17, 9, 3, 3)
    data = pil_bytes(Image.fromarray(odd), "JPEG", quality=90)
    compare_close(s, "decode Pillow odd size 9x17", ipf_decode(data, "j.jpg"),
                  np.array(Image.open(io.BytesIO(data))), max_tol=4, mean_tol=0.8)
    # ipf encoder -> Pillow
    for sub in (1, 0):
        np.save(tmp("e.npy"), rgb)
        run("encode", tmp("e.npy"), tmp("e.jpg"), 90, sub)
        pil = np.array(Image.open(tmp("e.jpg")).convert("RGB")).astype(np.float64)
        psnr = 10 * np.log10(255 ** 2 / np.mean((pil - rgb) ** 2))
        ref = np.array(Image.open(io.BytesIO(pil_bytes(Image.fromarray(rgb), "JPEG", quality=90,
                                                       subsampling=2 if sub else 0)))).astype(np.float64)
        ref_psnr = 10 * np.log10(255 ** 2 / np.mean((ref - rgb) ** 2))
        report(s, f"ipf JPEG quality vs Pillow encoder subsample={sub}", psnr > ref_psnr - 0.5,
               f"(ipf PSNR {psnr:.2f} dB, Pillow {ref_psnr:.2f} dB)")
        compare_close(s, f"ipf vs Pillow decoding ipf JPEG subsample={sub}", ipf_decode(tmp("e.jpg"), "e2"), pil,
                      max_tol=4, mean_tol=0.6)
        ours = os.path.getsize(tmp("e.jpg"))
        theirs = len(pil_bytes(Image.fromarray(rgb), "JPEG", quality=90, subsampling=2 if sub else 0))
        report(s, f"file size vs Pillow subsample={sub}", ours < theirs * 1.2, f"(ipf {ours} B, Pillow {theirs} B)")
    np.save(tmp("g.npy"), gray)
    run("encode", tmp("g.npy"), tmp("g.jpg"), 90)
    im = Image.open(tmp("g.jpg"))
    report(s, "Pillow decodes ipf gray JPEG", im.mode == "L" and
           10 * np.log10(255 ** 2 / np.mean((np.array(im).astype(float) - gray) ** 2)) > 33)


def section_bmp_tga_pnm():
    s = "bmp/tga/pnm"
    rgb = natural_image(19, 23, 3, 31)
    gray = natural_image(19, 23, 1, 32)
    alpha = natural_image(19, 23, 1, 33)
    rgba = np.dstack([rgb, alpha])
    for mode, im in (("1", Image.fromarray(gray).convert("1")), ("L", Image.fromarray(gray)),
                     ("P", Image.fromarray(rgb).quantize(50)), ("RGB", Image.fromarray(rgb)),
                     ("RGBA", Image.fromarray(rgba, "RGBA"))):
        data = pil_bytes(im, "BMP")
        ref = Image.open(io.BytesIO(data))
        want = np.array(ref.convert({"1": "L", "P": "RGB"}.get(mode, ref.mode)))
        if mode == "RGBA":  # Pillow ignores the alpha bytes it wrote; ipf keeps them
            want = rgba
        compare_exact(s, f"decode Pillow BMP {mode}", ipf_decode(data, "b.bmp"), want)
    for mode, im in (("L", Image.fromarray(gray)), ("LA", Image.fromarray(np.dstack([gray, alpha]), "LA")),
                     ("RGB", Image.fromarray(rgb)), ("RGBA", Image.fromarray(rgba, "RGBA")),
                     ("P", Image.fromarray(rgb).quantize(40))):
        for comp in (None, "tga_rle"):
            data = pil_bytes(im, "TGA", compression=comp) if comp else pil_bytes(im, "TGA")
            want = np.array(Image.open(io.BytesIO(data)).convert("RGB" if mode == "P" else mode))
            compare_exact(s, f"decode Pillow TGA {mode} {comp or 'raw'}", ipf_decode(data, "t.tga"), want)
    for mode, im in (("1", Image.fromarray(gray).convert("1")), ("L", Image.fromarray(gray)),
                     ("RGB", Image.fromarray(rgb))):
        data = pil_bytes(im, "PPM")
        compare_exact(s, f"decode Pillow PNM {mode}", ipf_decode(data, "n.pnm"),
                      np.array(Image.open(io.BytesIO(data)).convert("L" if mode == "1" else mode)))
    for c, mode in ((1, "L"), (3, "RGB"), (4, "RGBA")):
        a = {1: gray, 3: rgb, 4: rgba}[c]
        np.save(tmp("e.npy"), a)
        run("encode", tmp("e.npy"), tmp("e.bmp"))
        compare_exact(s, f"Pillow decodes ipf BMP {mode}", np.array(Image.open(tmp("e.bmp")).convert(mode)), a)
        for rle in (0, 1):
            run("encode", tmp("e.npy"), tmp("e.tga"), rle)
            compare_exact(s, f"Pillow decodes ipf TGA {mode} rle={rle}",
                          np.array(Image.open(tmp("e.tga")).convert(mode)), a)
    for c, mode in ((1, "L"), (3, "RGB")):
        np.save(tmp("e.npy"), {1: gray, 3: rgb}[c])
        run("encode", tmp("e.npy"), tmp("e.ppm"))
        compare_exact(s, f"Pillow decodes ipf PNM {mode}", np.array(Image.open(tmp("e.ppm"))), {1: gray, 3: rgb}[c])


SCIPY_MODE = {"constant": "constant", "replicate": "nearest", "reflect": "reflect", "reflect101": "mirror",
              "wrap": "wrap"}


def ipf_op(cmd, first, *rest):
    """Runs `ipf_crossval cmd <first.npy> <out.npy> rest...`; numpy arguments are saved to .npy files."""
    args = []
    for i, a in enumerate((first,) + rest):
        if isinstance(a, np.ndarray):
            p = tmp(f"arg{i}.npy")
            np.save(p, a)
            args.append(p)
        else:
            args.append(a)
    out = tmp("out.npy")
    run(cmd, args[0], out, *args[1:])
    return np.load(out)


def per_channel(fn, a):
    return fn(a) if a.ndim == 2 else np.stack([fn(a[..., c]) for c in range(a.shape[2])], -1)


def section_color():
    s = "color"
    rgb = natural_image(30, 40, 3, 41)
    compare_close(s, "rgbToGray vs Pillow L", ipf_op("gray", rgb), np.array(Image.fromarray(rgb).convert("L")), 1, 0.05)
    r, g, b = (rgb[..., k].astype(np.float64) for k in range(3))
    exact = np.stack([0.299 * r + 0.587 * g + 0.114 * b, 128 - 0.168735892 * r - 0.331264108 * g + 0.5 * b,
                      128 + 0.5 * r - 0.418687589 * g - 0.081312411 * b], -1)
    compare_exact(s, "rgbToYCbCr vs exact JFIF formula", ipf_op("ycbcr", rgb), np.clip(np.floor(exact + 0.5), 0, 255))
    # Pillow uses a fixed-point approximation that is off by up to 1
    compare_close(s, "rgbToYCbCr vs Pillow YCbCr", ipf_op("ycbcr", rgb),
                  np.array(Image.fromarray(rgb).convert("YCbCr")), 1, 0.6)
    ycc = np.array(Image.fromarray(rgb).convert("YCbCr")).astype(np.float64)
    y, cb, cr = ycc[..., 0], ycc[..., 1] - 128, ycc[..., 2] - 128
    inv = np.stack([y + 1.402 * cr, y - 0.344136286 * cb - 0.714136286 * cr, y + 1.772 * cb], -1)
    compare_exact(s, "yCbCrToRgb vs exact JFIF formula", ipf_op("ycbcr_inv", ycc.astype(np.uint8)),
                  np.clip(np.floor(inv + 0.5), 0, 255))
    flat = rgb.reshape(-1, 3) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in flat]).reshape(rgb.shape)
    hsv[..., 0] *= 360
    got = ipf_op("hsv", rgb)
    dh = np.abs(got[..., 0] - hsv[..., 0])
    dh = np.minimum(dh, 360 - dh)
    report(s, "rgbToHsv vs colorsys", dh.max() < 1e-3 and np.abs(got[..., 1:] - hsv[..., 1:]).max() < 1e-6,
           f"(max hue err {dh.max():.2g})")
    hls = np.array([colorsys.rgb_to_hls(*p) for p in flat]).reshape(rgb.shape)
    want = np.stack([hls[..., 0] * 360, hls[..., 2], hls[..., 1]], -1)
    got = ipf_op("hsl", rgb)
    dh = np.abs(got[..., 0] - want[..., 0])
    dh = np.minimum(dh, 360 - dh)
    report(s, "rgbToHsl vs colorsys", dh.max() < 1e-3 and np.abs(got[..., 1:] - want[..., 1:]).max() < 1e-6,
           f"(max hue err {dh.max():.2g})")
    # CIE Lab: independent numpy implementation
    c = rgb / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    lab = np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)
    compare_close(s, "rgbToLab vs numpy reference", ipf_op("lab", rgb), lab, 1e-3, 1e-4)
    for fac in (0.0, 0.5, 1.7):
        compare_close(s, f"adjustSaturation({fac}) vs Pillow ImageEnhance.Color", ipf_op("saturation", rgb, fac),
                      np.array(ImageEnhance.Color(Image.fromarray(rgb)).enhance(fac)), 2, 0.6)


def section_pixel():
    s = "pixel"
    rgb = natural_image(30, 40, 3, 51)
    low = (natural_image(30, 40, 1, 52) // 3 + 60).astype(np.uint8)
    for cut in (0, 2, 7.5):
        compare_exact(s, f"autoContrast(cutoff={cut}) vs Pillow", ipf_op("autocontrast", rgb, cut),
                      np.array(ImageOps.autocontrast(Image.fromarray(rgb), cutoff=cut)))
    h = np.bincount(low.ravel(), minlength=256)
    first = np.nonzero(h)[0][0]
    lut = np.zeros(256)
    lut[first + 1:] = np.cumsum(h[first + 1:]) * (255.0 / (low.size - h[first]))
    compare_exact(s, "equalizeHist vs OpenCV formula", ipf_op("equalize", low),
                  np.clip(np.floor(lut + 0.5), 0, 255)[low])
    p = h / h.sum()  # Otsu by brute-force maximization of the between-class variance
    best, bt = -1, 0
    for t in range(256):
        w0, w1 = p[: t + 1].sum(), p[t + 1:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (np.arange(t + 1) * p[: t + 1]).sum() / w0
        m1 = (np.arange(t + 1, 256) * p[t + 1:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best + 1e-15:
            best, bt = v, t
    report(s, "otsuThreshold vs brute force", int(ipf_op("otsu", low).ravel()[0]) == bt, f"(t={bt})")
    for block, cc in ((11, 2), (25, -3)):
        mean = ndi.uniform_filter(low.astype(np.float64), size=block, mode="nearest")
        want = np.where(low.astype(np.float64) - np.floor(mean + 0.5) > -cc, 255, 0)
        compare_exact(s, f"adaptiveThreshold mean block={block} C={cc}", ipf_op("adaptive", low, 0, block, cc), want)


def section_filter():
    s = "filter"
    rng = np.random.default_rng(61)
    img = natural_image(27, 33, 1, 62).astype(np.float32)
    img3 = natural_image(27, 33, 3, 63).astype(np.float32)
    kernel = rng.normal(size=(3, 5)).astype(np.float32)  # 3 rows x 5 columns
    np.save(tmp("k.npy"), kernel)
    for mode, smode in SCIPY_MODE.items():
        for name, a in (("gray", img), ("rgb", img3)):
            want = per_channel(lambda ch: ndi.correlate(ch.astype(np.float64), kernel.astype(np.float64), mode=smode,
                                                        cval=7.0), a)
            np.save(tmp("in.npy"), a)
            run("filter2d", tmp("in.npy"), tmp("k.npy"), tmp("o.npy"), mode, 7.0)
            compare_close(s, f"filter2D {name} border={mode} vs scipy.correlate", np.load(tmp("o.npy")), want,
                          2e-3, 1e-4)
        kx, ky = rng.normal(size=7), rng.normal(size=3)
        want = ndi.correlate1d(ndi.correlate1d(img.astype(np.float64), kx, axis=1, mode=smode), ky, axis=0, mode=smode)
        np.save(tmp("kx.npy"), kx)
        np.save(tmp("ky.npy"), ky)
        np.save(tmp("in.npy"), img)
        run("sep", tmp("in.npy"), tmp("kx.npy"), tmp("ky.npy"), tmp("o.npy"), mode)
        compare_close(s, f"sepFilter2D border={mode} vs scipy.correlate1d", np.load(tmp("o.npy")), want, 2e-3, 1e-4)
        if mode == "constant":
            continue
        want = per_channel(lambda ch: ndi.uniform_filter(ch.astype(np.float64), size=(3, 7), mode=smode), img3)
        compare_close(s, f"boxFilter 7x3 border={mode} vs scipy.uniform_filter", ipf_op("box", img3, 7, 3, mode),
                      want, 2e-3, 1e-4)
        for sx, sy in ((1.0, 0), (2.3, 0.8)):
            want = per_channel(lambda ch: ndi.gaussian_filter(ch.astype(np.float64), sigma=(sy or sx, sx),
                                                              truncate=3.0, mode=smode), img3)
            compare_close(s, f"gaussianBlur sigma=({sx},{sy or sx}) border={mode} vs scipy.gaussian_filter",
                          ipf_op("gauss", img3, sx, sy, mode), want, 2e-3, 1e-4)
    u8 = natural_image(27, 33, 3, 64)
    want = per_channel(lambda ch: ndi.gaussian_filter(ch.astype(np.float64), 1.5, truncate=3.0, mode="mirror"), u8)
    compare_close(s, "gaussianBlur uint8 rounding", ipf_op("gauss_u8", u8, 1.5), np.floor(want + 0.5), 1, 0.01)
    for k in (3, 5, 7, 11):
        for mode in ("replicate", "reflect", "reflect101", "wrap"):
            want = per_channel(lambda ch: ndi.median_filter(ch, size=k, mode=SCIPY_MODE[mode]), u8)
            compare_exact(s, f"medianBlur k={k} border={mode} vs scipy.median_filter", ipf_op("median", u8, k, mode),
                          want)
    compare_exact(s, "medianBlur float k=5", ipf_op("median_f", img, 5, "replicate"),
                  ndi.median_filter(img, size=5, mode="nearest"))
    # derivative filters vs explicit OpenCV kernels
    smooth = {1: [1.0], 3: [1, 2, 1], 5: [1, 4, 6, 4, 1]}
    deriv = {1: [-1, 0, 1], 3: [-1, 0, 1], 5: [-1, -2, 0, 2, 1]}
    d2 = {1: [1, -2, 1], 3: [1, -2, 1], 5: [1, 0, -2, 0, 1]}
    f64 = img.astype(np.float64)
    for ksize in (1, 3, 5):
        for dx, dy in ((1, 0), (0, 1), (2, 0), (1, 1)):
            kx = {0: smooth[ksize], 1: deriv[ksize], 2: d2[ksize]}[dx]
            ky = {0: smooth[ksize], 1: deriv[ksize], 2: d2[ksize]}[dy]
            want = ndi.correlate1d(ndi.correlate1d(f64, kx, axis=1, mode="mirror"), ky, axis=0, mode="mirror")
            compare_close(s, f"sobel ksize={ksize} dx={dx} dy={dy}", ipf_op("sobel", img, dx, dy, ksize, "reflect101"),
                          want, 2e-2, 1e-3)
    want = ndi.correlate1d(ndi.correlate1d(f64, [-1, 0, 1], axis=1, mode="mirror"), [3, 10, 3], axis=0, mode="mirror")
    compare_close(s, "scharr dx", ipf_op("scharr", img, 1, 0), want, 2e-2, 1e-3)
    want = ndi.correlate(f64, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float), mode="mirror")
    compare_close(s, "laplacian ksize=1", ipf_op("laplacian", img, 1), want, 1e-2, 1e-3)
    want = (ndi.correlate1d(ndi.correlate1d(f64, [1, -2, 1], axis=1, mode="mirror"), [1, 2, 1], axis=0, mode="mirror") +
            ndi.correlate1d(ndi.correlate1d(f64, [1, 2, 1], axis=1, mode="mirror"), [1, -2, 1], axis=0, mode="mirror"))
    compare_close(s, "laplacian ksize=3", ipf_op("laplacian", img, 3), want, 2e-2, 1e-3)
    # bilateral: brute-force numpy reference (OpenCV definition, L1 color distance)
    for d, sc, ss in ((5, 20.0, 3.0), (9, 50.0, 4.0)):
        r = d // 2
        a = u8.astype(np.float64)
        pad = np.pad(a, ((r, r), (r, r), (0, 0)), mode="reflect")  # numpy 'reflect' == Reflect101
        acc = np.zeros_like(a)
        wsum = np.zeros(a.shape[:2])
        hh, ww = a.shape[:2]
        for j in range(-r, r + 1):
            for i in range(-r, r + 1):
                if i * i + j * j > r * r:
                    continue
                sh = pad[r + j:r + j + hh, r + i:r + i + ww]
                dist = np.abs(sh - a).sum(-1)
                w = np.exp(-(i * i + j * j) / (2 * ss * ss)) * np.exp(-dist ** 2 / (2 * sc * sc))
                acc += w[..., None] * sh
                wsum += w
        compare_close(s, f"bilateralFilter d={d} sc={sc} ss={ss}", ipf_op("bilateral", u8, d, sc, ss),
                      np.floor(acc / wsum[..., None] + 0.5), 1, 0.01)


def section_geometry():
    s = "geometry"
    img = natural_image(27, 33, 1, 71).astype(np.float32)
    rgb = natural_image(27, 33, 3, 72)
    filters = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC,
               "lanczos": Image.LANCZOS, "area": Image.BOX, "hamming": Image.HAMMING}
    for name, pf in filters.items():
        for (w, h) in ((10, 13), (70, 50), (50, 11), (33, 64)):
            want = np.array(Image.fromarray(img, "F").resize((w, h), pf))
            compare_close(s, f"resize float {name} 33x27->{w}x{h} vs Pillow", ipf_op("resize", img, w, h, name), want,
                          2e-3, 1e-4)
        got = ipf_op("resize", rgb, 19, 41, name)
        exact = np.stack([np.array(Image.fromarray(rgb[..., c].astype(np.float32), "F").resize((19, 41), pf))
                          for c in range(3)], -1)
        compare_exact(s, f"resize uint8 RGB {name} -> 19x41 vs exactly rounded result", got,
                      np.clip(np.floor(exact + 0.5), 0, 255))
        # Pillow's own uint8 path rounds between its two passes, so allow +-1 there
        compare_close(s, f"resize uint8 RGB {name} -> 19x41 vs Pillow uint8", got,
                      np.array(Image.fromarray(rgb).resize((19, 41), pf)), 1, 0.35)
    # warps vs scipy.ndimage.map_coordinates (independent sampling implementation)
    modes = {"constant": "grid-constant", "replicate": "nearest", "reflect": "reflect", "reflect101": "mirror",
             "wrap": "grid-wrap"}
    f64 = img.astype(np.float64)
    hh, ww = 31, 40
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float64)
    m = np.array([[0.83, -0.41, 9.3], [0.37, 0.91, -6.1]])
    inv = np.linalg.inv(np.vstack([m, [0, 0, 1]]))
    sx = inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2]
    sy = inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2]
    for mode, smode in modes.items():
        want = ndi.map_coordinates(f64, [sy, sx], order=1, mode=smode, cval=5.0)
        compare_close(s, f"warpAffine bilinear border={mode} vs map_coordinates",
                      ipf_op("warpaffine", img, m.ravel(), ww, hh, "bilinear", mode, 5.0), want, 2e-3, 1e-4)
    want = ndi.map_coordinates(f64, [sy, sx], order=0, mode="nearest")
    compare_close(s, "warpAffine nearest vs map_coordinates order=0",
                  ipf_op("warpaffine", img, m.ravel(), ww, hh, "nearest", "replicate"), want, 0, 0)
    hm = np.array([[0.9, 0.12, 3.0], [-0.08, 1.05, 2.0], [0.002, -0.0015, 1.0]])
    inv = np.linalg.inv(hm)
    den = inv[2, 0] * xx + inv[2, 1] * yy + inv[2, 2]
    sx = (inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2]) / den
    sy = (inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2]) / den
    for mode in ("constant", "reflect101"):
        want = ndi.map_coordinates(f64, [sy, sx], order=1, mode=modes[mode], cval=0.0)
        compare_close(s, f"warpPerspective bilinear border={mode} vs map_coordinates",
                      ipf_op("warpperspective", img, hm.ravel(), ww, hh, "bilinear", mode, 0.0), want, 2e-3, 1e-4)
    rng = np.random.default_rng(73)
    mx = (rng.uniform(-3, 36, (20, 25))).astype(np.float32)
    my = (rng.uniform(-3, 30, (20, 25))).astype(np.float32)
    np.save(tmp("mx.npy"), mx)
    np.save(tmp("my.npy"), my)
    np.save(tmp("in.npy"), img)
    run("remap", tmp("in.npy"), tmp("o.npy"), tmp("mx.npy"), tmp("my.npy"), "bilinear", "reflect")
    want = ndi.map_coordinates(f64, [my.astype(np.float64), mx.astype(np.float64)], order=1, mode="reflect")
    compare_close(s, "remap bilinear reflect vs map_coordinates", np.load(tmp("o.npy")), want, 2e-3, 1e-4)
    k = np.array([1, 4, 6, 4, 1]) / 16.0
    want = ndi.correlate1d(ndi.correlate1d(f64, k, axis=1, mode="mirror"), k, axis=0, mode="mirror")[::2, ::2]
    compare_close(s, "pyrDown vs binomial blur + subsample", ipf_op("pyrdown", img), want, 1e-3, 1e-4)


def opencv_ellipse(w, h):
    """OpenCV's getStructuringElement(MORPH_ELLIPSE) rasterization."""
    se = np.zeros((h, w), bool)
    r, c = h // 2, w // 2
    for i in range(h):
        dy = i - r
        if abs(dy) <= r:
            dx = int(np.floor(c * np.sqrt((r * r - dy * dy) / (r * r)) + 0.5)) if r else c
            se[i, max(c - dx, 0):min(c + dx + 1, w)] = True
    return se


def section_morph_analysis():
    s = "morph/analysis"
    rng = np.random.default_rng(81)
    u8 = natural_image(31, 37, 1, 82)
    for shape in ("rect", "cross", "ellipse"):
        for (w, h) in ((3, 3), (5, 7), (9, 5)):
            if shape == "rect":
                se = np.ones((h, w), bool)
            elif shape == "cross":
                se = np.zeros((h, w), bool)
                se[h // 2, :] = True
                se[:, w // 2] = True
            else:
                se = opencv_ellipse(w, h)
            want = ndi.grey_erosion(u8, footprint=se, mode="constant", cval=255)
            compare_exact(s, f"erode {shape} {w}x{h} vs scipy.grey_erosion", ipf_op("erode", u8, shape, w, h), want)
            want = ndi.grey_dilation(u8, footprint=se, mode="constant", cval=0)
            compare_exact(s, f"dilate {shape} {w}x{h} vs scipy.grey_dilation", ipf_op("dilate", u8, shape, w, h), want)
    blobs = (ndi.gaussian_filter(rng.random((60, 70)), 2.5) > 0.5).astype(np.uint8) * 255
    compare_exact(s, "fillHoles vs scipy.binary_fill_holes", ipf_op("fillholes", blobs),
                  ndi.binary_fill_holes(blobs > 0).astype(np.uint8) * 255)
    for conn, structure in ((4, None), (8, np.ones((3, 3)))):
        want, _ = ndi.label(blobs > 0, structure=structure)
        compare_exact(s, f"connectedComponents {conn}-conn vs scipy.label", ipf_op("label", blobs, conn), want)
    sparse = (rng.random((45, 52)) > 0.03).astype(np.uint8)
    compare_close(s, "distanceTransform L2 vs scipy.distance_transform_edt", ipf_op("dist", sparse, "l2"),
                  ndi.distance_transform_edt(sparse), 1e-4, 1e-5)
    compare_exact(s, "distanceTransform L1 vs scipy cdt taxicab", ipf_op("dist", sparse, "l1"),
                  ndi.distance_transform_cdt(sparse, metric="taxicab"))
    compare_exact(s, "distanceTransform chessboard vs scipy cdt", ipf_op("dist", sparse, "chess"),
                  ndi.distance_transform_cdt(sparse, metric="chessboard"))
    # moments (numpy reference)
    img = natural_image(23, 29, 1, 83).astype(np.float64)
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float64)
    raw = {(p, q): (img * x ** p * y ** q).sum() for p in range(4) for q in range(4) if p + q <= 3}
    cx, cy = raw[(1, 0)] / raw[(0, 0)], raw[(0, 1)] / raw[(0, 0)]
    mu = {(p, q): (img * (x - cx) ** p * (y - cy) ** q).sum() for p in range(4) for q in range(4) if 2 <= p + q <= 3}
    nu = {k: v / raw[(0, 0)] ** (1 + sum(k) / 2) for k, v in mu.items()}
    n20, n11, n02 = nu[(2, 0)], nu[(1, 1)], nu[(0, 2)]
    n30, n21, n12, n03 = nu[(3, 0)], nu[(2, 1)], nu[(1, 2)], nu[(0, 3)]
    hu = [n20 + n02, (n20 - n02) ** 2 + 4 * n11 ** 2, (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2,
          (n30 + n12) ** 2 + (n21 + n03) ** 2,
          (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) +
          (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
          (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03),
          (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) -
          (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)]
    order_raw = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3))
    order_mu = ((2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3))
    want = np.array([raw[k] for k in order_raw] + [mu[k] for k in order_mu] + hu)
    got = ipf_op("moments", img.astype(np.float32)).ravel()
    rel = np.abs(got - want) / np.maximum(np.abs(want), 1e-12 * np.abs(want).max())
    report(s, "moments + Hu invariants vs numpy", rel.max() < 1e-6, f"(max rel err {rel.max():.2g})")


def section_features_metrics():
    s = "features/metrics"
    rng = np.random.default_rng(91)
    img = natural_image(30, 34, 3, 92).astype(np.float64)
    tpl = img[8:17, 11:23].copy()
    th, tw = tpl.shape[:2]
    win = np.lib.stride_tricks.sliding_window_view(img, (th, tw, 3))[:, :, 0]  # (H, W, th, tw, 3)
    tp = tpl[None, None]
    sqd = ((win - tp) ** 2).sum((2, 3, 4))
    cc = (win * tp).sum((2, 3, 4))
    t2 = (tpl ** 2).sum()
    w2 = (win ** 2).sum((2, 3, 4))
    tz = tpl - tpl.mean((0, 1))
    wz = win - win.mean((2, 3), keepdims=True)
    cco = (wz * tz[None, None]).sum((2, 3, 4))
    refs = [sqd, sqd / np.sqrt(t2 * w2), cc, cc / np.sqrt(t2 * w2), cco,
            cco / np.sqrt((tz ** 2).sum() * (wz ** 2).sum((2, 3, 4)))]
    names = ["SqDiff", "SqDiffNormed", "CCorr", "CCorrNormed", "CCoeff", "CCoeffNormed"]
    np.save(tmp("tpl.npy"), tpl.astype(np.float32))
    for m, (name, want) in enumerate(zip(names, refs)):
        got = ipf_op("match", img.astype(np.float32), tmp("tpl.npy"), m)
        scale = max(1.0, np.abs(want).max())
        compare_close(s, f"matchTemplate {name} vs numpy", got / scale, want / scale, 1e-5, 1e-6)
    g = natural_image(25, 27, 1, 93)
    p = np.pad(g, 1, mode="edge").astype(np.int32)
    c = p[1:-1, 1:-1]
    offs = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    want = np.zeros_like(c)
    for k, (ox, oy) in enumerate(offs):
        want |= (p[1 + oy:p.shape[0] - 1 + oy, 1 + ox:p.shape[1] - 1 + ox] >= c).astype(np.int32) << (7 - k)
    compare_exact(s, "localBinaryPattern vs numpy", ipf_op("lbp", g), want)
    # SSIM: Wang et al. reference (Gaussian 11x11, sigma 1.5, valid region)
    a = natural_image(40, 48, 1, 94)
    b = np.clip(a.astype(int) + rng.integers(-25, 26, a.shape), 0, 255).astype(np.uint8)
    ax = np.arange(11) - 5
    k1 = np.exp(-ax ** 2 / (2 * 1.5 ** 2))
    k1 /= k1.sum()
    win2 = np.outer(k1, k1)

    def filt(z):
        return ndi.correlate(z, win2, mode="mirror")[5:-5, 5:-5]

    x, yv = a.astype(np.float64), b.astype(np.float64)
    mx, my = filt(x), filt(yv)
    vx, vy, cxy = filt(x * x) - mx ** 2, filt(yv * yv) - my ** 2, filt(x * yv) - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ref = (((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))).mean()
    np.save(tmp("b.npy"), b)
    got = ipf_op("ssim", a, tmp("b.npy")).ravel()[0]
    report(s, "ssim vs Wang et al. reference", abs(got - ref) < 1e-5, f"(ipf {got:.6f}, ref {ref:.6f})")


def section_fft_cmaps():
    s = "fft/colormaps"
    rng = np.random.default_rng(101)
    for (h, w) in ((32, 64), (37, 51), (1, 17), (90, 1)):
        a = rng.normal(size=(h, w))
        got = ipf_op("dft", a)
        want = np.fft.fft2(a)
        err = np.abs(got[..., 0] + 1j * got[..., 1] - want).max() if got.ndim == 3 else np.abs(got - want.real).max()
        report(s, f"dft {w}x{h} vs numpy.fft.fft2", err < 1e-9 * a.size, f"(max err {err:.2g})")
    # large template -> FFT-based cross term
    img = natural_image(120, 160, 1, 102).astype(np.float64)
    tpl = img[30:85, 40:110].copy()
    th, tw = tpl.shape
    win = np.lib.stride_tricks.sliding_window_view(img, (th, tw))
    sqd = ((win - tpl) ** 2).sum((2, 3))
    tz = tpl - tpl.mean()
    wz = win - win.mean((2, 3), keepdims=True)
    ccn = (wz * tz).sum((2, 3)) / np.sqrt((tz ** 2).sum() * (wz ** 2).sum((2, 3)))
    np.save(tmp("tpl.npy"), tpl.astype(np.float32))
    got = ipf_op("match", img.astype(np.float32), tmp("tpl.npy"), 0)
    compare_close(s, "matchTemplate SqDiff, 70x55 template (FFT path)", got / sqd.max(), sqd / sqd.max(), 1e-5, 1e-6)
    got = ipf_op("match", img.astype(np.float32), tmp("tpl.npy"), 5)
    compare_close(s, "matchTemplate CCoeffNormed, 70x55 template (FFT path)", got, ccn, 1e-5, 1e-6)
    # exact colormap tables vs matplotlib's source data
    import ast
    src_path = os.environ.get("IPF_CM_LISTED", r"C:\tmp\ipf_ref\cmaps\_cm_listed.py")
    if os.path.exists(src_path):
        src = open(src_path, encoding="utf-8").read()

        def table(name):
            i = src.index(f"_{name}_data = ") + len(f"_{name}_data = ")
            depth = 0
            for j in range(i, len(src)):
                depth += {"[": 1, "]": -1}.get(src[j], 0)
                if depth == 0 and src[j] == "]":
                    return np.array(ast.literal_eval(src[i:j + 1]))

        for idx, name in ((3, "viridis"), (4, "turbo"), (5, "inferno"), (6, "magma"), (7, "plasma"), (8, "cividis")):
            want = np.floor(table(name) * 255 + 0.5)
            compare_exact(s, f"colormap {name} vs matplotlib data", ipf_op("colormap", np.zeros((1, 1), np.uint8), idx)[0],
                          want)


LJT_BIN = os.environ.get("IPF_LJT_BIN", r"C:\tmp\ipf_ref\build")
LJT_TESTIMAGES = os.environ.get("IPF_LJT_TESTIMAGES", r"C:\tmp\ipf_ref\src\libjpeg-turbo-3.1.2\testimages")


def ljt(tool, *args):
    exe = os.path.join(LJT_BIN, tool + "-static.exe")
    if not os.path.exists(exe):
        exe = os.path.join(LJT_BIN, tool)
    p = subprocess.run([exe, *map(str, args)], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"{tool} {' '.join(map(str, args))}: {p.stderr.decode(errors='replace')}")
    return p.stdout


def read_pnm(path):
    """Minimal PGM/PPM reader supporting maxval up to 65535 (Pillow cannot read 16-bit PPM)."""
    data = open(path, "rb").read()
    tokens, pos = [], 2
    while len(tokens) < 3:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            pos = data.index(b"\n", pos)
            continue
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(int(data[start:pos]))
    pos += 1
    w, h, maxval = tokens
    c = 3 if data[1:2] == b"6" else 1
    dt = np.dtype(">u2") if maxval > 255 else np.uint8
    a = np.frombuffer(data[pos:], dtype=dt, count=w * h * c).reshape(h, w, c).astype(np.int64)
    return (a[..., 0] if c == 1 else a), maxval


def write_pnm(path, a, maxval):
    c = 1 if a.ndim == 2 else a.shape[2]
    hdr = b"P%d\n%d %d\n%d\n" % (5 if c == 1 else 6, a.shape[1], a.shape[0], maxval)
    dt = ">u2" if maxval > 255 else np.uint8
    open(path, "wb").write(hdr + np.ascontiguousarray(a).astype(dt).tobytes())


def section_jpeg_reference():
    s = "jpeg/libjpeg-turbo"
    if not os.path.isdir(LJT_BIN):
        report(s, "libjpeg-turbo tools not found (set IPF_LJT_BIN)", False)
        return

    def check(name, jpg, tol=0, mean_tol=0.0):
        ljt("djpeg", "-dct", "float", "-outfile", tmp("ref.pnm"), jpg)
        ref, _ = read_pnm(tmp("ref.pnm"))
        run("decoderaw", jpg, tmp("raw.npy"))
        got = np.load(tmp("raw.npy")).astype(np.int64)
        if tol == 0:
            compare_exact(s, name, got, ref)
        else:
            compare_close(s, name, got, ref, tol, mean_tol)

    orig = os.path.join(LJT_TESTIMAGES, "testorig.jpg")
    ari = os.path.join(LJT_TESTIMAGES, "testimgari.jpg")
    # official arithmetic-coded test image (an independent encode, compared with djpeg's output)
    run("decoderaw", orig, tmp("a.npy"))
    check("testimgari.jpg (arithmetic) vs djpeg", ari, 3, 0.1)
    check("testimgint.jpg (progressive) vs djpeg", os.path.join(LJT_TESTIMAGES, "testimgint.jpg"), 3, 0.1)
    # lossless transcodes with jpegtran: arithmetic / progressive / restarts keep coefficients
    for opts in (["-arithmetic"], ["-arithmetic", "-progressive"], ["-arithmetic", "-restart", "2"],
                 ["-arithmetic", "-progressive", "-restart", "1"], ["-progressive", "-restart", "3"]):
        out = tmp("t.jpg")
        ljt("jpegtran", *opts, "-outfile", out, orig)
        run("decoderaw", out, tmp("c.npy"))
        compare_exact(s, f"jpegtran {' '.join(opts)} == original", np.load(tmp("c.npy")), np.load(tmp("a.npy")))
    # 8-bit and 12-bit encodes with cjpeg
    rgb = natural_image(53, 77, 3, 111)
    gray = natural_image(41, 66, 1, 112)
    rgb16 = (rgb.astype(np.int64) * 257 + np.arange(77)[None, :, None] * 3) % 65536
    write_pnm(tmp("in8.ppm"), rgb, 255)
    write_pnm(tmp("in16.ppm"), rgb16, 65535)
    write_pnm(tmp("g16.pgm"), (gray.astype(np.int64) * 257) % 65536, 65535)
    for prec, src in ((8, "in8.ppm"), (12, "in16.ppm")):
        for opts in ([], ["-progressive"], ["-arithmetic"], ["-arithmetic", "-progressive"], ["-sample", "1x1"],
                     ["-sample", "2x1", "-restart", "1"], ["-optimize"]):
            out = tmp("e.jpg")
            ljt("cjpeg", "-precision", prec, *opts, "-outfile", out, tmp(src))
            tol = 3  # float IDCT rounding differences (observed max: 2)
            check(f"cjpeg {prec}-bit {' '.join(opts) or 'baseline'} vs djpeg", out, tol, 0.05)
    ljt("cjpeg", "-precision", 12, "-grayscale", "-arithmetic", "-outfile", tmp("g.jpg"), tmp("g16.pgm"))
    check("cjpeg 12-bit gray arithmetic vs djpeg", tmp("g.jpg"), 3, 0.05)
    # lossless: every predictor, several precisions / point transforms, restart intervals
    for prec in (2, 8, 12, 16):
        maxval = (1 << prec) - 1
        src16 = np.clip(np.round(rgb.astype(np.float64) / 255 * maxval), 0, maxval).astype(np.int64)
        write_pnm(tmp("ll.ppm"), src16, maxval)
        write_pnm(tmp("ll.pgm"), src16[..., 1], maxval)
        for psv in range(1, 8):
            for pt in sorted({0, min(prec - 1, 3)}):
                for src in ("ll.ppm", "ll.pgm"):
                    extra = ["-restart", "1"] if psv in (4, 7) else []
                    out = tmp("l.jpg")
                    ljt("cjpeg", "-precision", prec, "-lossless", f"{psv},{pt}", *extra, "-outfile", out, tmp(src))
                    check(f"lossless P={prec} psv={psv} pt={pt} {src[-3:]} {' '.join(extra)}", out)


# ---------------------------------------------------------------------------
# Independent ICC profile writer (for littleCMS comparisons)
# ---------------------------------------------------------------------------

def _s15(v):
    return struct.pack(">i", int(round(v * 65536)))


def _xyz_tag(x, y, z):
    return b"XYZ \0\0\0\0" + _s15(x) + _s15(y) + _s15(z)


def _desc_tag(text):
    t = text.encode() + b"\0"
    return b"desc\0\0\0\0" + struct.pack(">I", len(t)) + t + b"\0" * 12 + b"\0" * 67


def _curv_gamma(g):
    return b"curv\0\0\0\0" + struct.pack(">IH", 1, int(round(g * 256))) + b"\0\0"


def _curv_table(vals):
    return b"curv\0\0\0\0" + struct.pack(">I", len(vals)) + b"".join(struct.pack(">H", v) for v in vals)


def _para(ftype, params):
    return b"para\0\0\0\0" + struct.pack(">HH", ftype, 0) + b"".join(_s15(p) for p in params)


def make_icc(colorspace, pcs, tags, version=0x02100000, cls=b"mntr"):
    n = len(tags)
    offset = 128 + 4 + 12 * n
    table, data = b"", b""
    for sig, blob in tags:
        blob += b"\0" * ((4 - len(blob) % 4) % 4)
        table += sig + struct.pack(">II", offset + len(data), len(blob))
        data += blob
    size = offset + len(data)
    hdr = struct.pack(">I4sI4s4s4s", size, b"ipf ", version, cls, colorspace, pcs) + b"\0" * 12
    hdr += b"acspMSFT" + b"\0" * 4 + b"\0" * 8 + b"\0" * 8 + struct.pack(">I", 0)
    hdr += _s15(0.9642) + _s15(1.0) + _s15(0.8249) + b"ipf " + b"\0" * 44
    assert len(hdr) == 128
    return hdr + struct.pack(">I", n) + table + data


def bradford_d50(prim, white):
    def xyz(x, y):
        return np.array([x / y, 1.0, (1 - x - y) / y])
    P = np.stack([xyz(*prim[0]), xyz(*prim[1]), xyz(*prim[2])], 1)
    Wt = xyz(*white)
    M = P * np.linalg.solve(P, Wt)
    B = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
    d50 = np.array([0.9642, 1.0, 0.8249])
    A = np.linalg.inv(B) @ np.diag((B @ d50) / (B @ Wt)) @ B
    return A @ M


def matrix_profile(prim, white, curve_tag, desc):
    M = bradford_d50(prim, white)
    return make_icc(b"RGB ", b"XYZ ", [
        (b"desc", _desc_tag(desc)), (b"wtpt", _xyz_tag(0.9642, 1.0, 0.8249)),
        (b"rXYZ", _xyz_tag(*M[:, 0])), (b"gXYZ", _xyz_tag(*M[:, 1])), (b"bXYZ", _xyz_tag(*M[:, 2])),
        (b"rTRC", curve_tag), (b"gTRC", curve_tag), (b"bTRC", curve_tag)])


def mab_profile():
    """v4 RGB profile whose A2B0 (lutAtoBType) maps RGB to Lab through a 9x9x9 CLUT."""
    g = 9
    grid = np.linspace(0, 1, g)
    r, gg, b = np.meshgrid(grid, grid, grid, indexing="ij")
    lab = np.stack([100 * (0.3 * r + 0.6 * gg + 0.1 * b) ** 0.8, 60 * (r - gg) + 128, 50 * (gg - b) * (1 - 0.3 * r) + 128], -1)
    enc = np.stack([lab[..., 0] / 100, lab[..., 1] / 255, lab[..., 2] / 255], -1)
    clut = np.clip(np.round(enc * 65535), 0, 65535).astype(">u2").tobytes()
    ident = _curv_table([0, 65535])
    acurves = _para(0, [2.2])
    acurves += b"\0" * ((4 - len(acurves) % 4) % 4)
    bcurves = (ident + b"\0" * ((4 - len(ident) % 4) % 4)) * 3
    clut_blob = bytes([g, g, g] + [0] * 13) + bytes([2, 0, 0, 0]) + clut
    clut_blob += b"\0" * ((4 - len(clut_blob) % 4) % 4)
    off_b = 32
    off_clut = off_b + len(bcurves)
    off_a = off_clut + len(clut_blob)
    mab = b"mAB \0\0\0\0" + bytes([3, 3, 0, 0]) + struct.pack(">IIIII", off_b, 0, 0, off_clut, off_a)
    mab += bcurves + clut_blob + acurves * 3
    return make_icc(b"RGB ", b"Lab ", [(b"desc", _desc_tag("ipf mAB test")), (b"wtpt", _xyz_tag(0.9642, 1.0, 0.8249)),
                                       (b"A2B0", mab)], version=0x04300000, cls=b"scnr")


def section_icc():
    from PIL import ImageCms
    s = "icc/littleCMS"
    srgb = ImageCms.createProfile("sRGB")
    rgb = natural_image(40, 57, 3, 121)
    ramp = np.zeros((16, 256, 3), np.uint8)
    ramp[..., 0] = np.arange(256)[None, :]
    ramp[..., 1] = (np.arange(16)[:, None] * 17)
    ramp[..., 2] = 255 - np.arange(256)[None, :]
    img = np.concatenate([rgb, ramp[:, :57][:16]], 0)

    def lcms(arr, src_bytes, mode_in="RGB", intent=0, dst=srgb, mode_out="RGB"):
        # NOOPTIMIZE: evaluate the exact pipeline (lcms otherwise resamples LUT pipelines)
        src = ImageCms.ImageCmsProfile(io.BytesIO(src_bytes)) if isinstance(src_bytes, bytes) else src_bytes
        t = ImageCms.buildTransform(src, dst, mode_in, mode_out, intent, ImageCms.Flags.NOOPTIMIZE)
        return np.array(ImageCms.applyTransform(Image.fromarray(arr, mode_in), t))

    prim_srgb = [(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)]
    prim_p3 = [(0.680, 0.320), (0.265, 0.690), (0.150, 0.060)]
    prim_adobe = [(0.64, 0.33), (0.21, 0.71), (0.15, 0.06)]
    d65 = (0.3127, 0.3290)
    srgb_curve = _para(3, [2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045])
    profiles = {
        "Adobe RGB (curv gamma 2.2)": matrix_profile(prim_adobe, d65, _curv_gamma(563 / 256), "adobe"),
        "Display P3 (para type 3)": matrix_profile(prim_p3, d65, srgb_curve, "p3"),
        "sRGB primaries, sampled TRC": matrix_profile(prim_srgb, d65,
                                                      _curv_table([int(round(65535 * (i / 1023) ** 1.8)) for i in range(1024)]), "t"),
        "v4 lutAtoB (mAB) RGB->Lab": mab_profile(),
        "libjpeg-turbo test3.icc (mft1)": open(os.path.join(LJT_TESTIMAGES, "test3.icc"), "rb").read(),
    }
    for name, prof in profiles.items():
        path = tmp("p.icc")
        open(path, "wb").write(prof)
        # littleCMS always applies black point compensation to v4 perceptual transforms (ipf does
        # not), so v4 profiles are compared with the relative colorimetric intent
        intent = 1 if "v4" in name else 0
        compare_close(s, f"{name} -> sRGB intent {intent} vs littleCMS", ipf_op("iccconvert", img, path, "srgb", intent),
                      lcms(img, prof, intent=intent), 1, 0.05)
    # sRGB -> other profile (inverse curves / matrix)
    for name in ("Adobe RGB (curv gamma 2.2)", "Display P3 (para type 3)"):
        prof = profiles[name]
        open(tmp("p.icc"), "wb").write(prof)
        want = lcms(img, srgb, dst=ImageCms.ImageCmsProfile(io.BytesIO(prof)))
        compare_close(s, f"sRGB -> {name} vs littleCMS", ipf_op("iccconvert", img, "srgb", tmp("p.icc"), 0), want, 1, 0.05)
    # gray profile
    gprof = make_icc(b"GRAY", b"XYZ ", [(b"desc", _desc_tag("gray")), (b"wtpt", _xyz_tag(0.9642, 1.0, 0.8249)),
                                        (b"kTRC", _curv_gamma(1.8))])
    gray = img[..., 1].copy()
    open(tmp("g.icc"), "wb").write(gprof)
    want = lcms(gray, gprof, "L")
    got = ipf_op("iccconvert", gray, tmp("g.icc"), "srgb", 0)
    compare_close(s, "gray (kTRC gamma 1.8) -> sRGB vs littleCMS", got, want, 1, 0.05)
    # CMYK printer profile (lut16 A2B with 4D CLUT, Lab PCS)
    cmyk_prof = open(os.path.join(LJT_TESTIMAGES, "test1.icc"), "rb").read()
    rng = np.random.default_rng(122)
    cmyk = np.concatenate([natural_image(30, 40, 4, 123), rng.integers(0, 256, (10, 40, 4)).astype(np.uint8)], 0)
    open(tmp("c.icc"), "wb").write(cmyk_prof)
    for intent in (0, 1):
        compare_close(s, f"CMYK test1.icc (mft2) -> sRGB intent {intent} vs littleCMS",
                      ipf_op("iccconvert", cmyk, tmp("c.icc"), "srgb", intent), lcms(cmyk, cmyk_prof, "CMYK", intent), 1, 0.05)
    want = lcms(img, srgb, dst=ImageCms.ImageCmsProfile(io.BytesIO(cmyk_prof)), mode_out="CMYK")
    compare_close(s, "sRGB -> CMYK test1.icc (mft1 B2A) vs littleCMS",
                  ipf_op("iccconvert", img, "srgb", tmp("c.icc"), 0), want, 1, 0.05)
    # end to end: CMYK JPEG with embedded profile, loaded with convertToSRGB
    data = pil_bytes(Image.fromarray(cmyk, "CMYK"), "JPEG", quality=95, icc_profile=cmyk_prof)
    open(tmp("cmyk.jpg"), "wb").write(data)
    # (littleCMS is applied to Pillow's CMYK decode; small decode differences are amplified near
    # the sRGB gamut boundary, hence the looser bound)
    pil = np.array(Image.open(io.BytesIO(data)))
    compare_close(s, "load CMYK JPEG + ICC as sRGB vs Pillow+littleCMS", ipf_op("load", tmp("cmyk.jpg"), 1, 1),
                  lcms(pil, cmyk_prof, "CMYK"), 2, 0.05)
    # RGB JPEG / PNG with an embedded Adobe RGB profile
    adobe = profiles["Adobe RGB (curv gamma 2.2)"]
    for fmt in ("PNG", "JPEG"):
        data = pil_bytes(Image.fromarray(img), fmt, icc_profile=adobe, **({"quality": 98} if fmt == "JPEG" else {}))
        p = tmp("emb." + fmt.lower())
        open(p, "wb").write(data)
        decoded = ipf_op("load", p, 0, 1)  # isolate the color conversion from JPEG decoding differences
        compare_close(s, f"load {fmt} with embedded Adobe RGB as sRGB", ipf_op("load", p, 1, 1), lcms(decoded, adobe), 1, 0.05)


def section_metadata_apng():
    from PIL import ImageOps, PngImagePlugin
    s = "metadata/apng"
    rgb = natural_image(30, 44, 3, 131)
    # EXIF orientation: Pillow writes the tag, ipf must match ImageOps.exif_transpose
    for o in range(1, 9):
        im = Image.fromarray(rgb)
        exif = im.getexif()
        exif[0x0112] = o
        data = pil_bytes(im, "JPEG", quality=95, exif=exif.tobytes())
        open(tmp("o.jpg"), "wb").write(data)
        want = np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(data))))
        compare_close(s, f"EXIF orientation {o} vs Pillow exif_transpose", ipf_op("load", tmp("o.jpg"), 0, 1), want, 3, 0.1)
        data = pil_bytes(im, "PNG", exif=exif.tobytes())
        open(tmp("o.png"), "wb").write(data)
        compare_exact(s, f"PNG eXIf orientation {o} vs Pillow", ipf_op("load", tmp("o.png"), 0, 1),
                      np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(data)))))
    # PNG metadata written by Pillow, read by ipf
    info = PngImagePlugin.PngInfo()
    info.add_text("Title", "hello")
    info.add_itxt("Description", "café ✓")
    info.add_text("Comment", "z" * 300, zip=True)
    icc = open(os.path.join(LJT_TESTIMAGES, "test3.icc"), "rb").read()
    data = pil_bytes(Image.fromarray(rgb), "PNG", pnginfo=info, dpi=(300, 300), icc_profile=icc)
    open(tmp("m.png"), "wb").write(data)
    run("metadump", tmp("m.png"), tmp("m.txt"))
    dump = open(tmp("m.txt"), encoding="utf-8").read()
    ok = ("text Title=hello" in dump and "text Description=café ✓" in dump and
          ("text Comment=" + "z" * 300) in dump and f"icc {len(icc)} {zlib.crc32(icc)}" in dump and "dpi 299.9" in dump)
    report(s, "Pillow PNG text/iTXt/zTXt/iCCP/pHYs read by ipf", ok)
    # metadata written by ipf, read by Pillow
    np.save(tmp("m.npy"), rgb)
    run("savemeta", tmp("m.npy"), tmp("w.png"), os.path.join(LJT_TESTIMAGES, "test3.icc"))
    im = Image.open(tmp("w.png"))
    ok = (im.text.get("Title") == "ipf" and im.text.get("Description") == "café ✓" and
          im.info.get("icc_profile") == icc and abs(im.info["dpi"][0] - 300) < 0.1)
    report(s, "ipf PNG metadata read by Pillow", ok, f"({im.text}, dpi {im.info.get('dpi')})")
    run("savemeta", tmp("m.npy"), tmp("w.jpg"), os.path.join(LJT_TESTIMAGES, "test1.icc"))  # 557 KB: 9 APP2 chunks
    im = Image.open(tmp("w.jpg"))
    ok = (im.info.get("icc_profile") == open(os.path.join(LJT_TESTIMAGES, "test1.icc"), "rb").read() and
          im.info.get("dpi") == (300, 300) and im.info.get("comment") == b"hello")
    report(s, "ipf JPEG ICC (multi-chunk) / DPI / comment read by Pillow", ok)
    # PNG optimized encodings decoded by Pillow
    pal = (natural_image(31, 45, 3, 132) // 64 * 64).astype(np.uint8)
    rgba = np.dstack([pal, np.where(pal[..., 0] > 100, 255, 90).astype(np.uint8)])
    mask = ((natural_image(31, 45, 1, 133) > 128) * 255).astype(np.uint8)
    for name, arr, mode in (("palette RGB", pal, "RGB"), ("palette RGBA", rgba, "RGBA"), ("1-bit mask", mask, "L"),
                            ("16-level gray", (natural_image(31, 45, 1, 134) // 16 * 17).astype(np.uint8), "L"),
                            ("full RGB", rgb, "RGB")):
        for il in (0, 1):
            np.save(tmp("o.npy"), arr)
            run("pngopt", tmp("o.npy"), tmp("o.png"), il)
            compare_exact(s, f"Pillow decodes ipf PNG {name} interlace={il}", np.array(Image.open(tmp("o.png")).convert(mode)), arr)
    # APNG written by Pillow (frame-difference optimized: sub-regions, dispose and blend ops)
    frames = []
    for i in range(6):
        f = np.zeros((32, 48, 4), np.uint8)
        f[..., 2] = 40
        f[..., 3] = 255
        f[4:20, 3 + 5 * i:19 + 5 * i] = [200, 30 * i, 0, 255]
        if i % 2:
            f[24:30, 5:40, 3] = 0  # binary transparency
        frames.append(f)
    for disposal, blend in ((0, 0), (1, 0), (2, 0), (0, 1), (2, 1)):
        ims = [Image.fromarray(f, "RGBA") for f in frames]
        data = pil_bytes(ims[0], "PNG", save_all=True, append_images=ims[1:], duration=[40 * (k + 1) for k in range(6)],
                         loop=2, disposal=disposal, blend=blend)
        open(tmp("a.png"), "wb").write(data)
        run("apngdecode", tmp("a.png"), tmp("a.npy"))
        got = np.load(tmp("a.npy")).reshape(-1, 32, 48, 4)
        pil = Image.open(io.BytesIO(data))
        want = []
        for k in range(pil.n_frames):
            pil.seek(k)
            want.append(np.array(pil.convert("RGBA")))
        meta = open(tmp("a.npy") + ".txt").read().split()
        ok_meta = meta[0] == "2" and [int(v) for v in meta[1:]] == [40 * (k + 1) for k in range(len(want))]
        compare_exact(s, f"APNG from Pillow disposal={disposal} blend={blend}", got, np.stack(want))
        report(s, f"APNG loop count + delays disposal={disposal} blend={blend}", ok_meta, f"({' '.join(meta)})")
    # APNG written by ipf, read by Pillow
    np.save(tmp("f.npy"), np.concatenate(frames, 0))
    run("apngencode", tmp("f.npy"), tmp("b.png"), 6, 0)
    pil = Image.open(tmp("b.png"))
    got = []
    for k in range(pil.n_frames):
        pil.seek(k)
        got.append(np.array(pil.convert("RGBA")))
    compare_exact(s, "Pillow decodes ipf APNG", np.stack(got), np.stack(frames))
    report(s, "ipf APNG frame count / duration", pil.n_frames == 6 and Image.open(tmp("b.png")).info.get("duration") == 40)


ALL = {
    "zlib": section_zlib,
    "png": section_png,
    "jpeg": section_jpeg,
    "bmp_tga_pnm": section_bmp_tga_pnm,
    "color": section_color,
    "pixel": section_pixel,
    "filter": section_filter,
    "geometry": section_geometry,
    "morph_analysis": section_morph_analysis,
    "features_metrics": section_features_metrics,
    "fft_cmaps": section_fft_cmaps,
    "jpeg_reference": section_jpeg_reference,
    "icc": section_icc,
    "metadata_apng": section_metadata_apng,
}


def main():
    names = SECTIONS or list(ALL)
    for n in names:
        print(f"== {n}")
        try:
            ALL[n]()
        except Exception as e:  # report and continue with other sections
            report(n, "section crashed", False, repr(e))
    failed = [r for r in RESULTS if not r[2]]
    print(f"\n{len(RESULTS)} cross-validation checks, {len(failed)} failed")
    for sec, name, _, detail in failed:
        print(f"  FAILED {sec}: {name} {detail}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
