// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf codec: Windows BMP (1/4/8-bit palette incl. RLE4/RLE8, 16/24/32-bit, bitfields).
#pragma once

#include "common.hpp"

namespace ipf {

namespace detail {

inline int bmpMaskShift(std::uint32_t m) {
    int s = 0;
    if (!m) return 0;
    while (!(m & 1u)) { m >>= 1; ++s; }
    return s;
}
inline int bmpMaskBits(std::uint32_t m) {
    int n = 0;
    for (; m; m &= m - 1) ++n;
    return n;
}
inline std::uint8_t bmpExtract(std::uint32_t v, std::uint32_t mask, int shift, int bits) {
    if (!mask) return 0;
    const std::uint32_t x = (v & mask) >> shift;
    const std::uint32_t maxv = bits >= 32 ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    return static_cast<std::uint8_t>((static_cast<unsigned long long>(x) * 255u + maxv / 2) / maxv);
}

}  // namespace detail

/// Decodes a BMP file. Output has 1 channel (grayscale palette), 3 (RGB) or
/// 4 (RGBA, when the file carries a non-trivial alpha channel).
inline ImageU8 decodeBMP(const std::vector<std::uint8_t>& bytes) {
    detail::ByteReader rd(bytes.data(), bytes.size(), "BMP");
    if (rd.u8() != 'B' || rd.u8() != 'M') detail::fail("BMP: bad signature");
    rd.skip(8);
    const std::uint32_t dataOffset = rd.u32le();
    const std::uint32_t hdrSize = rd.u32le();
    long long w, h;
    int bpp;
    std::uint32_t compression = 0, colorsUsed = 0;
    const bool os2 = hdrSize == 12;
    if (os2) {
        w = rd.u16le();
        h = static_cast<std::int16_t>(rd.u16le());
        rd.skip(2);
        bpp = rd.u16le();
    } else {
        if (hdrSize < 40) detail::fail("BMP: unsupported header size");
        w = rd.i32le();
        h = rd.i32le();
        rd.skip(2);
        bpp = rd.u16le();
        compression = rd.u32le();
        rd.skip(12);
        colorsUsed = rd.u32le();
        rd.skip(4);
    }
    const bool topDown = h < 0;
    if (topDown) h = -h;
    detail::checkDecodedSize(w, h, "BMP");

    std::uint32_t masks[4] = {0, 0, 0, 0};
    bool haveMasks = false;
    if (compression == 3 || compression == 6) {  // BI_BITFIELDS / BI_ALPHABITFIELDS
        // Masks live inside V2+ headers or directly after a 40-byte header.
        const int count = (compression == 6 || hdrSize >= 56) ? 4 : 3;
        for (int i = 0; i < count; ++i) masks[i] = rd.u32le();
        haveMasks = true;
    }
    if (compression != 0 && compression != 1 && compression != 2 && compression != 3 && compression != 6)
        detail::fail("BMP: unsupported compression (JPEG/PNG-in-BMP is not supported)");
    if (compression == 1 && bpp != 8) detail::fail("BMP: RLE8 requires 8 bpp");
    if (compression == 2 && bpp != 4) detail::fail("BMP: RLE4 requires 4 bpp");

    // Palette
    std::vector<std::uint8_t> pal;  // RGB triples
    if (bpp <= 8) {
        rd.seek(14 + hdrSize + ((compression == 3 && hdrSize == 40) ? 12 : 0));
        std::uint32_t n = colorsUsed ? colorsUsed : (1u << bpp);
        if (n > 256) n = 256;
        const int entry = os2 ? 3 : 4;
        pal.resize(static_cast<std::size_t>(n) * 3);
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::uint8_t* e = rd.take(entry);
            pal[i * 3 + 0] = e[2];
            pal[i * 3 + 1] = e[1];
            pal[i * 3 + 2] = e[0];
        }
    } else if (bpp != 16 && bpp != 24 && bpp != 32) {
        detail::fail("BMP: unsupported bit depth " + std::to_string(bpp));
    }

    const int W = static_cast<int>(w), H = static_cast<int>(h);
    rd.seek(dataOffset);

    if (bpp <= 8) {
        // Decode palette indices first.
        std::vector<std::uint8_t> idx(static_cast<std::size_t>(W) * H, 0);
        auto put = [&](int x, int yFile, int v) {
            if (x < 0 || x >= W || yFile < 0 || yFile >= H) return;
            const int y = topDown ? yFile : H - 1 - yFile;
            idx[static_cast<std::size_t>(y) * W + x] = static_cast<std::uint8_t>(v);
        };
        if (compression == 1 || compression == 2) {
            int x = 0, y = 0;
            const bool rle4 = compression == 2;
            while (rd.remaining() >= 2) {
                const int count = rd.u8(), val = rd.u8();
                if (count > 0) {  // encoded run
                    for (int i = 0; i < count; ++i, ++x)
                        put(x, y, rle4 ? ((i & 1) ? (val & 0x0F) : (val >> 4)) : val);
                } else if (val == 0) {  // end of line
                    x = 0;
                    ++y;
                } else if (val == 1) {  // end of bitmap
                    break;
                } else if (val == 2) {  // delta
                    x += rd.u8();
                    y += rd.u8();
                } else {  // absolute run of `val` pixels
                    const int nbytes = rle4 ? (val + 1) / 2 : val;
                    const std::uint8_t* p = rd.take(static_cast<std::size_t>(nbytes));
                    for (int i = 0; i < val; ++i, ++x)
                        put(x, y, rle4 ? ((i & 1) ? (p[i / 2] & 0x0F) : (p[i / 2] >> 4)) : p[i]);
                    if (nbytes & 1) rd.skip(1);  // word alignment
                }
            }
        } else {
            const std::size_t rowBytes = ((static_cast<std::size_t>(W) * bpp + 31) / 32) * 4;
            for (int yf = 0; yf < H; ++yf) {
                const std::uint8_t* r = rd.take(rowBytes);
                for (int x = 0; x < W; ++x) {
                    const std::size_t bit = static_cast<std::size_t>(x) * bpp;
                    const int v = (r[bit / 8] >> (8 - bpp - static_cast<int>(bit % 8))) & ((1 << bpp) - 1);
                    put(x, yf, v);
                }
            }
        }
        const std::size_t ncol = pal.size() / 3;
        bool gray = true;
        for (std::size_t i = 0; i < ncol; ++i)
            if (pal[i * 3] != pal[i * 3 + 1] || pal[i * 3] != pal[i * 3 + 2]) gray = false;
        ImageU8 img(W, H, gray ? 1 : 3);
        std::uint8_t* d = img.data();
        for (std::size_t i = 0; i < idx.size(); ++i) {
            const std::size_t e = idx[i] < ncol ? idx[i] : 0;
            if (gray) {
                d[i] = ncol ? pal[e * 3] : 0;
            } else {
                d[i * 3 + 0] = pal[e * 3 + 0];
                d[i * 3 + 1] = pal[e * 3 + 1];
                d[i * 3 + 2] = pal[e * 3 + 2];
            }
        }
        return img;
    }

    // Direct color
    if (!haveMasks) {
        if (bpp == 16) {
            masks[0] = 0x7C00; masks[1] = 0x03E0; masks[2] = 0x001F;
        } else {
            masks[0] = 0x00FF0000; masks[1] = 0x0000FF00; masks[2] = 0x000000FF;
            masks[3] = bpp == 32 ? 0xFF000000u : 0u;  // tentative, see alpha heuristic below
        }
    }
    int shift[4], bits[4];
    for (int i = 0; i < 4; ++i) {
        shift[i] = detail::bmpMaskShift(masks[i]);
        bits[i] = detail::bmpMaskBits(masks[i]);
    }
    const std::size_t rowBytes = ((static_cast<std::size_t>(W) * bpp + 31) / 32) * 4;
    ImageU8 img(W, H, 4);
    bool anyAlpha = false;
    for (int yf = 0; yf < H; ++yf) {
        const std::uint8_t* r = rd.take(rowBytes);
        std::uint8_t* d = img.row(topDown ? yf : H - 1 - yf);
        for (int x = 0; x < W; ++x) {
            std::uint32_t v;
            const std::uint8_t* p = r + static_cast<std::size_t>(x) * (bpp / 8);
            if (bpp == 16) v = p[0] | (p[1] << 8);
            else if (bpp == 24) v = p[0] | (p[1] << 8) | (static_cast<std::uint32_t>(p[2]) << 16);
            else v = p[0] | (p[1] << 8) | (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
            for (int c = 0; c < 3; ++c) d[x * 4 + c] = detail::bmpExtract(v, masks[c], shift[c], bits[c]);
            d[x * 4 + 3] = masks[3] ? detail::bmpExtract(v, masks[3], shift[3], bits[3]) : 255;
            if (masks[3] && d[x * 4 + 3] != 0) anyAlpha = true;
        }
    }
    // Alpha is only meaningful if explicitly declared via bitfields, or if a
    // plain 32-bit BI_RGB file actually uses it (all-zero alpha means "unused").
    const bool keepAlpha = masks[3] != 0 && (haveMasks || anyAlpha);
    return keepAlpha ? img : convertChannels(img, 3);
}

/// Encodes an image as BMP: 1 channel -> 8-bit grayscale palette, 3 channels ->
/// 24-bit, 2/4 channels -> 32-bit BGRA with a V4 header (alpha preserved).
inline std::vector<std::uint8_t> encodeBMP(const ImageU8& src) {
    detail::require(!src.empty(), "encodeBMP: empty image");
    const ImageU8 img = src.channels() == 2 ? convertChannels(src, 4) : src;
    detail::require(img.channels() <= 4, "encodeBMP: unsupported channel count");
    const int W = img.width(), H = img.height(), cn = img.channels();
    const int bpp = cn == 1 ? 8 : (cn == 3 ? 24 : 32);
    const std::uint32_t hdrSize = cn == 4 ? 108 : 40;
    const std::uint32_t palSize = cn == 1 ? 256 * 4 : 0;
    const std::size_t rowBytes = ((static_cast<std::size_t>(W) * bpp + 31) / 32) * 4;
    const std::uint32_t offset = 14 + hdrSize + palSize;
    const std::size_t total = offset + rowBytes * H;
    detail::require(total < 0xFFFFFFFFull, "encodeBMP: image too large for BMP");
    std::vector<std::uint8_t> o;
    o.reserve(total);
    detail::put8(o, 'B');
    detail::put8(o, 'M');
    detail::put32le(o, static_cast<std::uint32_t>(total));
    detail::put32le(o, 0);
    detail::put32le(o, offset);
    detail::put32le(o, hdrSize);
    detail::put32le(o, static_cast<std::uint32_t>(W));
    detail::put32le(o, static_cast<std::uint32_t>(H));  // bottom-up
    detail::put16le(o, 1);
    detail::put16le(o, static_cast<unsigned>(bpp));
    detail::put32le(o, cn == 4 ? 3u : 0u);  // BI_BITFIELDS for RGBA
    detail::put32le(o, static_cast<std::uint32_t>(rowBytes * H));
    detail::put32le(o, 2835);  // 72 DPI
    detail::put32le(o, 2835);
    detail::put32le(o, cn == 1 ? 256 : 0);
    detail::put32le(o, 0);
    if (cn == 4) {
        detail::put32le(o, 0x00FF0000);  // R
        detail::put32le(o, 0x0000FF00);  // G
        detail::put32le(o, 0x000000FF);  // B
        detail::put32le(o, 0xFF000000);  // A
        detail::put32le(o, 0x73524742);  // 'sRGB'
        for (int i = 0; i < 12; ++i) detail::put32le(o, 0);  // endpoints + gamma
    }
    if (cn == 1)
        for (unsigned i = 0; i < 256; ++i) {
            detail::put8(o, i); detail::put8(o, i); detail::put8(o, i); detail::put8(o, 0);
        }
    for (int y = H - 1; y >= 0; --y) {
        const std::uint8_t* r = img.row(y);
        const std::size_t start = o.size();
        for (int x = 0; x < W; ++x) {
            if (cn == 1) {
                o.push_back(r[x]);
            } else {
                o.push_back(r[x * cn + 2]);
                o.push_back(r[x * cn + 1]);
                o.push_back(r[x * cn + 0]);
                if (cn == 4) o.push_back(r[x * cn + 3]);
            }
        }
        o.resize(start + rowBytes, 0);
    }
    return o;
}

}  // namespace ipf
