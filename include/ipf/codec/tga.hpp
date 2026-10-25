// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf codec: Truevision TGA (color-mapped, true-color and grayscale; raw and RLE).
#pragma once

#include "common.hpp"

namespace ipf {

namespace detail {
inline void tgaPixel(const std::uint8_t* p, int bytesPerPixel, bool gray, std::uint8_t* out, bool& hasAlpha) {
    if (gray) {
        out[0] = out[1] = out[2] = p[0];
        out[3] = bytesPerPixel == 2 ? p[1] : 255;
        if (bytesPerPixel == 2) hasAlpha = true;
        return;
    }
    switch (bytesPerPixel) {
        case 2: {
            const unsigned v = p[0] | (p[1] << 8);
            out[0] = static_cast<std::uint8_t>(((v >> 10) & 31) * 255 / 31);
            out[1] = static_cast<std::uint8_t>(((v >> 5) & 31) * 255 / 31);
            out[2] = static_cast<std::uint8_t>((v & 31) * 255 / 31);
            out[3] = 255;
            break;
        }
        case 3:
            out[0] = p[2]; out[1] = p[1]; out[2] = p[0]; out[3] = 255;
            break;
        case 4:
            out[0] = p[2]; out[1] = p[1]; out[2] = p[0]; out[3] = p[3];
            hasAlpha = true;
            break;
        default:
            fail("TGA: unsupported pixel size");
    }
}
}  // namespace detail

/// Decodes a TGA file (types 1, 2, 3, 9, 10, 11). Output: 1 channel for
/// grayscale, 2 for gray+alpha, 3 for RGB, 4 for RGBA.
inline ImageU8 decodeTGA(const std::vector<std::uint8_t>& bytes) {
    detail::ByteReader rd(bytes.data(), bytes.size(), "TGA");
    const int idLen = rd.u8();
    const int cmapType = rd.u8();
    const int type = rd.u8();
    const int cmapFirst = rd.u16le();
    const int cmapLen = rd.u16le();
    const int cmapBits = rd.u8();
    rd.skip(4);  // origin
    const int W = rd.u16le(), H = rd.u16le();
    const int bitsPerPixel = rd.u8();
    const int desc = rd.u8();
    const int base = type & 7;
    const bool rle = type & 8;
    if (base < 1 || base > 3 || (type & ~0xB)) detail::fail("TGA: unsupported image type " + std::to_string(type));
    detail::checkDecodedSize(W, H, "TGA");
    rd.skip(static_cast<std::size_t>(idLen));

    const bool mapped = base == 1;
    const bool gray = base == 3;
    std::vector<std::uint8_t> cmap;  // RGBA entries
    bool hasAlpha = false;
    int cmapBpp = 0;
    if (cmapType == 1) {
        cmapBpp = (cmapBits + 1) / 8;
        if (cmapBpp < 2 || cmapBpp > 4) detail::fail("TGA: unsupported color map entry size");
        cmap.resize(static_cast<std::size_t>(cmapLen) * 4);
        for (int i = 0; i < cmapLen; ++i) detail::tgaPixel(rd.take(cmapBpp), cmapBpp, false, &cmap[i * 4], hasAlpha);
    } else if (mapped) {
        detail::fail("TGA: color-mapped image without a color map");
    }
    const int bpp = (bitsPerPixel + 1) / 8;
    if (mapped ? (bpp != 1 && bpp != 2) : gray ? (bpp != 1 && bpp != 2) : (bpp < 2 || bpp > 4))
        detail::fail("TGA: unsupported pixel depth " + std::to_string(bitsPerPixel));

    ImageU8 rgba(W, H, 4);
    const bool rightToLeft = desc & 0x10, topToBottom = desc & 0x20;
    const std::size_t total = static_cast<std::size_t>(W) * H;
    std::uint8_t px[4] = {0, 0, 0, 255};
    auto decodeOne = [&](const std::uint8_t* p) {
        if (mapped) {
            const int index = (bpp == 1 ? p[0] : (p[0] | (p[1] << 8))) - cmapFirst;
            if (index < 0 || index >= cmapLen) detail::fail("TGA: color map index out of range");
            std::memcpy(px, &cmap[static_cast<std::size_t>(index) * 4], 4);
        } else {
            detail::tgaPixel(p, bpp, gray, px, hasAlpha);
        }
    };
    auto store = [&](std::size_t i) {
        const int xf = static_cast<int>(i % W), yf = static_cast<int>(i / W);
        const int x = rightToLeft ? W - 1 - xf : xf;
        const int y = topToBottom ? yf : H - 1 - yf;
        std::memcpy(rgba.ptr(x, y), px, 4);
    };
    std::size_t i = 0;
    while (i < total) {
        if (rle) {
            const int hdr = rd.u8();
            const std::size_t count = static_cast<std::size_t>(hdr & 0x7F) + 1;
            if (hdr & 0x80) {
                decodeOne(rd.take(bpp));
                for (std::size_t k = 0; k < count && i < total; ++k) store(i++);
            } else {
                for (std::size_t k = 0; k < count && i < total; ++k) {
                    decodeOne(rd.take(bpp));
                    store(i++);
                }
            }
        } else {
            decodeOne(rd.take(bpp));
            store(i++);
        }
    }
    if (gray) return convertChannels(rgba, hasAlpha ? 2 : 1);
    // Many writers set 32 bpp but leave alpha at 0; treat all-zero alpha as opaque.
    if (hasAlpha) {
        bool allZero = true;
        for (std::size_t k = 0; k < total; ++k)
            if (rgba.data()[k * 4 + 3]) { allZero = false; break; }
        if (allZero && (desc & 0x0F) == 0) hasAlpha = false;
    }
    return convertChannels(rgba, hasAlpha ? 4 : 3);
}

/// Encodes an image as TGA (top-left origin). 1 channel -> grayscale, 2 -> gray+alpha,
/// 3 -> 24-bit BGR, 4 -> 32-bit BGRA. With rle=true, run-length encoding is used.
inline std::vector<std::uint8_t> encodeTGA(const ImageU8& img, bool rle = true) {
    detail::require(!img.empty() && img.channels() <= 4, "encodeTGA: invalid image");
    detail::require(img.width() <= 65535 && img.height() <= 65535, "encodeTGA: image too large for TGA");
    const int cn = img.channels();
    const bool gray = cn <= 2;
    std::vector<std::uint8_t> o;
    detail::put8(o, 0);
    detail::put8(o, 0);
    detail::put8(o, (gray ? 3u : 2u) | (rle ? 8u : 0u));
    for (int k = 0; k < 5; ++k) detail::put8(o, 0);  // color map spec
    detail::put16le(o, 0);
    detail::put16le(o, 0);
    detail::put16le(o, static_cast<unsigned>(img.width()));
    detail::put16le(o, static_cast<unsigned>(img.height()));
    detail::put8(o, static_cast<unsigned>(cn * 8));
    detail::put8(o, 0x20u | ((cn == 2 || cn == 4) ? 8u : 0u));  // top-left origin + alpha bits
    std::vector<std::uint8_t> pix(static_cast<std::size_t>(cn));
    auto fileOrder = [&](const std::uint8_t* p, std::uint8_t* q) {
        if (gray) {
            for (int c = 0; c < cn; ++c) q[c] = p[c];
        } else {
            q[0] = p[2]; q[1] = p[1]; q[2] = p[0];
            if (cn == 4) q[3] = p[3];
        }
    };
    const std::size_t n = img.pixelCount();
    const std::uint8_t* d = img.data();
    if (!rle) {
        o.reserve(o.size() + n * cn);
        for (std::size_t i = 0; i < n; ++i) {
            fileOrder(d + i * cn, pix.data());
            o.insert(o.end(), pix.begin(), pix.end());
        }
        return o;
    }
    // RLE packets never cross scanlines (required by the TGA 2.0 spec).
    const std::size_t W = static_cast<std::size_t>(img.width());
    auto same = [&](std::size_t a, std::size_t b) { return std::memcmp(d + a * cn, d + b * cn, cn) == 0; };
    for (std::size_t rowStart = 0; rowStart < n; rowStart += W) {
        const std::size_t rowEnd = rowStart + W;
        std::size_t i = rowStart;
        while (i < rowEnd) {
            std::size_t run = 1;
            while (i + run < rowEnd && run < 128 && same(i, i + run)) ++run;
            if (run >= 2) {
                detail::put8(o, 0x80u | static_cast<unsigned>(run - 1));
                fileOrder(d + i * cn, pix.data());
                o.insert(o.end(), pix.begin(), pix.end());
                i += run;
            } else {
                std::size_t lit = 1;
                while (i + lit < rowEnd && lit < 128 && !(i + lit + 1 < rowEnd && same(i + lit, i + lit + 1))) ++lit;
                detail::put8(o, static_cast<unsigned>(lit - 1));
                for (std::size_t k = 0; k < lit; ++k) {
                    fileOrder(d + (i + k) * cn, pix.data());
                    o.insert(o.end(), pix.begin(), pix.end());
                }
                i += lit;
            }
        }
    }
    return o;
}

}  // namespace ipf
