// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf codec helpers: bounds-checked byte reading, file access and channel-count adaptation.
#pragma once

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "../core.hpp"

namespace ipf {

namespace detail {

/// Bounds-checked little/big endian reader over a byte buffer.
class ByteReader {
public:
    ByteReader(const std::uint8_t* data, std::size_t size, const char* what = "data")
        : p_(data), n_(size), what_(what) {}

    std::size_t pos() const { return pos_; }
    std::size_t size() const { return n_; }
    std::size_t remaining() const { return n_ - pos_; }
    bool eof() const { return pos_ >= n_; }
    void seek(std::size_t pos) {
        if (pos > n_) truncated();
        pos_ = pos;
    }
    void skip(std::size_t k) {
        if (k > remaining()) truncated();
        pos_ += k;
    }
    const std::uint8_t* take(std::size_t k) {
        if (k > remaining()) truncated();
        const std::uint8_t* r = p_ + pos_;
        pos_ += k;
        return r;
    }
    std::uint8_t u8() { return *take(1); }
    std::uint16_t u16le() {
        const std::uint8_t* b = take(2);
        return static_cast<std::uint16_t>(b[0] | (b[1] << 8));
    }
    std::uint16_t u16be() {
        const std::uint8_t* b = take(2);
        return static_cast<std::uint16_t>((b[0] << 8) | b[1]);
    }
    std::uint32_t u32le() {
        const std::uint8_t* b = take(4);
        return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
               (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
    }
    std::uint32_t u32be() {
        const std::uint8_t* b = take(4);
        return (static_cast<std::uint32_t>(b[0]) << 24) | (static_cast<std::uint32_t>(b[1]) << 16) |
               (static_cast<std::uint32_t>(b[2]) << 8) | static_cast<std::uint32_t>(b[3]);
    }
    std::int32_t i32le() { return static_cast<std::int32_t>(u32le()); }

private:
    [[noreturn]] void truncated() const { fail(std::string(what_) + ": unexpected end of data (truncated file?)"); }
    const std::uint8_t* p_;
    std::size_t n_;
    std::size_t pos_ = 0;
    const char* what_;
};

inline void put8(std::vector<std::uint8_t>& o, unsigned v) { o.push_back(static_cast<std::uint8_t>(v)); }
inline void put16le(std::vector<std::uint8_t>& o, unsigned v) {
    put8(o, v & 0xFF);
    put8(o, (v >> 8) & 0xFF);
}
inline void put16be(std::vector<std::uint8_t>& o, unsigned v) {
    put8(o, (v >> 8) & 0xFF);
    put8(o, v & 0xFF);
}
inline void put32le(std::vector<std::uint8_t>& o, std::uint32_t v) {
    put16le(o, v & 0xFFFF);
    put16le(o, v >> 16);
}
inline void put32be(std::vector<std::uint8_t>& o, std::uint32_t v) {
    put16be(o, v >> 16);
    put16be(o, v & 0xFFFF);
}

/// Validates decoded image dimensions to protect against hostile headers.
inline void checkDecodedSize(long long w, long long h, const char* fmt) {
    if (w <= 0 || h <= 0) fail(std::string(fmt) + ": invalid image dimensions");
    if (w > (1 << 24) || h > (1 << 24) || w * h > (1LL << 30))
        fail(std::string(fmt) + ": image dimensions too large");
}

/// ITU-R BT.601 luma, the same weights used by JPEG, PIL and OpenCV.
template <class T>
inline T lumaOf(T r, T g, T b) {
    if constexpr (std::is_integral<T>::value) {
        const unsigned long long v = 299ULL * r + 587ULL * g + 114ULL * b;
        return static_cast<T>((v + 500) / 1000);
    } else {
        return static_cast<T>(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Metadata shared by the codecs
// ---------------------------------------------------------------------------

/// Information stored alongside the pixels of an image file. Decoders fill the fields they
/// find; encoders write the fields their format supports (PNG: ICC, EXIF, text, DPI, gamma;
/// JPEG: ICC, EXIF, comments, DPI).
struct ImageMetadata {
    std::vector<std::uint8_t> iccProfile;  ///< embedded ICC color profile (raw bytes), empty if none
    std::vector<std::uint8_t> exif;        ///< raw EXIF data (TIFF header onwards), empty if none
    int orientation = 1;                   ///< EXIF orientation 1..8 as stored in the file (1 = upright)
    double gamma = 0;                      ///< PNG gAMA file gamma (e.g. 0.45455); 0 = not specified
    int srgbIntent = -1;                   ///< PNG sRGB chunk rendering intent, -1 = absent
    double dpiX = 0, dpiY = 0;             ///< physical resolution in dots per inch, 0 = unknown
    std::vector<std::pair<std::string, std::string>> text;  ///< PNG text chunks; JPEG comments use key "comment"
    int bitDepth = 8;                      ///< bits per sample stored in the file (1-16)
};

namespace detail {

/// Locates the EXIF orientation tag (0x0112) in IFD0 of a TIFF/EXIF block.
/// Returns the byte offset of its 2-byte value, or 0 if not present/invalid.
inline std::size_t exifOrientationOffset(const std::vector<std::uint8_t>& t) {
    if (t.size() < 8) return 0;
    const bool le = t[0] == 'I' && t[1] == 'I';
    if (!le && !(t[0] == 'M' && t[1] == 'M')) return 0;
    auto u16 = [&](std::size_t p) -> unsigned {
        return le ? (t[p] | (t[p + 1] << 8)) : ((t[p] << 8) | t[p + 1]);
    };
    auto u32 = [&](std::size_t p) -> std::size_t {
        return le ? (static_cast<std::size_t>(u16(p)) | (static_cast<std::size_t>(u16(p + 2)) << 16))
                  : ((static_cast<std::size_t>(u16(p)) << 16) | u16(p + 2));
    };
    const std::size_t ifd = u32(4);
    if (ifd + 2 > t.size()) return 0;
    const unsigned count = u16(ifd);
    for (unsigned i = 0; i < count; ++i) {
        const std::size_t e = ifd + 2 + static_cast<std::size_t>(i) * 12;
        if (e + 12 > t.size()) return 0;
        if (u16(e) == 0x0112 && u16(e + 2) == 3) return e + 8;  // SHORT stored inline
    }
    return 0;
}

/// Reads the EXIF orientation (1..8), 1 if absent or invalid.
inline int exifOrientation(const std::vector<std::uint8_t>& tiff) {
    const std::size_t p = exifOrientationOffset(tiff);
    if (!p) return 1;
    const int v = tiff[0] == 'I' ? (tiff[p] | (tiff[p + 1] << 8)) : ((tiff[p] << 8) | tiff[p + 1]);
    return (v >= 1 && v <= 8) ? v : 1;
}

/// Rewrites the EXIF orientation tag in place (no-op if the tag is absent).
inline void setExifOrientation(std::vector<std::uint8_t>& tiff, int orientation) {
    const std::size_t p = exifOrientationOffset(tiff);
    if (!p) return;
    const bool le = tiff[0] == 'I';
    tiff[p] = static_cast<std::uint8_t>(le ? orientation : 0);
    tiff[p + 1] = static_cast<std::uint8_t>(le ? 0 : orientation);
}

}  // namespace detail

/// Transforms an image stored with EXIF orientation `orientation` (1..8) into its upright
/// form (1 = unchanged, 3 = rotate 180, 6 = rotate 90 clockwise, 8 = rotate 90
/// counter-clockwise, 2/4/5/7 = mirrored variants).
template <class T>
Image<T> applyExifOrientation(const Image<T>& src, int orientation) {
    if (orientation <= 1 || orientation > 8) return src;
    const int W = src.width(), H = src.height(), cn = src.channels();
    const bool swap = orientation >= 5;
    Image<T> out(swap ? H : W, swap ? W : H, cn);
    for (int y = 0; y < out.height(); ++y)
        for (int x = 0; x < out.width(); ++x) {
            int sx = x, sy = y;
            switch (orientation) {
                case 2: sx = W - 1 - x; break;
                case 3: sx = W - 1 - x; sy = H - 1 - y; break;
                case 4: sy = H - 1 - y; break;
                case 5: sx = y; sy = x; break;
                case 6: sx = y; sy = H - 1 - x; break;
                case 7: sx = W - 1 - y; sy = H - 1 - x; break;
                default: sx = W - 1 - y; sy = x; break;  // 8
            }
            std::memcpy(out.ptr(x, y), src.ptr(sx, sy), sizeof(T) * cn);
        }
    return out;
}

/// Reads a whole file into memory (throws ipf::Error on failure).
inline std::vector<std::uint8_t> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) detail::fail("cannot open file for reading: " + path);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

/// Writes a byte buffer to a file (throws ipf::Error on failure).
inline void writeFile(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) detail::fail("cannot open file for writing: " + path);
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!f) detail::fail("failed writing file: " + path);
}

/// Converts between the standard channel layouts used by image files:
/// 1 = gray, 2 = gray + alpha, 3 = RGB, 4 = RGBA. Gray is computed with
/// BT.601 luma weights, added alpha channels are fully opaque, and removed
/// alpha channels are simply dropped. desired == 0 returns the input unchanged.
template <class T>
Image<T> convertChannels(const Image<T>& src, int desired) {
    const int sc = src.channels();
    if (desired == 0 || desired == sc) return src;
    detail::require(desired >= 1 && desired <= 4, "convertChannels: desired channels must be 0..4");
    detail::require(sc >= 1 && sc <= 4, "convertChannels: source must have 1..4 channels");
    const T opaque = static_cast<T>(PixelTraits<T>::max());
    Image<T> out(src.width(), src.height(), desired);
    const T* s = src.data();
    T* d = out.data();
    for (std::size_t i = 0; i < src.pixelCount(); ++i, s += sc, d += desired) {
        T g, r, gg, b, a;
        if (sc <= 2) {
            g = r = gg = b = s[0];
            a = sc == 2 ? s[1] : opaque;
        } else {
            r = s[0];
            gg = s[1];
            b = s[2];
            g = detail::lumaOf(r, gg, b);
            a = sc == 4 ? s[3] : opaque;
        }
        switch (desired) {
            case 1: d[0] = g; break;
            case 2: d[0] = g; d[1] = a; break;
            case 3: d[0] = r; d[1] = gg; d[2] = b; break;
            default: d[0] = r; d[1] = gg; d[2] = b; d[3] = a; break;
        }
    }
    return out;
}

}  // namespace ipf
