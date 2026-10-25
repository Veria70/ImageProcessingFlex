// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf codec: PNG / APNG.
//  Decoder: all color types, bit depths 1-16, palette + tRNS, Adam7 interlacing, metadata
//  (iCCP, gAMA, sRGB, pHYs, tEXt/zTXt/iTXt, eXIf) and APNG animations (fully composited frames).
//  Encoder: 8/16-bit gray, gray+alpha, RGB, RGBA with adaptive filtering; lossless palette and
//  low-bit-depth output when possible; optional Adam7; metadata chunks; APNG animations.
#pragma once

#include <map>

#include "zlib.hpp"

namespace ipf {

/// Options for encodePNG.
struct PngOptions {
    int level = 6;                            ///< zlib compression level 0-9
    bool interlace = false;                   ///< write Adam7-interlaced
    bool optimize = true;                     ///< use palette / 1-, 2-, 4-bit gray when lossless
    const ImageMetadata* metadata = nullptr;  ///< ICC profile, EXIF, text, DPI, gamma/sRGB to embed
};

/// One frame of an animation: a full-canvas image and its display duration in seconds.
struct AnimationFrame {
    ImageU8 image;
    double delay = 0.1;
};

namespace detail {

inline int paeth(int a, int b, int c) {
    const int p = a + b - c;
    const int pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
    if (pa <= pb && pa <= pc) return a;
    return pb <= pc ? b : c;
}

inline void pngUnfilter(int filter, std::uint8_t* cur, const std::uint8_t* prev, std::size_t len, std::size_t bpp) {
    switch (filter) {
        case 0: break;
        case 1:
            for (std::size_t i = bpp; i < len; ++i) cur[i] = static_cast<std::uint8_t>(cur[i] + cur[i - bpp]);
            break;
        case 2:
            for (std::size_t i = 0; i < len; ++i) cur[i] = static_cast<std::uint8_t>(cur[i] + prev[i]);
            break;
        case 3:
            for (std::size_t i = 0; i < len; ++i) {
                const int a = i >= bpp ? cur[i - bpp] : 0;
                cur[i] = static_cast<std::uint8_t>(cur[i] + ((a + prev[i]) >> 1));
            }
            break;
        case 4:
            for (std::size_t i = 0; i < len; ++i) {
                const int a = i >= bpp ? cur[i - bpp] : 0;
                const int c = i >= bpp ? prev[i - bpp] : 0;
                cur[i] = static_cast<std::uint8_t>(cur[i] + paeth(a, prev[i], c));
            }
            break;
        default:
            fail("PNG: invalid filter type " + std::to_string(filter));
    }
}

inline std::string latin1ToUtf8(const std::uint8_t* p, std::size_t n) {
    std::string s;
    for (std::size_t i = 0; i < n; ++i) {
        if (p[i] < 0x80) {
            s += static_cast<char>(p[i]);
        } else {
            s += static_cast<char>(0xC0 | (p[i] >> 6));
            s += static_cast<char>(0x80 | (p[i] & 0x3F));
        }
    }
    return s;
}

struct PngHeader {
    std::uint32_t width = 0, height = 0;
    int depth = 0, ctype = -1, interlace = 0;
    std::vector<std::uint8_t> palette, trns;
    int samples() const { return ctype == 2 ? 3 : ctype == 4 ? 2 : ctype == 6 ? 4 : 1; }
};

struct ApngFrameInfo {
    std::uint32_t w = 0, h = 0, x = 0, y = 0;
    std::uint16_t delayNum = 0, delayDen = 0;
    int dispose = 0, blend = 0;
    std::vector<std::uint8_t> data;  // concatenated IDAT/fdAT payload
};

struct PngFile {
    PngHeader hdr;
    std::vector<std::uint8_t> idat;
    ImageMetadata meta;
    bool animated = false;
    std::uint32_t numPlays = 0;
    bool idatIsFrame = false;  // the default image is the first animation frame
    std::vector<ApngFrameInfo> frames;
};

inline PngFile parsePng(const std::vector<std::uint8_t>& bytes) {
    static const std::uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (bytes.size() < 8 || std::memcmp(bytes.data(), sig, 8) != 0) fail("PNG: bad signature");
    ByteReader rd(bytes.data(), bytes.size(), "PNG");
    rd.skip(8);
    PngFile f;
    PngHeader& hd = f.hdr;
    bool seenIHDR = false, seenIEND = false, seenIDAT = false;
    ApngFrameInfo* current = nullptr;
    while (!seenIEND) {
        const std::uint32_t len = rd.u32be();
        const std::size_t typePos = rd.pos();
        const std::uint8_t* type = rd.take(4);
        const std::uint8_t* body = rd.take(len);
        const std::uint32_t crc = rd.u32be();
        if (crc != crc32(bytes.data() + typePos, static_cast<std::size_t>(len) + 4))
            fail("PNG: CRC mismatch in chunk " + std::string(reinterpret_cast<const char*>(type), 4));
        const std::string t(reinterpret_cast<const char*>(type), 4);
        ByteReader b(body, len, "PNG chunk");
        if (t == "IHDR") {
            if (len != 13) fail("PNG: bad IHDR");
            hd.width = b.u32be();
            hd.height = b.u32be();
            hd.depth = b.u8();
            hd.ctype = b.u8();
            const int comp = b.u8(), filt = b.u8();
            hd.interlace = b.u8();
            if (comp != 0 || filt != 0 || hd.interlace > 1) fail("PNG: unsupported compression/filter/interlace method");
            seenIHDR = true;
        } else if (t == "PLTE") {
            hd.palette.assign(body, body + len);
        } else if (t == "tRNS") {
            hd.trns.assign(body, body + len);
        } else if (t == "IDAT") {
            f.idat.insert(f.idat.end(), body, body + len);
            if (current && !seenIDAT) f.idatIsFrame = true;
            seenIDAT = true;
        } else if (t == "IEND") {
            seenIEND = true;
        } else if (t == "acTL") {
            if (len >= 8) {
                f.animated = true;
                b.skip(4);
                f.numPlays = b.u32be();
            }
        } else if (t == "fcTL") {
            if (len < 26) fail("PNG: bad fcTL chunk");
            ApngFrameInfo fi;
            b.skip(4);  // sequence number
            fi.w = b.u32be();
            fi.h = b.u32be();
            fi.x = b.u32be();
            fi.y = b.u32be();
            fi.delayNum = b.u16be();
            fi.delayDen = b.u16be();
            fi.dispose = b.u8();
            fi.blend = b.u8();
            if (fi.dispose > 2 || fi.blend > 1) fail("PNG: invalid APNG dispose/blend operation");
            f.frames.push_back(std::move(fi));
            current = &f.frames.back();
        } else if (t == "fdAT") {
            if (len < 4 || !current) fail("PNG: fdAT without fcTL");
            current->data.insert(current->data.end(), body + 4, body + len);
        } else if (t == "gAMA") {
            if (len == 4) f.meta.gamma = b.u32be() / 100000.0;
        } else if (t == "sRGB") {
            if (len == 1) f.meta.srgbIntent = body[0];
        } else if (t == "pHYs") {
            if (len == 9) {
                const double px = b.u32be(), py = b.u32be();
                if (b.u8() == 1) {  // pixels per meter
                    f.meta.dpiX = px * 0.0254;
                    f.meta.dpiY = py * 0.0254;
                }
            }
        } else if (t == "iCCP") {
            const std::uint8_t* nul = static_cast<const std::uint8_t*>(std::memchr(body, 0, len));
            if (nul && nul + 2 <= body + len && nul[1] == 0) {
                const std::size_t off = static_cast<std::size_t>(nul - body) + 2;
                try {
                    f.meta.iccProfile = zlibDecompress(body + off, len - off);
                } catch (const Error&) {  // a broken ancillary chunk must not prevent decoding
                    f.meta.iccProfile.clear();
                }
            }
        } else if (t == "tEXt" || t == "zTXt" || t == "iTXt") {
            const std::uint8_t* nul = static_cast<const std::uint8_t*>(std::memchr(body, 0, len));
            if (nul) {
                const std::string key = latin1ToUtf8(body, static_cast<std::size_t>(nul - body));
                std::size_t off = static_cast<std::size_t>(nul - body) + 1;
                try {
                    if (t == "tEXt") {
                        f.meta.text.push_back({key, latin1ToUtf8(body + off, len - off)});
                    } else if (t == "zTXt" && off + 1 <= len) {
                        const auto txt = zlibDecompress(body + off + 1, len - off - 1);
                        f.meta.text.push_back({key, latin1ToUtf8(txt.data(), txt.size())});
                    } else if (t == "iTXt" && off + 2 <= len) {
                        const bool compressed = body[off] != 0;
                        off += 2;
                        for (int skip = 0; skip < 2 && off <= len; ++skip) {  // language tag, translated keyword
                            const std::uint8_t* z = static_cast<const std::uint8_t*>(std::memchr(body + off, 0, len - off));
                            off = z ? static_cast<std::size_t>(z - body) + 1 : len + 1;
                        }
                        if (off <= len) {
                            if (compressed) {
                                const auto txt = zlibDecompress(body + off, len - off);
                                f.meta.text.push_back({key, std::string(txt.begin(), txt.end())});
                            } else {
                                f.meta.text.push_back({key, std::string(body + off, body + len)});
                            }
                        }
                    }
                } catch (const Error&) {
                }
            }
        } else if (t == "eXIf") {
            f.meta.exif.assign(body, body + len);
            f.meta.orientation = exifOrientation(f.meta.exif);
        } else if (!(type[0] & 0x20)) {
            fail("PNG: unknown critical chunk " + t);
        }
        if (rd.eof() && !seenIEND) break;  // tolerate a missing IEND
    }
    if (!seenIHDR) fail("PNG: missing IHDR");
    checkDecodedSize(hd.width, hd.height, "PNG");
    const int ct = hd.ctype, d = hd.depth;
    if (ct != 0 && ct != 2 && ct != 3 && ct != 4 && ct != 6) fail("PNG: invalid color type");
    const bool depthOk = (ct == 0 && (d == 1 || d == 2 || d == 4 || d == 8 || d == 16)) ||
                         (ct == 3 && (d == 1 || d == 2 || d == 4 || d == 8)) ||
                         ((ct == 2 || ct == 4 || ct == 6) && (d == 8 || d == 16));
    if (!depthOk) fail("PNG: invalid bit depth for color type");
    if (ct == 3 && (hd.palette.empty() || hd.palette.size() % 3)) fail("PNG: missing or invalid palette");
    f.meta.bitDepth = d;
    if (f.animated && f.idatIsFrame && !f.frames.empty()) f.frames.front().data = f.idat;
    return f;
}

/// Expands zlib-compressed PNG image data of size w x h into 16-bit storage
/// (8-bit range unless the bit depth is 16; palette images become RGB(A)).
inline ImageU16 pngExpand(const PngHeader& hd, std::uint32_t W, std::uint32_t H, const std::vector<std::uint8_t>& zdata,
                          bool interlace) {
    const int samples = hd.samples(), depth = hd.depth, ctype = hd.ctype;
    const bool hasKey = !hd.trns.empty() && (ctype == 0 || ctype == 2);
    int outCn = samples;
    if (ctype == 3) outCn = hd.trns.empty() ? 3 : 4;
    else if (hasKey) outCn = samples + 1;
    const bool is16 = depth == 16;
    ImageU16 img(static_cast<int>(W), static_cast<int>(H), outCn);
    const int bitsPP = samples * depth;
    const std::size_t bpp = std::max(1, bitsPP / 8);
    static const int x0[7] = {0, 4, 0, 2, 0, 1, 0}, y0[7] = {0, 0, 4, 0, 2, 0, 1};
    static const int dx[7] = {8, 8, 4, 4, 2, 2, 1}, dy[7] = {8, 8, 8, 4, 4, 2, 2};
    const int passes = interlace ? 7 : 1;
    auto passDims = [&](int p, std::uint32_t& pw, std::uint32_t& ph) {
        if (!interlace) {
            pw = W;
            ph = H;
            return;
        }
        pw = W > static_cast<std::uint32_t>(x0[p]) ? (W - x0[p] + dx[p] - 1) / dx[p] : 0;
        ph = H > static_cast<std::uint32_t>(y0[p]) ? (H - y0[p] + dy[p] - 1) / dy[p] : 0;
    };
    std::size_t expected = 0;
    for (int p = 0; p < passes; ++p) {
        std::uint32_t pw, ph;
        passDims(p, pw, ph);
        if (pw && ph) expected += ph * (1 + (static_cast<std::size_t>(pw) * bitsPP + 7) / 8);
    }
    const std::vector<std::uint8_t> raw = zlibDecompress(zdata.data(), zdata.size(), expected);
    if (raw.size() < expected) fail("PNG: image data is truncated");
    const int maxNative = (1 << depth) - 1;
    const int scale8 = depth < 8 && ctype != 3 ? 255 / maxNative : 1;
    std::size_t off = 0;
    std::vector<std::uint8_t> prevRow, curRow;
    std::vector<int> px(static_cast<std::size_t>(samples));
    for (int p = 0; p < passes; ++p) {
        std::uint32_t pw, ph;
        passDims(p, pw, ph);
        if (!pw || !ph) continue;
        const std::size_t rowBytes = (static_cast<std::size_t>(pw) * bitsPP + 7) / 8;
        prevRow.assign(rowBytes, 0);
        for (std::uint32_t r = 0; r < ph; ++r) {
            const int filter = raw[off];
            curRow.assign(raw.begin() + static_cast<std::ptrdiff_t>(off + 1), raw.begin() + static_cast<std::ptrdiff_t>(off + 1 + rowBytes));
            off += rowBytes + 1;
            pngUnfilter(filter, curRow.data(), prevRow.data(), rowBytes, bpp);
            const int y = interlace ? y0[p] + static_cast<int>(r) * dy[p] : static_cast<int>(r);
            for (std::uint32_t c = 0; c < pw; ++c) {
                const int x = interlace ? x0[p] + static_cast<int>(c) * dx[p] : static_cast<int>(c);
                for (int s = 0; s < samples; ++s) {
                    const std::size_t si = static_cast<std::size_t>(c) * samples + s;
                    if (depth == 16) {
                        px[s] = (curRow[si * 2] << 8) | curRow[si * 2 + 1];
                    } else if (depth == 8) {
                        px[s] = curRow[si];
                    } else {
                        const std::size_t bit = si * depth;
                        px[s] = (curRow[bit / 8] >> (8 - depth - static_cast<int>(bit % 8))) & maxNative;
                    }
                }
                std::uint16_t* o = img.ptr(x, y);
                if (ctype == 3) {
                    const std::size_t e = static_cast<std::size_t>(px[0]);
                    if (e * 3 + 2 >= hd.palette.size()) fail("PNG: palette index out of range");
                    o[0] = hd.palette[e * 3];
                    o[1] = hd.palette[e * 3 + 1];
                    o[2] = hd.palette[e * 3 + 2];
                    if (outCn == 4) o[3] = e < hd.trns.size() ? hd.trns[e] : 255;
                } else {
                    for (int s = 0; s < samples; ++s) o[s] = static_cast<std::uint16_t>(px[s] * scale8);
                    if (hasKey) {
                        bool match = hd.trns.size() >= static_cast<std::size_t>(2 * samples);
                        for (int s = 0; s < samples && match; ++s)
                            match = px[s] == (((hd.trns[2 * s] << 8) | hd.trns[2 * s + 1]) & (is16 ? 0xFFFF : maxNative));
                        o[samples] = static_cast<std::uint16_t>(match ? 0 : (is16 ? 65535 : 255));
                    }
                }
            }
            std::swap(prevRow, curRow);
        }
    }
    return img;
}

/// Decodes a PNG into 16-bit storage; `is16` tells whether samples use the full 16-bit range.
inline ImageU16 decodePNGRaw(const std::vector<std::uint8_t>& bytes, bool& is16, ImageMetadata* meta = nullptr) {
    const PngFile f = parsePng(bytes);
    is16 = f.hdr.depth == 16;
    if (meta) *meta = f.meta;
    return pngExpand(f.hdr, f.hdr.width, f.hdr.height, f.idat, f.hdr.interlace != 0);
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Filters and compresses rows of packed samples. rows(p, r) must return a pointer to the
/// packed bytes of row r of Adam7 pass p (or of the image when not interlaced).
template <class RowFn>
std::vector<std::uint8_t> pngCompress(std::uint32_t W, std::uint32_t H, int bitsPP, bool interlace, bool adaptive,
                                      int level, RowFn rows) {
    static const int x0[7] = {0, 4, 0, 2, 0, 1, 0}, y0[7] = {0, 0, 4, 0, 2, 0, 1};
    static const int dx[7] = {8, 8, 4, 4, 2, 2, 1}, dy[7] = {8, 8, 8, 4, 4, 2, 2};
    const std::size_t bpp = static_cast<std::size_t>(std::max(1, bitsPP / 8));
    std::vector<std::uint8_t> raw;
    for (int p = 0; p < (interlace ? 7 : 1); ++p) {
        const std::uint32_t pw = interlace ? (W > static_cast<std::uint32_t>(x0[p]) ? (W - x0[p] + dx[p] - 1) / dx[p] : 0) : W;
        const std::uint32_t ph = interlace ? (H > static_cast<std::uint32_t>(y0[p]) ? (H - y0[p] + dy[p] - 1) / dy[p] : 0) : H;
        if (!pw || !ph) continue;
        const std::size_t rowBytes = (static_cast<std::size_t>(pw) * bitsPP + 7) / 8;
        std::vector<std::uint8_t> prev(rowBytes, 0), best(rowBytes), trial(rowBytes);
        for (std::uint32_t r = 0; r < ph; ++r) {
            const std::uint8_t* cur = rows(p, r);
            int bestFilter = 0;
            best.assign(cur, cur + rowBytes);
            if (adaptive) {
                // filter minimizing the sum of absolute signed residuals (libpng heuristic)
                unsigned long long bestScore = ~0ULL;
                for (int f = 0; f < 5; ++f) {
                    unsigned long long score = 0;
                    for (std::size_t i = 0; i < rowBytes; ++i) {
                        const int a = i >= bpp ? cur[i - bpp] : 0, b = prev[i], c = i >= bpp ? prev[i - bpp] : 0;
                        int pred = 0;
                        switch (f) {
                            case 1: pred = a; break;
                            case 2: pred = b; break;
                            case 3: pred = (a + b) >> 1; break;
                            case 4: pred = paeth(a, b, c); break;
                            default: break;
                        }
                        trial[i] = static_cast<std::uint8_t>(cur[i] - pred);
                        score += static_cast<unsigned long long>(std::abs(static_cast<int>(static_cast<std::int8_t>(trial[i]))));
                    }
                    if (score < bestScore) {
                        bestScore = score;
                        bestFilter = f;
                        best = trial;
                    }
                }
            }
            raw.push_back(static_cast<std::uint8_t>(bestFilter));
            raw.insert(raw.end(), best.begin(), best.end());
            prev.assign(cur, cur + rowBytes);
        }
    }
    return zlibCompress(raw.data(), raw.size(), level);
}

inline void pngChunk(std::vector<std::uint8_t>& out, const char* type, const std::uint8_t* body, std::size_t len) {
    put32be(out, static_cast<std::uint32_t>(len));
    const std::size_t start = out.size();
    out.insert(out.end(), type, type + 4);
    if (len) out.insert(out.end(), body, body + len);
    put32be(out, crc32(out.data() + start, len + 4));
}
inline void pngChunk(std::vector<std::uint8_t>& out, const char* type, const std::vector<std::uint8_t>& body) {
    pngChunk(out, type, body.data(), body.size());
}

inline void pngWriteMetadata(std::vector<std::uint8_t>& out, const ImageMetadata& m, bool beforePalette) {
    if (beforePalette) {
        if (m.gamma > 0) {
            std::vector<std::uint8_t> b;
            put32be(b, static_cast<std::uint32_t>(std::lround(m.gamma * 100000)));
            pngChunk(out, "gAMA", b);
        }
        if (!m.iccProfile.empty()) {
            const char name[] = "ICC Profile";
            std::vector<std::uint8_t> b(name, name + sizeof(name));  // includes the terminating 0
            b.push_back(0);                                           // compression method: zlib
            const auto z = zlibCompress(m.iccProfile.data(), m.iccProfile.size(), 9);
            b.insert(b.end(), z.begin(), z.end());
            pngChunk(out, "iCCP", b);
        } else if (m.srgbIntent >= 0 && m.srgbIntent <= 3) {
            const std::uint8_t intent = static_cast<std::uint8_t>(m.srgbIntent);
            pngChunk(out, "sRGB", &intent, 1);
        }
        return;
    }
    if (m.dpiX > 0 && m.dpiY > 0) {
        std::vector<std::uint8_t> b;
        put32be(b, static_cast<std::uint32_t>(std::lround(m.dpiX / 0.0254)));
        put32be(b, static_cast<std::uint32_t>(std::lround(m.dpiY / 0.0254)));
        b.push_back(1);
        pngChunk(out, "pHYs", b);
    }
    if (!m.exif.empty()) pngChunk(out, "eXIf", m.exif);
    for (const auto& kv : m.text) {
        if (kv.first.empty() || kv.first.size() > 79) continue;
        const bool ascii = std::all_of(kv.second.begin(), kv.second.end(), [](char ch) { return static_cast<unsigned char>(ch) < 0x80; }) &&
                           std::all_of(kv.first.begin(), kv.first.end(), [](char ch) { return static_cast<unsigned char>(ch) < 0x80; });
        std::vector<std::uint8_t> b(kv.first.begin(), kv.first.end());
        b.push_back(0);
        if (ascii) {
            b.insert(b.end(), kv.second.begin(), kv.second.end());
            pngChunk(out, "tEXt", b);
        } else {  // UTF-8 text: uncompressed iTXt with empty language tag / translated keyword
            b.push_back(0);
            b.push_back(0);
            b.push_back(0);
            b.push_back(0);
            b.insert(b.end(), kv.second.begin(), kv.second.end());
            pngChunk(out, "iTXt", b);
        }
    }
}

/// Packed PNG representation of an image (possibly palette / low bit depth).
struct PngPacked {
    int ctype = 0, depth = 8;
    std::vector<std::uint8_t> palette, trns;  // for ctype 3
    std::vector<std::uint8_t> data;            // full-resolution packed rows, rowBytes each
    std::size_t rowBytes = 0;
};

template <class T>
PngPacked pngPack(const Image<T>& img, bool optimize) {
    PngPacked pk;
    const int W = img.width(), H = img.height(), cn = img.channels();
    static const int ctypes[5] = {0, 0, 4, 2, 6};
    pk.ctype = ctypes[cn];
    pk.depth = static_cast<int>(sizeof(T)) * 8;
    std::vector<std::uint8_t> index;  // palette indices or low-bit gray values per pixel
    if (sizeof(T) == 1 && optimize) {
        if (cn == 1) {
            int d = 8;
            for (int cand : {1, 2, 4}) {
                const int step = 255 / ((1 << cand) - 1);
                if (std::all_of(img.begin(), img.end(), [step](T v) { return static_cast<int>(v) % step == 0; })) {
                    d = cand;
                    break;
                }
            }
            if (d < 8) {
                pk.depth = d;
                const int step = 255 / ((1 << d) - 1);
                index.resize(img.pixelCount());
                for (std::size_t i = 0; i < index.size(); ++i) index[i] = static_cast<std::uint8_t>(img.data()[i] / step);
            }
        } else if (cn >= 3) {
            std::map<std::uint32_t, int> colors;
            bool fits = true;
            for (std::size_t i = 0; i < img.pixelCount() && fits; ++i) {
                const T* p = img.data() + i * cn;
                const std::uint32_t key = (static_cast<std::uint32_t>(cn == 4 ? p[3] : 255) << 24) |
                                          (static_cast<std::uint32_t>(p[0]) << 16) | (static_cast<std::uint32_t>(p[1]) << 8) | p[2];
                colors.emplace(key, 0);
                fits = colors.size() <= 256;
            }
            if (fits) {
                // keys sort translucent colors first, so tRNS can stop at the last translucent entry
                int n = 0, lastTranslucent = -1;
                for (auto& kv : colors) {
                    kv.second = n;
                    const std::uint8_t alpha = static_cast<std::uint8_t>(kv.first >> 24);
                    pk.palette.push_back(static_cast<std::uint8_t>(kv.first >> 16));
                    pk.palette.push_back(static_cast<std::uint8_t>(kv.first >> 8));
                    pk.palette.push_back(static_cast<std::uint8_t>(kv.first));
                    pk.trns.push_back(alpha);
                    if (alpha != 255) lastTranslucent = n;
                    ++n;
                }
                // RGBA input keeps a tRNS chunk (at least one entry) so it decodes back to 4 channels
                if (cn == 4) pk.trns.resize(static_cast<std::size_t>(std::max(1, lastTranslucent + 1)));
                else pk.trns.clear();
                pk.ctype = 3;
                pk.depth = n <= 2 ? 1 : n <= 4 ? 2 : n <= 16 ? 4 : 8;
                index.resize(img.pixelCount());
                for (std::size_t i = 0; i < index.size(); ++i) {
                    const T* p = img.data() + i * cn;
                    const std::uint32_t key = (static_cast<std::uint32_t>(cn == 4 ? p[3] : 255) << 24) |
                                              (static_cast<std::uint32_t>(p[0]) << 16) | (static_cast<std::uint32_t>(p[1]) << 8) | p[2];
                    index[i] = static_cast<std::uint8_t>(colors[key]);
                }
            }
        }
    }
    const int bitsPP = index.empty() ? cn * pk.depth : pk.depth;
    pk.rowBytes = (static_cast<std::size_t>(W) * bitsPP + 7) / 8;
    pk.data.assign(pk.rowBytes * H, 0);
    for (int y = 0; y < H; ++y) {
        std::uint8_t* row = &pk.data[static_cast<std::size_t>(y) * pk.rowBytes];
        if (!index.empty()) {
            const std::uint8_t* iv = &index[static_cast<std::size_t>(y) * W];
            for (int x = 0; x < W; ++x) {
                const std::size_t bit = static_cast<std::size_t>(x) * pk.depth;
                row[bit / 8] = static_cast<std::uint8_t>(row[bit / 8] | (iv[x] << (8 - pk.depth - static_cast<int>(bit % 8))));
            }
        } else {
            const T* r = img.row(y);
            for (std::size_t i = 0; i < img.stride(); ++i) {
                if (sizeof(T) == 2) {
                    row[i * 2] = static_cast<std::uint8_t>(static_cast<unsigned>(r[i]) >> 8);
                    row[i * 2 + 1] = static_cast<std::uint8_t>(static_cast<unsigned>(r[i]) & 0xFF);
                } else {
                    row[i] = static_cast<std::uint8_t>(r[i]);
                }
            }
        }
    }
    return pk;
}

/// Compresses packed rows, optionally Adam7-interlaced.
inline std::vector<std::uint8_t> pngCompressPacked(const PngPacked& pk, std::uint32_t W, std::uint32_t H, int bitsPP,
                                                   bool interlace, int level) {
    static const int x0[7] = {0, 4, 0, 2, 0, 1, 0}, y0[7] = {0, 0, 4, 0, 2, 0, 1};
    static const int dx[7] = {8, 8, 4, 4, 2, 2, 1}, dy[7] = {8, 8, 8, 4, 4, 2, 2};
    const bool adaptive = level > 0 && pk.ctype != 3 && pk.depth >= 8;  // libpng: no filters for palette/low-bit
    std::vector<std::uint8_t> passRow;
    return pngCompress(W, H, bitsPP, interlace, adaptive, level, [&](int p, std::uint32_t r) -> const std::uint8_t* {
        if (!interlace) return &pk.data[static_cast<std::size_t>(r) * pk.rowBytes];
        // gather the pixels of row r of pass p
        const std::uint32_t y = static_cast<std::uint32_t>(y0[p]) + r * static_cast<std::uint32_t>(dy[p]);
        const std::uint8_t* src = &pk.data[static_cast<std::size_t>(y) * pk.rowBytes];
        const std::uint32_t pw = (W - x0[p] + dx[p] - 1) / dx[p];
        passRow.assign((static_cast<std::size_t>(pw) * bitsPP + 7) / 8, 0);
        for (std::uint32_t c = 0; c < pw; ++c) {
            const std::size_t x = static_cast<std::size_t>(x0[p]) + c * static_cast<std::size_t>(dx[p]);
            if (bitsPP >= 8) {
                const std::size_t nb = static_cast<std::size_t>(bitsPP / 8);
                std::memcpy(&passRow[c * nb], src + x * nb, nb);
            } else {
                const std::size_t sb = x * bitsPP, db = static_cast<std::size_t>(c) * bitsPP;
                const int v = (src[sb / 8] >> (8 - bitsPP - static_cast<int>(sb % 8))) & ((1 << bitsPP) - 1);
                passRow[db / 8] = static_cast<std::uint8_t>(passRow[db / 8] | (v << (8 - bitsPP - static_cast<int>(db % 8))));
            }
        }
        return passRow.data();
    });
}

template <class T>
std::vector<std::uint8_t> encodePNGImpl(const Image<T>& img, const PngOptions& opt) {
    detail::require(!img.empty(), "encodePNG: empty image");
    detail::require(img.channels() >= 1 && img.channels() <= 4, "encodePNG: image must have 1-4 channels");
    const PngPacked pk = pngPack(img, opt.optimize);
    const std::uint32_t W = static_cast<std::uint32_t>(img.width()), H = static_cast<std::uint32_t>(img.height());
    const int bitsPP = pk.ctype == 3 || (pk.ctype == 0 && pk.depth < 8) ? pk.depth : img.channels() * pk.depth;
    const std::vector<std::uint8_t> z = pngCompressPacked(pk, W, H, bitsPP, opt.interlace, opt.level);
    std::vector<std::uint8_t> out{137, 80, 78, 71, 13, 10, 26, 10};
    std::vector<std::uint8_t> ihdr;
    put32be(ihdr, W);
    put32be(ihdr, H);
    put8(ihdr, static_cast<unsigned>(pk.depth));
    put8(ihdr, static_cast<unsigned>(pk.ctype));
    put8(ihdr, 0);
    put8(ihdr, 0);
    put8(ihdr, opt.interlace ? 1u : 0u);
    pngChunk(out, "IHDR", ihdr);
    if (opt.metadata) pngWriteMetadata(out, *opt.metadata, true);
    if (pk.ctype == 3) {
        pngChunk(out, "PLTE", pk.palette);
        if (!pk.trns.empty()) pngChunk(out, "tRNS", pk.trns);
    }
    if (opt.metadata) pngWriteMetadata(out, *opt.metadata, false);
    const std::size_t kMaxChunk = 1 << 20;
    for (std::size_t p = 0; p < z.size(); p += kMaxChunk) pngChunk(out, "IDAT", z.data() + p, std::min(kMaxChunk, z.size() - p));
    pngChunk(out, "IEND", nullptr, 0);
    return out;
}

}  // namespace detail

/// Decodes a PNG to 8 bits per sample. Channels: 1 gray, 2 gray+alpha, 3 RGB, 4 RGBA
/// (palette images become RGB, or RGBA when they carry transparency).
inline ImageU8 decodePNG(const std::vector<std::uint8_t>& bytes, ImageMetadata* meta = nullptr) {
    bool is16 = false;
    const ImageU16 raw = detail::decodePNGRaw(bytes, is16, meta);
    return raw.convertTo<std::uint8_t>(is16 ? 255.0 / 65535.0 : 1.0);
}

/// Decodes a PNG to 16 bits per sample (8-bit files are scaled by 257).
inline ImageU16 decodePNG16(const std::vector<std::uint8_t>& bytes, ImageMetadata* meta = nullptr) {
    bool is16 = false;
    ImageU16 raw = detail::decodePNGRaw(bytes, is16, meta);
    if (!is16)
        for (auto& v : raw) v = static_cast<std::uint16_t>(v * 257);
    return raw;
}

/// Encodes an 8- or 16-bit image (1-4 channels) as PNG with the given options.
inline std::vector<std::uint8_t> encodePNG(const ImageU8& img, const PngOptions& options) {
    return detail::encodePNGImpl(img, options);
}
inline std::vector<std::uint8_t> encodePNG(const ImageU16& img, const PngOptions& options) {
    return detail::encodePNGImpl(img, options);
}
/// Encodes an 8-bit image (1-4 channels) as PNG. level: zlib level 0-9.
inline std::vector<std::uint8_t> encodePNG(const ImageU8& img, int level = 6) {
    PngOptions o;
    o.level = level;
    return encodePNG(img, o);
}
/// Encodes a 16-bit image (1-4 channels) as a 16-bit PNG.
inline std::vector<std::uint8_t> encodePNG(const ImageU16& img, int level = 6) {
    PngOptions o;
    o.level = level;
    return encodePNG(img, o);
}

// ---------------------------------------------------------------------------
// APNG
// ---------------------------------------------------------------------------

/// Decodes an animated PNG into fully composited RGBA frames (dispose and blend operations
/// applied). A non-animated PNG yields a single frame. `loops` receives the play count
/// (0 = infinite).
inline std::vector<AnimationFrame> decodeAPNG(const std::vector<std::uint8_t>& bytes, int* loops = nullptr,
                                              ImageMetadata* meta = nullptr) {
    const detail::PngFile f = detail::parsePng(bytes);
    if (meta) *meta = f.meta;
    if (loops) *loops = static_cast<int>(f.numPlays);
    std::vector<AnimationFrame> out;
    auto toRgba8 = [&](const ImageU16& im) {
        const ImageU8 b = im.convertTo<std::uint8_t>(f.hdr.depth == 16 ? 255.0 / 65535.0 : 1.0);
        return convertChannels(b, 4);
    };
    if (!f.animated || f.frames.empty()) {
        out.push_back({toRgba8(detail::pngExpand(f.hdr, f.hdr.width, f.hdr.height, f.idat, f.hdr.interlace != 0)), 0.0});
        return out;
    }
    const int W = static_cast<int>(f.hdr.width), H = static_cast<int>(f.hdr.height);
    ImageU8 canvas(W, H, 4, 0);
    for (std::size_t i = 0; i < f.frames.size(); ++i) {
        const detail::ApngFrameInfo& fi = f.frames[i];
        if (fi.data.empty()) continue;
        if (fi.w == 0 || fi.h == 0 || fi.x + fi.w > f.hdr.width || fi.y + fi.h > f.hdr.height)
            detail::fail("PNG: APNG frame outside the canvas");
        const ImageU8 sub = toRgba8(detail::pngExpand(f.hdr, fi.w, fi.h, fi.data, f.hdr.interlace != 0));
        const Rect region(static_cast<int>(fi.x), static_cast<int>(fi.y), static_cast<int>(fi.w), static_cast<int>(fi.h));
        int dispose = fi.dispose;
        if (i == 0 && dispose == 2) dispose = 1;  // "previous" on the first frame means "background"
        const ImageU8 saved = dispose == 2 ? canvas.crop(region) : ImageU8();
        for (int y = 0; y < region.height; ++y)
            for (int x = 0; x < region.width; ++x) {
                const std::uint8_t* s = sub.ptr(x, y);
                std::uint8_t* d = canvas.ptr(region.x + x, region.y + y);
                if (fi.blend == 0 || s[3] == 255) {
                    std::memcpy(d, s, 4);
                } else if (s[3] != 0) {  // alpha "over" compositing (APNG spec)
                    const double sa = s[3] / 255.0, da = d[3] / 255.0, oa = sa + da * (1 - sa);
                    for (int c = 0; c < 3; ++c) d[c] = saturate_cast<std::uint8_t>((s[c] * sa + d[c] * da * (1 - sa)) / oa);
                    d[3] = saturate_cast<std::uint8_t>(oa * 255);
                }
            }
        out.push_back({canvas, fi.delayNum / (fi.delayDen ? static_cast<double>(fi.delayDen) : 100.0)});
        if (dispose == 1) {
            for (int y = 0; y < region.height; ++y) std::memset(canvas.ptr(region.x, region.y + y), 0, static_cast<std::size_t>(region.width) * 4);
        } else if (dispose == 2) {
            canvas.paste(saved, region.x, region.y);
        }
    }
    return out;
}

/// Encodes frames (all the same size, 1-4 channels) as an animated PNG. loops: 0 = infinite.
inline std::vector<std::uint8_t> encodeAPNG(const std::vector<AnimationFrame>& frames, int loops = 0, int level = 6) {
    detail::require(!frames.empty(), "encodeAPNG: no frames");
    const int W = frames[0].image.width(), H = frames[0].image.height();
    bool alpha = false;
    std::vector<ImageU8> rgba;
    for (const auto& fr : frames) {
        detail::require(fr.image.width() == W && fr.image.height() == H && !fr.image.empty(),
                        "encodeAPNG: all frames must have the same, non-zero size");
        rgba.push_back(convertChannels(fr.image, 4));
        for (std::size_t i = 0; i < rgba.back().pixelCount() && !alpha; ++i) alpha = rgba.back().data()[i * 4 + 3] != 255;
    }
    std::vector<std::uint8_t> out{137, 80, 78, 71, 13, 10, 26, 10};
    std::vector<std::uint8_t> b;
    detail::put32be(b, static_cast<std::uint32_t>(W));
    detail::put32be(b, static_cast<std::uint32_t>(H));
    detail::put8(b, 8);
    detail::put8(b, alpha ? 6u : 2u);
    detail::put8(b, 0);
    detail::put8(b, 0);
    detail::put8(b, 0);
    detail::pngChunk(out, "IHDR", b);
    b.clear();
    detail::put32be(b, static_cast<std::uint32_t>(frames.size()));
    detail::put32be(b, static_cast<std::uint32_t>(std::max(0, loops)));
    detail::pngChunk(out, "acTL", b);
    std::uint32_t seq = 0;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        b.clear();
        detail::put32be(b, seq++);
        detail::put32be(b, static_cast<std::uint32_t>(W));
        detail::put32be(b, static_cast<std::uint32_t>(H));
        detail::put32be(b, 0);
        detail::put32be(b, 0);
        const double ms = std::max(0.0, frames[i].delay) * 1000.0;
        const bool coarse = ms > 65535;
        detail::put16be(b, static_cast<unsigned>(std::min(65535.0, std::round(coarse ? ms / 10 : ms))));
        detail::put16be(b, coarse ? 100u : 1000u);
        detail::put8(b, 0);  // dispose: none
        detail::put8(b, 0);  // blend: source
        detail::pngChunk(out, "fcTL", b);
        const ImageU8 img = alpha ? rgba[i] : convertChannels(rgba[i], 3);
        const detail::PngPacked pk = detail::pngPack(img, false);
        const auto z = detail::pngCompressPacked(pk, static_cast<std::uint32_t>(W), static_cast<std::uint32_t>(H),
                                                 img.channels() * 8, false, level);
        if (i == 0) {
            detail::pngChunk(out, "IDAT", z);
        } else {
            b.clear();
            detail::put32be(b, seq++);
            b.insert(b.end(), z.begin(), z.end());
            detail::pngChunk(out, "fdAT", b);
        }
    }
    detail::pngChunk(out, "IEND", nullptr, 0);
    return out;
}

}  // namespace ipf
