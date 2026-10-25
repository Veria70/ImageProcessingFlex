// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf codec: JPEG decoder and baseline encoder.
//  Decoder: baseline, extended (12-bit) and progressive DCT with Huffman or arithmetic coding,
//  lossless JPEG (2-16 bit, all 7 predictors), any chroma subsampling, restart markers,
//  gray / YCbCr / RGB / Adobe CMYK & YCCK, plus EXIF, ICC profile, JFIF density and comments.
//  (Not supported: hierarchical JPEG and lossless + arithmetic coding, which no common
//  encoder produces.)
//  Encoder: baseline, quality 1-100, 4:2:0 or 4:4:4 chroma, optional ICC/EXIF/DPI/comments.
#pragma once

#include "common.hpp"

namespace ipf {

namespace detail {
namespace jpg {

// Natural (row-major) index of the k-th coefficient in zigzag order; padded for
// robustness against corrupt streams that run past coefficient 63.
inline constexpr std::uint8_t kZigzag[80] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,
    6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31,
    39, 46, 53, 60, 61, 54, 47, 55, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63};

// Annex K standard tables (used by the encoder).
inline constexpr std::uint8_t kStdLumQ[64] = {
    16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,  14, 13, 16, 24, 40,  57,
    69, 56, 14, 17, 22,  29,  51,  87,  80, 62, 18, 22, 37,  56,  68,  109, 103, 77, 24, 35, 55,  64,
    81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};
inline constexpr std::uint8_t kStdChromQ[64] = {
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99,
    99, 99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};
inline constexpr std::uint8_t kDcLumBits[16] = {0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
inline constexpr std::uint8_t kDcChromBits[16] = {0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
inline constexpr std::uint8_t kDcVals[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
inline constexpr std::uint8_t kAcLumBits[16] = {0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d};
inline constexpr std::uint8_t kAcLumVals[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71,
    0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72,
    0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83,
    0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3,
    0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};
inline constexpr std::uint8_t kAcChromBits[16] = {0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77};
inline constexpr std::uint8_t kAcChromVals[162] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22,
    0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1,
    0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36,
    0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a,
    0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a,
    0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba,
    0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};

/// cos table for the 8-point DCT: kCos[u][x] = C(u)/2 * cos((2x+1) u pi / 16).
inline const float* dctMatrix() {
    static const auto m = [] {
        std::vector<float> t(64);
        for (int u = 0; u < 8; ++u)
            for (int x = 0; x < 8; ++x) {
                const double cu = u == 0 ? std::sqrt(0.5) : 1.0;
                t[u * 8 + x] = static_cast<float>(cu / 2.0 * std::cos((2 * x + 1) * u * kPi / 16.0));
            }
        return t;
    }();
    return m.data();
}

/// Inverse DCT of a dequantized block (natural order), level-shifted by `center` and
/// clamped to [0, maxVal] (center 128 / maxVal 255 for 8-bit, 2048 / 4095 for 12-bit).
inline void idct8x8(const float* in, std::uint16_t* out, std::size_t outStride, float center, float maxVal) {
    const float* c = dctMatrix();
    float tmp[64];
    for (int v = 0; v < 8; ++v) {  // rows: tmp[v][x] = sum_u c[u][x] * F[v][u]
        const float* f = in + v * 8;
        bool zero = true;
        for (int u = 1; u < 8; ++u)
            if (f[u] != 0.f) { zero = false; break; }
        for (int x = 0; x < 8; ++x) {
            float s = f[0] * c[x];
            if (!zero)
                for (int u = 1; u < 8; ++u) s += f[u] * c[u * 8 + x];
            tmp[v * 8 + x] = s;
        }
    }
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y) {
            float s = center;
            for (int v = 0; v < 8; ++v) s += tmp[v * 8 + x] * c[v * 8 + y];
            s = s < 0.f ? 0.f : (s > maxVal ? maxVal : s);
            out[y * outStride + x] = static_cast<std::uint16_t>(s + 0.5f);
        }
}

/// Forward DCT of a level-shifted 8x8 block (natural order).
inline void fdct8x8(const float* in, float* out) {
    const float* c = dctMatrix();
    float tmp[64];
    for (int y = 0; y < 8; ++y)
        for (int u = 0; u < 8; ++u) {
            float s = 0.f;
            for (int x = 0; x < 8; ++x) s += in[y * 8 + x] * c[u * 8 + x];
            tmp[y * 8 + u] = s;
        }
    for (int u = 0; u < 8; ++u)
        for (int v = 0; v < 8; ++v) {
            float s = 0.f;
            for (int y = 0; y < 8; ++y) s += tmp[y * 8 + u] * c[v * 8 + y];
            out[v * 8 + u] = s;
        }
}

struct HuffTable {
    bool defined = false;
    std::uint8_t vals[256] = {0};
    int maxcode[18] = {0};
    int valptr[17] = {0};
    int mincode[17] = {0};
    std::uint16_t fast[512] = {0};  // (length << 8) | value for codes up to 9 bits

    void build(const std::uint8_t* bits, const std::uint8_t* v, int nvals) {
        std::memcpy(vals, v, static_cast<std::size_t>(nvals));
        std::fill(std::begin(fast), std::end(fast), std::uint16_t(0));
        int code = 0, k = 0;
        for (int l = 1; l <= 16; ++l) {
            valptr[l] = k;
            mincode[l] = code;
            for (int i = 0; i < bits[l - 1]; ++i, ++k, ++code)
                if (l <= 9) {
                    const int shift = 9 - l;
                    for (int j = 0; j < (1 << shift); ++j)
                        fast[(code << shift) | j] = static_cast<std::uint16_t>((l << 8) | vals[k]);
                }
            maxcode[l] = bits[l - 1] ? code - 1 : -1;
            if (code > (1 << l)) fail("JPEG: invalid Huffman table");
            code <<= 1;
        }
        maxcode[17] = 0x7FFFFFFF;
        defined = true;
    }
};

class BitReader {
public:
    BitReader(const std::uint8_t* p, std::size_t n, std::size_t pos) : p_(p), n_(n), pos_(pos) {}
    int bits(int n) {
        if (!n) return 0;
        fill();
        const int v = static_cast<int>(buf_ >> (32 - n));
        buf_ <<= n;
        cnt_ -= n;
        return v;
    }
    int bit() { return bits(1); }
    /// Reads n bits and sign-extends them per JPEG's magnitude category coding.
    int receiveExtend(int n) {
        if (!n) return 0;
        const int v = bits(n);
        return v < (1 << (n - 1)) ? v - (1 << n) + 1 : v;
    }
    int decode(const HuffTable& h) {
        if (!h.defined) fail("JPEG: undefined Huffman table");
        fill();
        const std::uint16_t e = h.fast[buf_ >> 23];
        if (e) {
            const int l = e >> 8;
            buf_ <<= l;
            cnt_ -= l;
            return e & 0xFF;
        }
        for (int l = 10; l <= 16; ++l) {
            const int code = static_cast<int>(buf_ >> (32 - l));
            if (code <= h.maxcode[l]) {
                buf_ <<= l;
                cnt_ -= l;
                const int idx = h.valptr[l] + code - h.mincode[l];
                if (idx < 0 || idx > 255) fail("JPEG: corrupt Huffman data");
                return h.vals[idx];
            }
        }
        fail("JPEG: corrupt Huffman data");
    }
    /// Discards buffered bits and consumes the next RSTn marker if present.
    void restart() {
        buf_ = 0;
        cnt_ = 0;
        marker_ = false;
        while (pos_ + 1 < n_ && !(p_[pos_] == 0xFF && p_[pos_ + 1] != 0 && p_[pos_ + 1] != 0xFF)) ++pos_;
        if (pos_ + 1 < n_ && p_[pos_ + 1] >= 0xD0 && p_[pos_ + 1] <= 0xD7) pos_ += 2;
    }
    /// Position of the first byte after the entropy-coded segment (the next marker).
    std::size_t endPos() {
        std::size_t p = pos_;
        while (p + 1 < n_ && !(p_[p] == 0xFF && p_[p + 1] != 0 && !(p_[p + 1] >= 0xD0 && p_[p + 1] <= 0xD7))) ++p;
        return p;
    }

private:
    void fill() {
        while (cnt_ <= 24) {
            std::uint32_t b = 0;
            if (!marker_ && pos_ < n_) {
                b = p_[pos_];
                if (b == 0xFF) {
                    const std::uint8_t nx = pos_ + 1 < n_ ? p_[pos_ + 1] : 0xD9;
                    if (nx == 0x00) {
                        pos_ += 2;
                    } else {
                        marker_ = true;  // stop at the marker and feed zeros from now on
                        b = 0;
                    }
                } else {
                    ++pos_;
                }
            }
            buf_ |= b << (24 - cnt_);
            cnt_ += 8;
        }
    }
    const std::uint8_t* p_;
    std::size_t n_;
    std::size_t pos_;
    std::uint32_t buf_ = 0;
    int cnt_ = 0;
    bool marker_ = false;
};

// Probability estimation state machine of the arithmetic coder (ITU-T T.81 Table D.2),
// packed as (Qe << 16) | (Next_Index_MPS << 8) | (Switch_MPS << 7) | Next_Index_LPS.
// Entry 113 is the fixed 0.5 probability state used for sign/refinement bits.
inline constexpr std::uint32_t kQeTable[114] = {
    0x5A1D0181, 0x2586020E, 0x11140310, 0x080B0412, 0x03D80514, 0x01DA0617,
    0x00E50719, 0x006F081C, 0x0036091E, 0x001A0A21, 0x000D0B23, 0x00060C09,
    0x00030D0A, 0x00010D0C, 0x5A7F0F8F, 0x3F251024, 0x2CF21126, 0x207C1227,
    0x17B91328, 0x1182142A, 0x0CEF152B, 0x09A1162D, 0x072F172E, 0x055C1830,
    0x04061931, 0x03031A33, 0x02401B34, 0x01B11C36, 0x01441D38, 0x00F51E39,
    0x00B71F3B, 0x008A203C, 0x0068213E, 0x004E223F, 0x003B2320, 0x002C0921,
    0x5AE125A5, 0x484C2640, 0x3A0D2741, 0x2EF12843, 0x261F2944, 0x1F332A45,
    0x19A82B46, 0x15182C48, 0x11772D49, 0x0E742E4A, 0x0BFB2F4B, 0x09F8304D,
    0x0861314E, 0x0706324F, 0x05CD3330, 0x04DE3432, 0x040F3532, 0x03633633,
    0x02D43734, 0x025C3835, 0x01F83936, 0x01A43A37, 0x01603B38, 0x01253C39,
    0x00F63D3A, 0x00CB3E3B, 0x00AB3F3D, 0x008F203D, 0x5B1241C1, 0x4D044250,
    0x412C4351, 0x37D84452, 0x2FE84553, 0x293C4654, 0x23794756, 0x1EDF4857,
    0x1AA94957, 0x174E4A48, 0x14244B48, 0x119C4C4A, 0x0F6B4D4A, 0x0D514E4B,
    0x0BB64F4D, 0x0A40304D, 0x583251D0, 0x4D1C5258, 0x438E5359, 0x3BDD545A,
    0x34EE555B, 0x2EAE565C, 0x299A575D, 0x25164756, 0x557059D8, 0x4CA95A5F,
    0x44D95B60, 0x3E225C61, 0x38245D63, 0x32B45E63, 0x2E17565D, 0x56A860DF,
    0x4F466165, 0x47E56266, 0x41CF6367, 0x3C3D6468, 0x375E5D63, 0x52316669,
    0x4C0F676A, 0x4639686B, 0x415E6367, 0x56276AE9, 0x50E76B6C, 0x4B85676D,
    0x55976D6E, 0x504F6B6F, 0x5A106FEE, 0x55226D70, 0x59EB6FF0, 0x5A1D7171,
};

/// Arithmetic-coded segment reader (T.81 Annex D decoder, section D.2).
class ArithReader {
public:
    ArithReader(const std::uint8_t* p, std::size_t n, std::size_t pos) : p_(p), n_(n), pos_(pos) {}
    void reset() {
        c_ = 0;
        a_ = 0;
        ct_ = -16;  // forces reading two initial bytes
    }
    /// Decodes one binary decision using (and updating) the statistics bin *st.
    int decode(std::uint8_t* st) {
        while (a_ < 0x8000) {  // renormalization and data input (D.2.6)
            if (--ct_ < 0) {
                c_ = (c_ << 8) | nextByte();
                if ((ct_ += 8) < 0)
                    if (++ct_ == 0) a_ = 0x8000;  // got 2 initial bytes
            }
            a_ <<= 1;
        }
        const int sv = *st;
        std::uint32_t qe = kQeTable[sv & 0x7F];
        const int nl = static_cast<int>(qe & 0xFF);
        qe >>= 8;
        const int nm = static_cast<int>(qe & 0xFF);
        qe >>= 8;
        std::int64_t temp = a_ - static_cast<std::int64_t>(qe);
        a_ = temp;
        temp <<= ct_;
        int bit = sv >> 7;
        if (c_ >= temp) {
            c_ -= temp;
            if (a_ < static_cast<std::int64_t>(qe)) {  // conditional exchange
                a_ = qe;
                *st = static_cast<std::uint8_t>((sv & 0x80) ^ nm);
            } else {
                a_ = qe;
                *st = static_cast<std::uint8_t>((sv & 0x80) ^ nl);
                bit ^= 1;
            }
        } else if (a_ < 0x8000) {
            if (a_ < static_cast<std::int64_t>(qe)) {
                *st = static_cast<std::uint8_t>((sv & 0x80) ^ nl);
                bit ^= 1;
            } else {
                *st = static_cast<std::uint8_t>((sv & 0x80) ^ nm);
            }
        }
        return bit;
    }
    /// Consumes the next RSTn marker (if present) and resets the decoder registers.
    void restart() {
        if (!marker_) {
            while (pos_ + 1 < n_ && !(p_[pos_] == 0xFF && p_[pos_ + 1] != 0 && p_[pos_ + 1] != 0xFF)) ++pos_;
        }
        if (pos_ + 1 < n_ && p_[pos_] == 0xFF && p_[pos_ + 1] >= 0xD0 && p_[pos_ + 1] <= 0xD7) pos_ += 2;
        marker_ = false;
        reset();
    }
    std::size_t endPos() const {
        std::size_t p = pos_;
        while (p + 1 < n_ && !(p_[p] == 0xFF && p_[p + 1] != 0 && !(p_[p + 1] >= 0xD0 && p_[p + 1] <= 0xD7))) ++p;
        return p;
    }

private:
    std::int64_t nextByte() {
        if (marker_ || pos_ >= n_) {
            marker_ = true;
            return 0;  // a marker ends the segment: feed zeros (legal in arithmetic coding)
        }
        const std::uint8_t b = p_[pos_++];
        if (b != 0xFF) return b;
        std::size_t j = pos_;
        while (j < n_ && p_[j] == 0xFF) ++j;  // swallow fill bytes
        if (j < n_ && p_[j] == 0) {
            pos_ = j + 1;
            return 0xFF;  // stuffed zero
        }
        marker_ = true;
        pos_ = j - 1;  // keep pointing at the 0xFF of the marker
        return 0;
    }
    const std::uint8_t* p_;
    std::size_t n_, pos_;
    std::int64_t c_ = 0, a_ = 0;
    int ct_ = -16;
    bool marker_ = false;
};

struct Component {
    int id = 0, h = 1, v = 1, tq = 0;
    int td = 0, ta = 0;  // entropy table selectors of the current scan
    int bw = 0, bh = 0;  // blocks (DCT) or samples (lossless) per line / column, padded to whole MCUs
    int pred = 0;        // DC predictor (Huffman) / last DC value (arithmetic)
    int dcContext = 0;   // arithmetic DC conditioning category
    bool resetPending = true;  // lossless: next sample starts a restart interval
    int rowStart = 0;          // lossless: first row of the current restart interval
    std::vector<std::int32_t> coef;     // DCT coefficients, 64 per block
    std::vector<std::uint16_t> samples;  // lossless samples
    std::int32_t* block(int bx, int by) { return coef.data() + (static_cast<std::size_t>(by) * bw + bx) * 64; }
};

class Decoder {
public:
    /// Decodes to 16-bit storage holding samples in [0, 2^precision - 1].
    ImageU16 decode(const std::vector<std::uint8_t>& bytes, int& precision, ImageMetadata* meta, bool keepCmyk = false) {
        keepCmyk_ = keepCmyk;
        ByteReader rd(bytes.data(), bytes.size(), "JPEG");
        if (rd.u8() != 0xFF || rd.u8() != 0xD8) fail("JPEG: missing SOI marker");
        bool frame = false, anyScan = false;
        std::vector<std::vector<std::uint8_t>> icc;
        ImageMetadata info;
        for (;;) {
            if (rd.remaining() < 2) break;  // tolerate missing EOI
            if (rd.u8() != 0xFF) fail("JPEG: expected marker");
            int m = rd.u8();
            while (m == 0xFF) m = rd.u8();
            if (m == 0xD9) break;  // EOI
            if (m == 0xD8 || (m >= 0xD0 && m <= 0xD7) || m == 0x01) continue;  // standalone markers
            const std::size_t segStart = rd.pos();
            const std::size_t len = rd.u16be();
            if (len < 2) fail("JPEG: invalid segment length");
            const std::uint8_t* body = rd.take(len - 2);
            ByteReader seg(body, len - 2, "JPEG segment");
            switch (m) {
                case 0xC0: case 0xC1: case 0xC2: case 0xC3: case 0xC9: case 0xCA:
                    if (frame) fail("JPEG: multiple frames are not supported");
                    readSOF(seg, m);
                    frame = true;
                    break;
                case 0xCB:
                    fail("JPEG: lossless arithmetic-coded JPEG (SOF11) is not supported");
                case 0xC5: case 0xC6: case 0xC7: case 0xCD: case 0xCE: case 0xCF:
                    fail("JPEG: hierarchical (differential) JPEG is not supported");
                case 0xC4: readDHT(seg); break;
                case 0xCC: readDAC(seg); break;
                case 0xDB: readDQT(seg); break;
                case 0xDD: restartInterval_ = seg.u16be(); break;
                case 0xDA: {
                    if (!frame) fail("JPEG: scan before frame header");
                    readSOS(seg);
                    std::size_t end;
                    if (arith_) {
                        ArithReader ar(bytes.data(), bytes.size(), segStart + len);
                        decodeScanArith(ar);
                        end = ar.endPos();
                    } else {
                        BitReader br(bytes.data(), bytes.size(), segStart + len);
                        if (lossless_) decodeScanLossless(br);
                        else decodeScanHuffman(br);
                        end = br.endPos();
                    }
                    rd.seek(end);
                    anyScan = true;
                    break;
                }
                case 0xE0:  // JFIF: pixel density
                    if (len >= 16 && std::memcmp(body, "JFIF", 5) == 0 && (body[7] == 1 || body[7] == 2)) {
                        const double f = body[7] == 1 ? 1.0 : 2.54;
                        info.dpiX = ((body[8] << 8) | body[9]) * f;
                        info.dpiY = ((body[10] << 8) | body[11]) * f;
                    }
                    break;
                case 0xE1:  // EXIF
                    if (len > 8 && std::memcmp(body, "Exif\0\0", 6) == 0 && info.exif.empty()) {
                        info.exif.assign(body + 6, body + len - 2);
                        info.orientation = exifOrientation(info.exif);
                    }
                    break;
                case 0xE2:  // ICC profile, possibly split over several APP2 segments
                    if (len > 16 && std::memcmp(body, "ICC_PROFILE\0", 12) == 0) {
                        const int seq = body[12], count = body[13];
                        if (seq >= 1 && seq <= count) {
                            if (icc.size() < static_cast<std::size_t>(count)) icc.resize(static_cast<std::size_t>(count));
                            icc[static_cast<std::size_t>(seq - 1)].assign(body + 14, body + len - 2);
                        }
                    }
                    break;
                case 0xEE:
                    if (len >= 14 && std::memcmp(body, "Adobe", 5) == 0) adobeTransform_ = body[11];
                    break;
                case 0xFE:
                    info.text.push_back({"comment", std::string(reinterpret_cast<const char*>(body), len - 2)});
                    break;
                default: break;  // other APPn segments are skipped
            }
        }
        if (!frame || !anyScan) fail("JPEG: no image data");
        precision = P_;
        if (meta) {
            for (const auto& chunk : icc) info.iccProfile.insert(info.iccProfile.end(), chunk.begin(), chunk.end());
            info.bitDepth = P_;
            *meta = std::move(info);
        }
        return output();
    }

private:
    void readSOF(ByteReader& s, int marker) {
        progressive_ = marker == 0xC2 || marker == 0xCA;
        lossless_ = marker == 0xC3;
        arith_ = marker == 0xC9 || marker == 0xCA;
        P_ = s.u8();
        if (lossless_ ? (P_ < 2 || P_ > 16) : (P_ != 8 && P_ != 12)) fail("JPEG: unsupported sample precision");
        if (marker == 0xC0 && P_ != 8) fail("JPEG: baseline JPEG must use 8-bit samples");
        H_ = s.u16be();
        W_ = s.u16be();
        if (H_ == 0) fail("JPEG: DNL-defined image height is not supported");
        checkDecodedSize(W_, H_, "JPEG");
        const int n = s.u8();
        if (n < 1 || n > 4) fail("JPEG: unsupported number of components");
        comps_.resize(static_cast<std::size_t>(n));
        hmax_ = vmax_ = 1;
        for (auto& c : comps_) {
            c.id = s.u8();
            const int hv = s.u8();
            c.h = hv >> 4;
            c.v = hv & 15;
            c.tq = s.u8();
            if (c.h < 1 || c.h > 4 || c.v < 1 || c.v > 4 || c.tq > 3) fail("JPEG: invalid component parameters");
            hmax_ = std::max(hmax_, c.h);
            vmax_ = std::max(vmax_, c.v);
        }
        const int unit = lossless_ ? 1 : 8;
        mcusX_ = (W_ + unit * hmax_ - 1) / (unit * hmax_);
        mcusY_ = (H_ + unit * vmax_ - 1) / (unit * vmax_);
        for (auto& c : comps_) {
            c.bw = mcusX_ * c.h;
            c.bh = mcusY_ * c.v;
            if (lossless_) c.samples.assign(static_cast<std::size_t>(c.bw) * c.bh, 0);
            else c.coef.assign(static_cast<std::size_t>(c.bw) * c.bh * 64, 0);
        }
    }

    void readDHT(ByteReader& s) {
        while (!s.eof()) {
            const int tcth = s.u8();
            const int tc = tcth >> 4, th = tcth & 15;
            if (tc > 1 || th > 3) fail("JPEG: invalid Huffman table id");
            const std::uint8_t* bits = s.take(16);
            int total = 0;
            for (int i = 0; i < 16; ++i) total += bits[i];
            if (total > 256) fail("JPEG: invalid Huffman table");
            const std::uint8_t* vals = s.take(static_cast<std::size_t>(total));
            (tc == 0 ? dc_ : ac_)[th].build(bits, vals, total);
        }
    }

    void readDAC(ByteReader& s) {  // arithmetic conditioning (T.81 B.2.4.3)
        while (!s.eof()) {
            const int t = s.u8(), v = s.u8();
            const int tc = t >> 4, tb = t & 15;
            if (tb > 3 || tc > 1) fail("JPEG: invalid DAC table id");
            if (tc == 0) {
                dcL_[tb] = v & 15;
                dcU_[tb] = v >> 4;
                if (dcL_[tb] > dcU_[tb]) fail("JPEG: invalid DAC conditioning values");
            } else {
                if (v < 1 || v > 63) fail("JPEG: invalid DAC conditioning value");
                acK_[tb] = v;
            }
        }
    }

    void readDQT(ByteReader& s) {
        while (!s.eof()) {
            const int pqtq = s.u8();
            const int pq = pqtq >> 4, tq = pqtq & 15;
            if (tq > 3 || pq > 1) fail("JPEG: invalid quantization table");
            for (int k = 0; k < 64; ++k) qt_[tq][kZigzag[k]] = pq ? s.u16be() : s.u8();
        }
    }

    void readSOS(ByteReader& s) {
        const int n = s.u8();
        if (n < 1 || n > 4) fail("JPEG: invalid scan component count");
        scan_.clear();
        for (int i = 0; i < n; ++i) {
            const int id = s.u8(), t = s.u8();
            Component* c = nullptr;
            for (auto& cc : comps_)
                if (cc.id == id) c = &cc;
            if (!c) fail("JPEG: scan references unknown component");
            c->td = t >> 4;
            c->ta = t & 15;
            if (c->td > 3 || c->ta > 3) fail("JPEG: invalid entropy table selector");
            scan_.push_back(c);
        }
        ss_ = s.u8();
        se_ = s.u8();
        const int a = s.u8();
        ah_ = a >> 4;
        al_ = a & 15;
        if (lossless_) {
            if (ss_ < 1 || ss_ > 7 || al_ >= P_) fail("JPEG: invalid lossless predictor or point transform");
        } else if (!progressive_) {
            ss_ = 0;
            se_ = 63;
        } else if (ss_ > se_ || se_ > 63 || (ss_ == 0 && se_ != 0) || (ss_ > 0 && n != 1) || al_ > 13) {
            fail("JPEG: invalid progressive scan parameters");
        }
    }

    /// Calls unitFn(component, x, y) for every block (DCT) or sample (lossless) of the scan in
    /// stream order, and restartFn() at every restart interval boundary.
    template <class UnitFn, class RestartFn>
    void forEachUnit(UnitFn unitFn, RestartFn restartFn) {
        int mcuCount = 0;
        auto maybeRestart = [&]() {
            if (restartInterval_ && mcuCount > 0 && mcuCount % restartInterval_ == 0) restartFn();
            ++mcuCount;
        };
        if (scan_.size() == 1) {
            Component& c = *scan_[0];
            const int unit = lossless_ ? 1 : 8;
            const int cw = (W_ * c.h + hmax_ - 1) / hmax_, ch = (H_ * c.v + vmax_ - 1) / vmax_;
            const int nx = (cw + unit - 1) / unit, ny = (ch + unit - 1) / unit;
            for (int by = 0; by < ny; ++by)
                for (int bx = 0; bx < nx; ++bx) {
                    maybeRestart();
                    unitFn(c, bx, by);
                }
        } else {
            for (int my = 0; my < mcusY_; ++my)
                for (int mx = 0; mx < mcusX_; ++mx) {
                    maybeRestart();
                    for (auto* c : scan_)
                        for (int v = 0; v < c->v; ++v)
                            for (int h = 0; h < c->h; ++h) unitFn(*c, mx * c->h + h, my * c->v + v);
                }
        }
    }

    // ---- Huffman-coded DCT scans -------------------------------------------------
    void decodeBlockHuffman(BitReader& br, Component& c, std::int32_t* blk) {
        if (!progressive_) {
            const int t = br.decode(dc_[c.td]);
            c.pred += br.receiveExtend(t);
            blk[0] = c.pred;
            const HuffTable& ac = ac_[c.ta];
            for (int k = 1; k < 64;) {
                const int rs = br.decode(ac);
                const int r = rs >> 4, sz = rs & 15;
                if (sz) {
                    k += r;
                    if (k > 63) break;
                    blk[kZigzag[k]] = br.receiveExtend(sz);
                    ++k;
                } else {
                    if (r != 15) break;
                    k += 16;
                }
            }
            return;
        }
        if (ss_ == 0) {  // DC scan
            if (ah_ == 0) {
                const int t = br.decode(dc_[c.td]);
                c.pred += br.receiveExtend(t);
                blk[0] = c.pred * (1 << al_);
            } else if (br.bit()) {
                blk[0] |= (1 << al_);
            }
            return;
        }
        const HuffTable& ac = ac_[c.ta];
        if (ah_ == 0) {  // AC first pass
            if (eobrun_ > 0) {
                --eobrun_;
                return;
            }
            for (int k = ss_; k <= se_; ++k) {
                const int rs = br.decode(ac);
                const int r = rs >> 4, sz = rs & 15;
                if (sz == 0) {
                    if (r < 15) {
                        eobrun_ = (1 << r) - 1;
                        if (r) eobrun_ += br.bits(r);
                        break;
                    }
                    k += 15;
                } else {
                    k += r;
                    if (k > 63) break;
                    blk[kZigzag[k]] = br.receiveExtend(sz) * (1 << al_);
                }
            }
            return;
        }
        // AC refinement pass (follows libjpeg's decode_mcu_AC_refine)
        const int p1 = 1 << al_, m1 = -1 * (1 << al_);
        auto refine = [&](std::int32_t& coef) {
            if (br.bit() && (coef & p1) == 0) coef += coef >= 0 ? p1 : m1;
        };
        int k = ss_;
        if (eobrun_ == 0) {
            for (; k <= se_; ++k) {
                const int rs = br.decode(ac);
                int r = rs >> 4;
                int sval = 0;
                if (rs & 15) {
                    sval = br.bit() ? p1 : m1;
                } else if (r != 15) {
                    eobrun_ = 1 << r;
                    if (r) eobrun_ += br.bits(r);
                    break;
                }
                while (k <= se_) {
                    std::int32_t& coef = blk[kZigzag[k]];
                    if (coef != 0) {
                        refine(coef);
                    } else {
                        if (--r < 0) break;
                    }
                    ++k;
                }
                if (sval && k <= 63) blk[kZigzag[k]] = sval;
            }
        }
        if (eobrun_ > 0) {
            for (; k <= se_; ++k) {
                std::int32_t& coef = blk[kZigzag[k]];
                if (coef != 0) refine(coef);
            }
            --eobrun_;
        }
    }

    void decodeScanHuffman(BitReader& br) {
        for (auto* c : scan_) c->pred = 0;
        eobrun_ = 0;
        forEachUnit([&](Component& c, int bx, int by) { decodeBlockHuffman(br, c, c.block(bx, by)); },
                    [&]() {
                        br.restart();
                        for (auto* c : scan_) c->pred = 0;
                        eobrun_ = 0;
                    });
    }

    // ---- lossless scans (T.81 Annex H) ------------------------------------------
    void decodeScanLossless(BitReader& br) {
        const int pt = al_, predictor = ss_;
        const int initial = 1 << (P_ - pt - 1);
        for (auto* c : scan_) {
            c->resetPending = true;
            c->rowStart = 0;
        }
        auto halfFloor = [](int v) { return v >= 0 ? v >> 1 : -((-v + 1) >> 1); };
        forEachUnit(
            [&](Component& c, int x, int y) {
                const int s = br.decode(dc_[c.td]);
                int diff = 0;
                if (s == 16) diff = 32768;
                else if (s) diff = br.receiveExtend(s);
                std::uint16_t* plane = c.samples.data();
                const std::size_t W = static_cast<std::size_t>(c.bw);
                auto S = [&](int xx, int yy) { return static_cast<int>(plane[static_cast<std::size_t>(yy) * W + xx]); };
                int pred;
                if (c.resetPending) {
                    pred = initial;
                    c.resetPending = false;
                    c.rowStart = y;
                } else if (y == c.rowStart) {
                    pred = S(x - 1, y);
                } else if (x == 0) {
                    pred = S(x, y - 1);
                } else {
                    const int ra = S(x - 1, y), rb = S(x, y - 1), rc = S(x - 1, y - 1);
                    switch (predictor) {
                        case 1: pred = ra; break;
                        case 2: pred = rb; break;
                        case 3: pred = rc; break;
                        case 4: pred = ra + rb - rc; break;
                        case 5: pred = ra + halfFloor(rb - rc); break;
                        case 6: pred = rb + halfFloor(ra - rc); break;
                        default: pred = halfFloor(ra + rb); break;
                    }
                }
                plane[static_cast<std::size_t>(y) * W + x] = static_cast<std::uint16_t>((pred + diff) & 0xFFFF);
            },
            [&]() {
                br.restart();
                for (auto* c : scan_) c->resetPending = true;
            });
        if (pt)
            for (auto* c : scan_)
                for (auto& v : c->samples) v = static_cast<std::uint16_t>(v << pt);
    }

    // ---- arithmetic-coded DCT scans (T.81 Annexes D, F, G) ------------------------
    void resetArithStats() {
        for (auto* c : scan_) {
            if (!progressive_ || (ss_ == 0 && ah_ == 0)) {
                std::memset(dcStats_[c->td], 0, sizeof(dcStats_[0]));
                c->pred = 0;
                c->dcContext = 0;
            }
            if (!progressive_ || ss_) std::memset(acStats_[c->ta], 0, sizeof(acStats_[0]));
        }
    }

    /// Decodes one DC difference (Figures F.19 - F.24) with context conditioning.
    int arithDcDiff(ArithReader& ar, Component& c) {
        std::uint8_t* st = dcStats_[c.td] + c.dcContext;
        if (ar.decode(st) == 0) {
            c.dcContext = 0;
            return 0;
        }
        const int sign = ar.decode(st + 1);
        st += 2 + sign;
        int m = ar.decode(st);
        if (m) {
            st = dcStats_[c.td] + 20;
            while (ar.decode(st)) {
                if ((m <<= 1) == 0x8000) fail("JPEG: corrupt arithmetic-coded data");
                ++st;
            }
        }
        if (m < ((1 << dcL_[c.td]) >> 1)) c.dcContext = 0;
        else if (m > ((1 << dcU_[c.td]) >> 1)) c.dcContext = 12 + sign * 4;
        else c.dcContext = 4 + sign * 4;
        int v = m;
        st += 14;
        while (m >>= 1)
            if (ar.decode(st)) v |= m;
        v += 1;
        return sign ? -v : v;
    }

    /// Decodes AC coefficients ss..se of one block (Figure F.20), scaled by 2^al.
    void arithAc(ArithReader& ar, Component& c, std::int32_t* blk, int ss, int se, int al) {
        std::uint8_t* stats = acStats_[c.ta];
        for (int k = ss; k <= se; ++k) {
            std::uint8_t* st = stats + 3 * (k - 1);
            if (ar.decode(st)) break;  // end of block
            while (ar.decode(st + 1) == 0) {
                st += 3;
                if (++k > se) fail("JPEG: corrupt arithmetic-coded data");
            }
            const int sign = ar.decode(&fixedBin_);
            st += 2;
            int m = ar.decode(st);
            if (m && ar.decode(st)) {
                m <<= 1;
                st = stats + (k <= acK_[c.ta] ? 189 : 217);
                while (ar.decode(st)) {
                    if ((m <<= 1) == 0x8000) fail("JPEG: corrupt arithmetic-coded data");
                    ++st;
                }
            }
            int v = m;
            st += 14;
            while (m >>= 1)
                if (ar.decode(st)) v |= m;
            v += 1;
            if (sign) v = -v;
            blk[kZigzag[k]] = v * (1 << al);
        }
    }

    void decodeBlockArith(ArithReader& ar, Component& c, std::int32_t* blk) {
        if (!progressive_) {
            c.pred = (c.pred + arithDcDiff(ar, c)) & 0xFFFF;
            blk[0] = static_cast<std::int16_t>(static_cast<std::uint16_t>(c.pred));
            arithAc(ar, c, blk, 1, 63, 0);
            return;
        }
        if (ss_ == 0) {
            if (ah_ == 0) {
                c.pred = (c.pred + arithDcDiff(ar, c)) & 0xFFFF;
                blk[0] = static_cast<std::int16_t>(static_cast<std::uint16_t>(c.pred)) * (1 << al_);
            } else if (ar.decode(&fixedBin_)) {
                blk[0] |= 1 << al_;
            }
            return;
        }
        if (ah_ == 0) {
            arithAc(ar, c, blk, ss_, se_, al_);
            return;
        }
        // AC refinement (Figure G.10 / libjpeg decode_mcu_AC_refine)
        std::uint8_t* stats = acStats_[c.ta];
        const int p1 = 1 << al_, m1 = -1 * (1 << al_);
        int kex = se_;
        for (; kex > 0; --kex)
            if (blk[kZigzag[kex]]) break;
        for (int k = ss_; k <= se_; ++k) {
            std::uint8_t* st = stats + 3 * (k - 1);
            if (k > kex && ar.decode(st)) break;  // end of block
            for (;;) {
                std::int32_t& coef = blk[kZigzag[k]];
                if (coef) {
                    if (ar.decode(st + 2)) coef += coef < 0 ? m1 : p1;
                    break;
                }
                if (ar.decode(st + 1)) {
                    coef = ar.decode(&fixedBin_) ? m1 : p1;
                    break;
                }
                st += 3;
                if (++k > se_) fail("JPEG: corrupt arithmetic-coded data");
            }
        }
    }

    void decodeScanArith(ArithReader& ar) {
        ar.reset();
        fixedBin_ = 113;
        resetArithStats();
        forEachUnit([&](Component& c, int bx, int by) { decodeBlockArith(ar, c, c.block(bx, by)); },
                    [&]() {
                        ar.restart();
                        resetArithStats();
                    });
    }

    // ---- output: IDCT, upsampling, color conversion -----------------------------
    /// Upsamples a component plane to full resolution (libjpeg "fancy" triangle filter for 2x
    /// factors, pixel replication otherwise). Rows are processed in parallel.
    static std::vector<std::uint16_t> upsample(const std::vector<std::uint16_t>& in, int stride, int cw, int ch, int sx,
                                               int sy, int W, int H) {
        std::vector<std::uint16_t> out(static_cast<std::size_t>(W) * H);
        auto at = [&](int x, int y) -> int {
            return in[static_cast<std::size_t>(clamp(y, 0, ch - 1)) * stride + clamp(x, 0, cw - 1)];
        };
        parallelFor(0, H, [&](int y0, int y1) {
            std::vector<int> colsum(static_cast<std::size_t>(cw));
            for (int y = y0; y < y1; ++y) {
                std::uint16_t* o = &out[static_cast<std::size_t>(y) * W];
                if (sx == 1 && sy == 1) {
                    for (int x = 0; x < W; ++x) o[x] = static_cast<std::uint16_t>(at(x, y));
                } else if (sx == 2 && sy == 1) {
                    for (int x = 0; x < W; ++x) {
                        const int i = x >> 1;
                        o[x] = static_cast<std::uint16_t>((x & 1) ? (3 * at(i, y) + at(i + 1, y) + 2) >> 2
                                                                  : (3 * at(i, y) + at(i - 1, y) + 1) >> 2);
                    }
                } else if (sx == 1 && sy == 2) {
                    const int j = y >> 1;
                    for (int x = 0; x < W; ++x)
                        o[x] = static_cast<std::uint16_t>((y & 1) ? (3 * at(x, j) + at(x, j + 1) + 2) >> 2
                                                                  : (3 * at(x, j) + at(x, j - 1) + 1) >> 2);
                } else if (sx == 2 && sy == 2) {
                    const int j = y >> 1, jn = (y & 1) ? j + 1 : j - 1;
                    for (int i = 0; i < cw; ++i) colsum[i] = 3 * at(i, j) + at(i, jn);
                    for (int x = 0; x < W; ++x) {
                        const int i = x >> 1;
                        o[x] = static_cast<std::uint16_t>((x & 1) ? (3 * colsum[i] + colsum[std::min(i + 1, cw - 1)] + 7) >> 4
                                                                  : (3 * colsum[i] + colsum[std::max(i - 1, 0)] + 8) >> 4);
                    }
                } else {
                    for (int x = 0; x < W; ++x) o[x] = static_cast<std::uint16_t>(at(x / sx, y / sy));
                }
            }
        }, static_cast<double>(W) * 4);
        return out;
    }

    ImageU16 output() {
        const int maxVal = (1 << P_) - 1;
        const float center = static_cast<float>(1 << (P_ - 1));
        std::vector<std::vector<std::uint16_t>> planes;
        for (auto& c : comps_) {
            if (hmax_ % c.h || vmax_ % c.v) fail("JPEG: unsupported sampling factors");
            const int cw = (W_ * c.h + hmax_ - 1) / hmax_, ch = (H_ * c.v + vmax_ - 1) / vmax_;
            if (lossless_) {
                planes.push_back(upsample(c.samples, c.bw, cw, ch, hmax_ / c.h, vmax_ / c.v, W_, H_));
                continue;
            }
            const int stride = c.bw * 8;
            std::vector<std::uint16_t> plane(static_cast<std::size_t>(stride) * c.bh * 8);
            const int* q = qt_[c.tq];
            parallelFor(0, c.bh, [&](int by0, int by1) {
                float deq[64];
                for (int by = by0; by < by1; ++by)
                    for (int bx = 0; bx < c.bw; ++bx) {
                        const std::int32_t* b = c.block(bx, by);
                        for (int i = 0; i < 64; ++i) deq[i] = static_cast<float>(b[i]) * static_cast<float>(q[i]);
                        idct8x8(deq, &plane[(static_cast<std::size_t>(by) * 8 * stride) + static_cast<std::size_t>(bx) * 8],
                                static_cast<std::size_t>(stride), center, static_cast<float>(maxVal));
                    }
            }, static_cast<double>(c.bw) * 64 * 20);
            planes.push_back(upsample(plane, stride, cw, ch, hmax_ / c.h, vmax_ / c.v, W_, H_));
        }
        if (comps_.size() == 1) return ImageU16(W_, H_, 1, std::move(planes[0]));
        if (comps_.size() == 4 && keepCmyk_) return cmykInk(planes);
        ImageU16 img(W_, H_, 3);
        // 3 components: YCbCr unless Adobe says "no transform" or ids spell 'R','G','B'.
        const bool rgbIds = comps_.size() == 3 && comps_[0].id == 'R' && comps_[1].id == 'G' && comps_[2].id == 'B';
        const bool ycc = comps_.size() == 3 ? !(adobeTransform_ == 0 || rgbIds) : adobeTransform_ == 2;
        const bool cmyk = comps_.size() == 4;
        const float fmax = static_cast<float>(maxVal);
        parallelForN(img.pixelCount(), [&](std::size_t i0, std::size_t i1) {
            std::uint16_t* d = img.data();
            for (std::size_t i = i0; i < i1; ++i) {
                int r = planes[0][i], g = planes[1][i], b = planes[2][i];
                if (ycc) {
                    const float y = static_cast<float>(r), cb = static_cast<float>(g) - center, cr = static_cast<float>(b) - center;
                    r = static_cast<int>(clamp(y + 1.402f * cr, 0.f, fmax) + 0.5f);
                    g = static_cast<int>(clamp(y - 0.344136f * cb - 0.714136f * cr, 0.f, fmax) + 0.5f);
                    b = static_cast<int>(clamp(y + 1.772f * cb, 0.f, fmax) + 0.5f);
                }
                if (cmyk) {  // Adobe CMYK/YCCK are stored inverted
                    const int k = planes[3][i];
                    if (adobeTransform_ == 2) {
                        r = maxVal - r;
                        g = maxVal - g;
                        b = maxVal - b;
                    }
                    r = (r * k + maxVal / 2) / maxVal;
                    g = (g * k + maxVal / 2) / maxVal;
                    b = (b * k + maxVal / 2) / maxVal;
                }
                d[i * 3 + 0] = static_cast<std::uint16_t>(r);
                d[i * 3 + 1] = static_cast<std::uint16_t>(g);
                d[i * 3 + 2] = static_cast<std::uint16_t>(b);
            }
        }, 10);
        return img;
    }

    /// 4-component output as ink amounts (0 = no ink), undoing Adobe's inverted storage and
    /// the YCCK transform, ready for an ICC CMYK profile.
    ImageU16 cmykInk(const std::vector<std::vector<std::uint16_t>>& planes) const {
        const int maxVal = (1 << P_) - 1;
        const float center = static_cast<float>(1 << (P_ - 1)), fmax = static_cast<float>(maxVal);
        const bool inverted = adobeTransform_ >= 0;
        ImageU16 img(W_, H_, 4);
        parallelForN(img.pixelCount(), [&](std::size_t i0, std::size_t i1) {
            for (std::size_t i = i0; i < i1; ++i) {
                int v[4] = {planes[0][i], planes[1][i], planes[2][i], planes[3][i]};
                if (adobeTransform_ == 2) {  // YCCK: YCbCr encodes (max - C, max - M, max - Y)
                    const float y = static_cast<float>(v[0]), cb = v[1] - center, cr = v[2] - center;
                    v[0] = maxVal - static_cast<int>(clamp(y + 1.402f * cr, 0.f, fmax) + 0.5f);
                    v[1] = maxVal - static_cast<int>(clamp(y - 0.344136f * cb - 0.714136f * cr, 0.f, fmax) + 0.5f);
                    v[2] = maxVal - static_cast<int>(clamp(y + 1.772f * cb, 0.f, fmax) + 0.5f);
                }
                for (int k = 0; k < 4; ++k) img.data()[i * 4 + k] = static_cast<std::uint16_t>(inverted ? maxVal - v[k] : v[k]);
            }
        }, 8);
        return img;
    }

    HuffTable dc_[4], ac_[4];
    bool keepCmyk_ = false;
    int qt_[4][64] = {};
    std::vector<Component> comps_;
    std::vector<Component*> scan_;
    bool progressive_ = false, lossless_ = false, arith_ = false;
    int P_ = 8;
    int W_ = 0, H_ = 0, hmax_ = 1, vmax_ = 1, mcusX_ = 0, mcusY_ = 0;
    int restartInterval_ = 0, adobeTransform_ = -1;
    int ss_ = 0, se_ = 63, ah_ = 0, al_ = 0, eobrun_ = 0;
    // arithmetic coding state
    std::uint8_t dcStats_[4][64] = {};
    std::uint8_t acStats_[4][256] = {};
    std::uint8_t fixedBin_ = 113;
    int dcL_[4] = {0, 0, 0, 0}, dcU_[4] = {1, 1, 1, 1}, acK_[4] = {5, 5, 5, 5};
};

class BitWriter {
public:
    explicit BitWriter(std::vector<std::uint8_t>& o) : o_(o) {}
    void put(std::uint32_t code, int len) {
        buf_ = (buf_ << len) | (code & ((1u << len) - 1u));
        cnt_ += len;
        while (cnt_ >= 8) {
            const std::uint8_t b = static_cast<std::uint8_t>(buf_ >> (cnt_ - 8));
            o_.push_back(b);
            if (b == 0xFF) o_.push_back(0);  // byte stuffing
            cnt_ -= 8;
        }
        buf_ &= (1u << cnt_) - 1u;
    }
    void flush() {
        if (cnt_) put((1u << (8 - cnt_)) - 1u, 8 - cnt_);  // pad with 1-bits
    }

private:
    std::vector<std::uint8_t>& o_;
    std::uint32_t buf_ = 0;
    int cnt_ = 0;
};

struct HuffCodes {
    std::uint16_t code[256] = {0};
    std::uint8_t size[256] = {0};
    HuffCodes(const std::uint8_t* bits, const std::uint8_t* vals) {
        int k = 0, c = 0;
        for (int l = 1; l <= 16; ++l, c <<= 1)
            for (int i = 0; i < bits[l - 1]; ++i, ++k, ++c) {
                code[vals[k]] = static_cast<std::uint16_t>(c);
                size[vals[k]] = static_cast<std::uint8_t>(l);
            }
    }
};

inline int bitLength(int v) {
    int n = 0;
    for (v = std::abs(v); v; v >>= 1) ++n;
    return n;
}

}  // namespace jpg
}  // namespace detail

/// Decodes a JPEG to 16-bit storage with samples in [0, 2^precision - 1] (precision is 8 or
/// 12 for DCT JPEGs, 2-16 for lossless JPEGs). Optionally returns the file's metadata.
/// With cmyk = true, CMYK/YCCK JPEGs are returned as 4-channel ink amounts (0 = no ink) instead
/// of being converted to RGB (use convertColorProfile with the embedded ICC profile).
inline ImageU16 decodeJPEGRaw(const std::vector<std::uint8_t>& bytes, int& precision, ImageMetadata* meta = nullptr,
                              bool cmyk = false) {
    detail::jpg::Decoder dec;
    return dec.decode(bytes, precision, meta, cmyk);
}

/// Decodes a JPEG file to 8 bits per sample (12/16-bit JPEGs are scaled down).
/// Output: 1 channel for grayscale JPEGs, otherwise 3 (RGB).
inline ImageU8 decodeJPEG(const std::vector<std::uint8_t>& bytes, ImageMetadata* meta = nullptr) {
    int precision = 8;
    const ImageU16 raw = decodeJPEGRaw(bytes, precision, meta);
    return raw.convertTo<std::uint8_t>(precision == 8 ? 1.0 : 255.0 / ((1 << precision) - 1));
}

/// Decodes a JPEG file to the full 16-bit range (keeps the precision of 12-bit and lossless JPEGs).
inline ImageU16 decodeJPEG16(const std::vector<std::uint8_t>& bytes, ImageMetadata* meta = nullptr) {
    int precision = 8;
    const ImageU16 raw = decodeJPEGRaw(bytes, precision, meta);
    return raw.convertTo<std::uint16_t>(65535.0 / ((1 << precision) - 1));
}

/// Options for encodeJPEG.
struct JpegOptions {
    int quality = 90;                         ///< 1-100
    bool subsample = true;                    ///< 4:2:0 chroma (true) or 4:4:4 (false)
    const ImageMetadata* metadata = nullptr;  ///< ICC profile, EXIF, DPI and "comment" text to embed
};

/// Encodes an image as baseline JPEG. 1-2 channel images are written as grayscale, 3-4
/// channel images as YCbCr (alpha is discarded).
inline std::vector<std::uint8_t> encodeJPEG(const ImageU8& src, const JpegOptions& options) {
    using namespace detail::jpg;
    int quality = options.quality;
    const bool subsample = options.subsample;
    detail::require(!src.empty(), "encodeJPEG: empty image");
    detail::require(src.width() <= 65535 && src.height() <= 65535, "encodeJPEG: image too large for JPEG");
    const bool gray = src.channels() <= 2;
    quality = clamp(quality, 1, 100);
    const int scale = quality < 50 ? 5000 / quality : 200 - quality * 2;
    int q[2][64];
    for (int i = 0; i < 64; ++i) {
        q[0][i] = clamp((kStdLumQ[i] * scale + 50) / 100, 1, 255);
        q[1][i] = clamp((kStdChromQ[i] * scale + 50) / 100, 1, 255);
    }
    const int W = src.width(), H = src.height();
    // Float planes (Y, Cb, Cr), level shifted by -128.
    const int nc = gray ? 1 : 3;
    std::vector<std::vector<float>> planes(static_cast<std::size_t>(nc), std::vector<float>(static_cast<std::size_t>(W) * H));
    const int sc = src.channels();
    for (std::size_t i = 0; i < src.pixelCount(); ++i) {
        const std::uint8_t* p = src.data() + i * sc;
        if (gray) {
            planes[0][i] = p[0] - 128.f;
        } else {
            const float r = p[0], g = p[1], b = p[2];
            planes[0][i] = 0.299f * r + 0.587f * g + 0.114f * b - 128.f;
            planes[1][i] = -0.168736f * r - 0.331264f * g + 0.5f * b;
            planes[2][i] = 0.5f * r - 0.418688f * g - 0.081312f * b;
        }
    }
    const bool sub = subsample && !gray;
    const int hY = sub ? 2 : 1;

    std::vector<std::uint8_t> o{0xFF, 0xD8};
    auto marker = [&](int m, const std::vector<std::uint8_t>& body) {
        detail::put8(o, 0xFF);
        detail::put8(o, static_cast<unsigned>(m));
        detail::put16be(o, static_cast<unsigned>(body.size() + 2));
        o.insert(o.end(), body.begin(), body.end());
    };
    const ImageMetadata* meta = options.metadata;
    {
        std::vector<std::uint8_t> jfif = {'J', 'F', 'I', 'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0};
        if (meta && meta->dpiX > 0 && meta->dpiY > 0) {
            jfif[7] = 1;  // dots per inch
            const unsigned dx = static_cast<unsigned>(clamp(std::lround(meta->dpiX), 1L, 65535L));
            const unsigned dy = static_cast<unsigned>(clamp(std::lround(meta->dpiY), 1L, 65535L));
            jfif[8] = static_cast<std::uint8_t>(dx >> 8);
            jfif[9] = static_cast<std::uint8_t>(dx & 0xFF);
            jfif[10] = static_cast<std::uint8_t>(dy >> 8);
            jfif[11] = static_cast<std::uint8_t>(dy & 0xFF);
        }
        marker(0xE0, jfif);
    }
    if (meta && !meta->exif.empty() && meta->exif.size() <= 65527) {
        std::vector<std::uint8_t> b = {'E', 'x', 'i', 'f', 0, 0};
        b.insert(b.end(), meta->exif.begin(), meta->exif.end());
        marker(0xE1, b);
    }
    if (meta && !meta->iccProfile.empty()) {  // split into APP2 chunks of at most 65519 bytes
        const std::size_t kChunk = 65519;
        const std::size_t count = (meta->iccProfile.size() + kChunk - 1) / kChunk;
        detail::require(count <= 255, "encodeJPEG: ICC profile too large");
        for (std::size_t i = 0; i < count; ++i) {
            std::vector<std::uint8_t> b = {'I', 'C', 'C', '_', 'P', 'R', 'O', 'F', 'I', 'L', 'E', 0};
            b.push_back(static_cast<std::uint8_t>(i + 1));
            b.push_back(static_cast<std::uint8_t>(count));
            const std::size_t from = i * kChunk, to = std::min(meta->iccProfile.size(), from + kChunk);
            b.insert(b.end(), meta->iccProfile.begin() + static_cast<std::ptrdiff_t>(from),
                     meta->iccProfile.begin() + static_cast<std::ptrdiff_t>(to));
            marker(0xE2, b);
        }
    }
    if (meta)
        for (const auto& kv : meta->text)
            if (kv.first == "comment" && !kv.second.empty() && kv.second.size() <= 65533)
                marker(0xFE, std::vector<std::uint8_t>(kv.second.begin(), kv.second.end()));
    for (int t = 0; t < (gray ? 1 : 2); ++t) {
        std::vector<std::uint8_t> b{static_cast<std::uint8_t>(t)};
        for (int k = 0; k < 64; ++k) b.push_back(static_cast<std::uint8_t>(q[t][kZigzag[k]]));
        marker(0xDB, b);
    }
    {
        std::vector<std::uint8_t> b{8};
        detail::put16be(b, static_cast<unsigned>(H));
        detail::put16be(b, static_cast<unsigned>(W));
        detail::put8(b, static_cast<unsigned>(nc));
        for (int c = 0; c < nc; ++c) {
            detail::put8(b, static_cast<unsigned>(c + 1));
            detail::put8(b, c == 0 ? static_cast<unsigned>((hY << 4) | hY) : 0x11u);
            detail::put8(b, c == 0 ? 0u : 1u);
        }
        marker(0xC0, b);
    }
    auto dht = [&](int cls, int id, const std::uint8_t* bits, const std::uint8_t* vals) {
        std::vector<std::uint8_t> b{static_cast<std::uint8_t>((cls << 4) | id)};
        int total = 0;
        for (int i = 0; i < 16; ++i) {
            b.push_back(bits[i]);
            total += bits[i];
        }
        b.insert(b.end(), vals, vals + total);
        marker(0xC4, b);
    };
    dht(0, 0, kDcLumBits, kDcVals);
    dht(1, 0, kAcLumBits, kAcLumVals);
    if (!gray) {
        dht(0, 1, kDcChromBits, kDcVals);
        dht(1, 1, kAcChromBits, kAcChromVals);
    }
    {
        std::vector<std::uint8_t> b{static_cast<std::uint8_t>(nc)};
        for (int c = 0; c < nc; ++c) {
            detail::put8(b, static_cast<unsigned>(c + 1));
            detail::put8(b, c == 0 ? 0x00u : 0x11u);
        }
        b.push_back(0);
        b.push_back(63);
        b.push_back(0);
        marker(0xDA, b);
    }

    const HuffCodes dcCodes[2] = {HuffCodes(kDcLumBits, kDcVals), HuffCodes(kDcChromBits, kDcVals)};
    const HuffCodes acCodes[2] = {HuffCodes(kAcLumBits, kAcLumVals), HuffCodes(kAcChromBits, kAcChromVals)};
    BitWriter bw(o);
    int pred[3] = {0, 0, 0};
    float blk[64], coef[64];
    auto encodeBlock = [&](int comp) {
        const int t = comp == 0 ? 0 : 1;
        fdct8x8(blk, coef);
        int zz[64];
        for (int k = 0; k < 64; ++k) {
            const int i = kZigzag[k];
            zz[k] = static_cast<int>(std::lround(coef[i] / static_cast<float>(q[t][i])));
        }
        const int diff = zz[0] - pred[comp];
        pred[comp] = zz[0];
        int cat = bitLength(diff);
        bw.put(dcCodes[t].code[cat], dcCodes[t].size[cat]);
        if (cat) bw.put(static_cast<std::uint32_t>(diff < 0 ? diff - 1 : diff), cat);
        int run = 0;
        for (int k = 1; k < 64; ++k) {
            const int v = zz[k];
            if (v == 0) {
                ++run;
                continue;
            }
            while (run > 15) {
                bw.put(acCodes[t].code[0xF0], acCodes[t].size[0xF0]);
                run -= 16;
            }
            cat = bitLength(v);
            const int sym = (run << 4) | cat;
            bw.put(acCodes[t].code[sym], acCodes[t].size[sym]);
            bw.put(static_cast<std::uint32_t>(v < 0 ? v - 1 : v), cat);
            run = 0;
        }
        if (run > 0) bw.put(acCodes[t].code[0], acCodes[t].size[0]);
    };
    auto sampleAt = [&](int c, int x, int y) {
        return planes[c][static_cast<std::size_t>(clamp(y, 0, H - 1)) * W + clamp(x, 0, W - 1)];
    };
    const int mcu = 8 * hY;
    for (int my = 0; my < H; my += mcu)
        for (int mx = 0; mx < W; mx += mcu) {
            for (int by = 0; by < hY; ++by)
                for (int bx = 0; bx < hY; ++bx) {
                    for (int y = 0; y < 8; ++y)
                        for (int x = 0; x < 8; ++x) blk[y * 8 + x] = sampleAt(0, mx + bx * 8 + x, my + by * 8 + y);
                    encodeBlock(0);
                }
            for (int c = 1; c < nc; ++c) {
                for (int y = 0; y < 8; ++y)
                    for (int x = 0; x < 8; ++x) {
                        if (sub) {
                            const int sx = mx + 2 * x, sy = my + 2 * y;
                            blk[y * 8 + x] = 0.25f * (sampleAt(c, sx, sy) + sampleAt(c, sx + 1, sy) +
                                                      sampleAt(c, sx, sy + 1) + sampleAt(c, sx + 1, sy + 1));
                        } else {
                            blk[y * 8 + x] = sampleAt(c, mx + x, my + y);
                        }
                    }
                encodeBlock(c);
            }
        }
    bw.flush();
    detail::put8(o, 0xFF);
    detail::put8(o, 0xD9);
    return o;
}

/// Encodes an image as baseline JPEG. quality: 1-100. With subsample=true the chroma planes
/// are stored at half resolution (4:2:0), otherwise at full resolution (4:4:4).
inline std::vector<std::uint8_t> encodeJPEG(const ImageU8& src, int quality = 90, bool subsample = true) {
    JpegOptions o;
    o.quality = quality;
    o.subsample = subsample;
    return encodeJPEG(src, o);
}

}  // namespace ipf
