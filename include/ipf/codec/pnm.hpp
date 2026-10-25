// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf codec: Netpbm formats (PBM/PGM/PPM, P1-P6) and PFM (portable float map).
#pragma once

#include <cctype>
#include <cstdio>
#include <sstream>

#include "common.hpp"

namespace ipf {

namespace detail {

class PnmTokenizer {
public:
    PnmTokenizer(const std::vector<std::uint8_t>& b, std::size_t pos) : b_(b), pos_(pos) {}
    void skipSpaceAndComments() {
        while (pos_ < b_.size()) {
            if (b_[pos_] == '#') {
                while (pos_ < b_.size() && b_[pos_] != '\n' && b_[pos_] != '\r') ++pos_;
            } else if (std::isspace(b_[pos_])) {
                ++pos_;
            } else {
                break;
            }
        }
    }
    long long number() {
        skipSpaceAndComments();
        if (pos_ >= b_.size() || !std::isdigit(b_[pos_])) fail("PNM: malformed header or pixel data");
        long long v = 0;
        while (pos_ < b_.size() && std::isdigit(b_[pos_])) {
            v = v * 10 + (b_[pos_++] - '0');
            if (v > (1LL << 31)) fail("PNM: number too large");
        }
        return v;
    }
    int bit() {  // P1 pixels may be written without separators
        skipSpaceAndComments();
        if (pos_ >= b_.size() || (b_[pos_] != '0' && b_[pos_] != '1')) fail("PNM: malformed P1 pixel data");
        return b_[pos_++] - '0';
    }
    std::string word() {
        skipSpaceAndComments();
        std::string s;
        while (pos_ < b_.size() && !std::isspace(b_[pos_])) s += static_cast<char>(b_[pos_++]);
        return s;
    }
    std::size_t& pos() { return pos_; }

private:
    const std::vector<std::uint8_t>& b_;
    std::size_t pos_;
};

}  // namespace detail

/// Decodes PBM/PGM/PPM (P1-P6). Returns 1 channel for P1/P2/P4/P5, 3 for P3/P6.
/// Samples are rescaled from the file's maxval to the full range of T
/// (use std::uint8_t or std::uint16_t).
template <class T = std::uint8_t>
Image<T> decodePNM(const std::vector<std::uint8_t>& bytes) {
    static_assert(std::is_integral<T>::value, "decodePNM: T must be an integer type");
    if (bytes.size() < 3 || bytes[0] != 'P' || bytes[1] < '1' || bytes[1] > '6')
        detail::fail("PNM: not a P1-P6 netpbm file");
    const int kind = bytes[1] - '0';
    detail::PnmTokenizer tok(bytes, 2);
    const long long w = tok.number(), h = tok.number();
    detail::checkDecodedSize(w, h, "PNM");
    const bool bitmap = kind == 1 || kind == 4;
    const long long maxval = bitmap ? 1 : tok.number();
    if (maxval < 1 || maxval > 65535) detail::fail("PNM: invalid maxval");
    const int cn = (kind == 3 || kind == 6) ? 3 : 1;
    Image<T> img(static_cast<int>(w), static_cast<int>(h), cn);
    const double tmax = PixelTraits<T>::max();
    auto scale = [&](long long v) -> T {
        if (v > maxval) v = maxval;
        return static_cast<T>((static_cast<double>(v) * tmax) / static_cast<double>(maxval) + 0.5);
    };
    T* d = img.data();
    const std::size_t n = img.sampleCount();
    if (kind == 1) {
        for (std::size_t i = 0; i < n; ++i) d[i] = tok.bit() ? T(0) : static_cast<T>(tmax);
    } else if (kind == 2 || kind == 3) {
        for (std::size_t i = 0; i < n; ++i) d[i] = scale(tok.number());
    } else {
        std::size_t& pos = tok.pos();
        if (pos >= bytes.size() || !std::isspace(bytes[pos])) detail::fail("PNM: malformed header");
        ++pos;  // exactly one whitespace byte separates header and raster
        detail::ByteReader rd(bytes.data(), bytes.size(), "PNM");
        rd.seek(pos);
        if (kind == 4) {
            const std::size_t rowBytes = (static_cast<std::size_t>(w) + 7) / 8;
            for (int y = 0; y < h; ++y) {
                const std::uint8_t* r = rd.take(rowBytes);
                for (int x = 0; x < w; ++x) img(x, y) = ((r[x >> 3] >> (7 - (x & 7))) & 1) ? T(0) : static_cast<T>(tmax);
            }
        } else if (maxval < 256) {
            const std::uint8_t* r = rd.take(n);
            for (std::size_t i = 0; i < n; ++i) d[i] = scale(r[i]);
        } else {
            const std::uint8_t* r = rd.take(n * 2);
            for (std::size_t i = 0; i < n; ++i) d[i] = scale((r[2 * i] << 8) | r[2 * i + 1]);
        }
    }
    return img;
}

/// Encodes a 1-channel image as PGM (P5/P2) or a 3-channel image as PPM (P6/P3).
/// 2/4-channel images have their alpha dropped. uint16 images are written with
/// maxval 65535, uint8 images with maxval 255.
template <class T>
std::vector<std::uint8_t> encodePNM(const Image<T>& src, bool binary = true) {
    static_assert(std::is_same<T, std::uint8_t>::value || std::is_same<T, std::uint16_t>::value,
                  "encodePNM: image must be uint8 or uint16");
    detail::require(!src.empty(), "encodePNM: empty image");
    const Image<T> img = convertChannels(src, src.channels() <= 2 ? 1 : 3);
    const unsigned maxval = sizeof(T) == 1 ? 255u : 65535u;
    std::ostringstream hdr;
    hdr << 'P' << (img.channels() == 1 ? (binary ? 5 : 2) : (binary ? 6 : 3)) << '\n'
        << img.width() << ' ' << img.height() << '\n' << maxval << '\n';
    const std::string h = hdr.str();
    std::vector<std::uint8_t> out(h.begin(), h.end());
    if (binary) {
        out.reserve(out.size() + img.sampleCount() * sizeof(T));
        for (T v : img) {
            if (sizeof(T) == 2) detail::put8(out, static_cast<unsigned>(v) >> 8);
            detail::put8(out, static_cast<unsigned>(v) & 0xFF);
        }
    } else {
        const std::size_t perLine = img.stride();
        std::size_t i = 0;
        for (T v : img) {
            const std::string s = std::to_string(static_cast<unsigned>(v));
            out.insert(out.end(), s.begin(), s.end());
            out.push_back(++i % perLine == 0 ? '\n' : ' ');
        }
    }
    return out;
}

/// Decodes a PFM (Portable Float Map, "PF" = RGB, "Pf" = gray) into a float image.
/// Values are returned as stored (not normalized).
inline ImageF decodePFM(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() < 3 || bytes[0] != 'P' || (bytes[1] != 'F' && bytes[1] != 'f'))
        detail::fail("PFM: not a PFM file");
    const int cn = bytes[1] == 'F' ? 3 : 1;
    detail::PnmTokenizer tok(bytes, 2);
    const long long w = tok.number(), h = tok.number();
    detail::checkDecodedSize(w, h, "PFM");
    const std::string scaleStr = tok.word();
    double scale = 0;
    try {
        scale = std::stod(scaleStr);
    } catch (...) {
        detail::fail("PFM: invalid scale value");
    }
    if (scale == 0) detail::fail("PFM: invalid scale value");
    const bool little = scale < 0;
    std::size_t pos = tok.pos();
    if (pos >= bytes.size()) detail::fail("PFM: truncated");
    ++pos;  // single whitespace
    detail::ByteReader rd(bytes.data(), bytes.size(), "PFM");
    rd.seek(pos);
    ImageF img(static_cast<int>(w), static_cast<int>(h), cn);
    for (long long y = h - 1; y >= 0; --y) {  // PFM rows are stored bottom-to-top
        float* row = img.row(static_cast<int>(y));
        for (std::size_t i = 0; i < img.stride(); ++i) {
            const std::uint32_t u = little ? rd.u32le() : rd.u32be();
            std::memcpy(&row[i], &u, 4);
        }
    }
    return img;
}

/// Encodes a 1- or 3-channel float image as little-endian PFM.
template <class T>
std::vector<std::uint8_t> encodePFM(const Image<T>& src) {
    detail::require(src.channels() == 1 || src.channels() == 3, "encodePFM: image must have 1 or 3 channels");
    detail::require(!src.empty(), "encodePFM: empty image");
    const std::string h = std::string(src.channels() == 3 ? "PF" : "Pf") + "\n" + std::to_string(src.width()) +
                          " " + std::to_string(src.height()) + "\n-1.0\n";
    std::vector<std::uint8_t> out(h.begin(), h.end());
    out.reserve(out.size() + src.sampleCount() * 4);
    for (int y = src.height() - 1; y >= 0; --y) {
        const T* row = src.row(y);
        for (std::size_t i = 0; i < src.stride(); ++i) {
            const float f = static_cast<float>(row[i]);
            std::uint32_t u;
            std::memcpy(&u, &f, 4);
            detail::put32le(out, u);
        }
    }
    return out;
}

}  // namespace ipf
