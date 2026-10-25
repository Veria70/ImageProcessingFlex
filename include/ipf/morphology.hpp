// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf morphology: erosion/dilation with arbitrary structuring elements, compound
// operations (open, close, gradient, top-hat, black-hat, hit-or-miss) and binary
// shape tools (thinning/skeletonization, hole filling, boundaries).
//
// Pixels outside the image never influence the result (OpenCV's default behavior).
// Binary helpers treat any non-zero pixel as foreground and return 0/255 masks.
#pragma once

#include <deque>

#include "core.hpp"

namespace ipf {

/// A structuring element: single-channel uint8 mask (non-zero = part of the element),
/// anchored at its center (width / 2, height / 2).
using StructuringElement = ImageU8;

enum class MorphShape { Rect, Cross, Ellipse };

/// Creates a structuring element (same shapes as OpenCV's getStructuringElement).
inline StructuringElement getStructuringElement(MorphShape shape, int width, int height = 0) {
    if (height <= 0) height = width;
    detail::require(width > 0 && height > 0, "getStructuringElement: size must be positive");
    StructuringElement se(width, height, 1, 0);
    const int r = height / 2, c = width / 2;
    const double invR2 = r ? 1.0 / (static_cast<double>(r) * r) : 0.0;
    for (int i = 0; i < height; ++i) {
        int j1 = 0, j2 = 0;
        if (shape == MorphShape::Rect || (shape == MorphShape::Cross && i == r)) {
            j1 = 0;
            j2 = width;
        } else if (shape == MorphShape::Cross) {
            j1 = c;
            j2 = c + 1;
        } else {
            const int dy = i - r;
            if (std::abs(dy) <= r) {
                const int dx = saturate_cast<int>(c * std::sqrt((r * r - dy * dy) * invR2));
                j1 = std::max(c - dx, 0);
                j2 = std::min(c + dx + 1, width);
            }
        }
        for (int j = j1; j < j2; ++j) se(j, i) = 1;
    }
    return se;
}

namespace detail {

template <class T, bool kMin>
Image<T> morphOnce(const Image<T>& src, const StructuringElement& se) {
    require(se.channels() == 1 && !se.empty(), "morphology: structuring element must be a non-empty single-channel mask");
    const int W = src.width(), H = src.height(), cn = src.channels();
    const int ax = se.width() / 2, ay = se.height() / 2;
    auto better = [](T a, T b) { return kMin ? (a < b) : (b < a); };
    const bool isRect = std::all_of(se.begin(), se.end(), [](std::uint8_t v) { return v != 0; });
    if (isRect) {  // separable: 1D min/max along rows, then columns
        Image<T> tmp(W, H, cn), out(W, H, cn);
        const int kw = se.width(), kh = se.height();
        parallelFor(0, H, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x)
                    for (int c = 0; c < cn; ++c) {
                        T v = src(x, y, c);
                        for (int i = std::max(0, x - ax); i <= std::min(W - 1, x - ax + kw - 1); ++i)
                            if (better(src(i, y, c), v)) v = src(i, y, c);
                        tmp(x, y, c) = v;
                    }
        }, static_cast<double>(W) * cn * kw);
        parallelFor(0, H, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x)
                    for (int c = 0; c < cn; ++c) {
                        T v = tmp(x, y, c);
                        for (int j = std::max(0, y - ay); j <= std::min(H - 1, y - ay + kh - 1); ++j)
                            if (better(tmp(x, j, c), v)) v = tmp(x, j, c);
                        out(x, y, c) = v;
                    }
        }, static_cast<double>(W) * cn * kh);
        return out;
    }
    std::vector<Point> offs;
    for (int j = 0; j < se.height(); ++j)
        for (int i = 0; i < se.width(); ++i)
            if (se(i, j)) offs.push_back({i - ax, j - ay});
    Image<T> out(W, H, cn);
    parallelFor(0, H, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < cn; ++c) {
                    bool any = false;
                    T v = src(x, y, c);
                    for (const Point& o : offs) {
                        const int sx = x + o.x, sy = y + o.y;
                        if (sx < 0 || sy < 0 || sx >= W || sy >= H) continue;
                        const T sv = src(sx, sy, c);
                        if (!any || better(sv, v)) v = sv;
                        any = true;
                    }
                    out(x, y, c) = v;
                }
    }, static_cast<double>(W) * cn * offs.size());
    return out;
}

}  // namespace detail

/// Grayscale erosion (minimum over the structuring element), applied `iterations` times.
template <class T>
Image<T> erode(const Image<T>& src, const StructuringElement& se = getStructuringElement(MorphShape::Rect, 3),
               int iterations = 1) {
    Image<T> out = src;
    for (int i = 0; i < iterations; ++i) out = detail::morphOnce<T, true>(out, se);
    return out;
}

/// Grayscale dilation (maximum over the structuring element), applied `iterations` times.
template <class T>
Image<T> dilate(const Image<T>& src, const StructuringElement& se = getStructuringElement(MorphShape::Rect, 3),
                int iterations = 1) {
    Image<T> out = src;
    for (int i = 0; i < iterations; ++i) out = detail::morphOnce<T, false>(out, se);
    return out;
}

enum class MorphOp {
    Erode,
    Dilate,
    Open,      ///< dilate(erode(x)): removes small bright details
    Close,     ///< erode(dilate(x)): fills small dark gaps
    Gradient,  ///< dilate(x) - erode(x): object outlines
    TopHat,    ///< x - open(x): small bright details
    BlackHat   ///< close(x) - x: small dark details
};

/// Compound morphological operation (OpenCV's morphologyEx semantics for iterations).
template <class T>
Image<T> morphologyEx(const Image<T>& src, MorphOp op,
                      const StructuringElement& se = getStructuringElement(MorphShape::Rect, 3), int iterations = 1) {
    auto diff = [](const Image<T>& a, const Image<T>& b) {
        Image<T> out(a.width(), a.height(), a.channels());
        for (std::size_t i = 0; i < a.sampleCount(); ++i)
            out.data()[i] = saturate_cast<T>(static_cast<double>(a.data()[i]) - static_cast<double>(b.data()[i]));
        return out;
    };
    switch (op) {
        case MorphOp::Erode: return erode(src, se, iterations);
        case MorphOp::Dilate: return dilate(src, se, iterations);
        case MorphOp::Open: return dilate(erode(src, se, iterations), se, iterations);
        case MorphOp::Close: return erode(dilate(src, se, iterations), se, iterations);
        case MorphOp::Gradient: return diff(dilate(src, se, iterations), erode(src, se, iterations));
        case MorphOp::TopHat: return diff(src, dilate(erode(src, se, iterations), se, iterations));
        default: return diff(erode(dilate(src, se, iterations), se, iterations), src);
    }
}

/// Binary hit-or-miss transform. `hit` pixels must be foreground and `miss` pixels background
/// (both masks share the same size/anchor). Returns a 0/255 mask of matching positions.
inline ImageU8 hitOrMiss(const ImageU8& binary, const StructuringElement& hit, const StructuringElement& miss) {
    detail::require(binary.channels() == 1, "hitOrMiss: expected a single-channel binary image");
    detail::require(hit.sameShape(miss), "hitOrMiss: hit and miss masks must have the same size");
    const int W = binary.width(), H = binary.height(), ax = hit.width() / 2, ay = hit.height() / 2;
    ImageU8 out(W, H, 1, 0);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            bool ok = true;
            for (int j = 0; j < hit.height() && ok; ++j)
                for (int i = 0; i < hit.width() && ok; ++i) {
                    const int sx = x + i - ax, sy = y + j - ay;
                    const bool fg = sx >= 0 && sy >= 0 && sx < W && sy < H && binary(sx, sy) != 0;
                    if ((hit(i, j) && !fg) || (miss(i, j) && fg)) ok = false;
                }
            if (ok) out(x, y) = 255;
        }
    return out;
}

/// Zhang-Suen thinning: reduces foreground shapes to 1-pixel-wide skeletons (0/255 output).
inline ImageU8 thinning(const ImageU8& binary) {
    detail::require(binary.channels() == 1, "thinning: expected a single-channel binary image");
    const int W = binary.width(), H = binary.height();
    ImageU8 img(W, H, 1);
    for (std::size_t i = 0; i < img.pixelCount(); ++i) img.data()[i] = binary.data()[i] ? 1 : 0;
    auto px = [&](int x, int y) -> int { return (x < 0 || y < 0 || x >= W || y >= H) ? 0 : img(x, y); };
    std::vector<Point> toClear;
    bool changed = true;
    while (changed) {
        changed = false;
        for (int pass = 0; pass < 2; ++pass) {
            toClear.clear();
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    if (!img(x, y)) continue;
                    // neighbours P2..P9 clockwise starting north
                    const int p[8] = {px(x, y - 1), px(x + 1, y - 1), px(x + 1, y), px(x + 1, y + 1),
                                      px(x, y + 1), px(x - 1, y + 1), px(x - 1, y), px(x - 1, y - 1)};
                    int b = 0, a = 0;
                    for (int k = 0; k < 8; ++k) {
                        b += p[k];
                        if (!p[k] && p[(k + 1) % 8]) ++a;
                    }
                    if (b < 2 || b > 6 || a != 1) continue;
                    if (pass == 0 && (p[0] * p[2] * p[4] || p[2] * p[4] * p[6])) continue;
                    if (pass == 1 && (p[0] * p[2] * p[6] || p[0] * p[4] * p[6])) continue;
                    toClear.push_back({x, y});
                }
            for (const Point& q : toClear) img(q.x, q.y) = 0;
            if (!toClear.empty()) changed = true;
        }
    }
    for (auto& v : img) v = v ? 255 : 0;
    return img;
}

/// Fills holes: background regions not connected (4-connectivity) to the image border
/// become foreground. Returns a 0/255 mask.
inline ImageU8 fillHoles(const ImageU8& binary) {
    detail::require(binary.channels() == 1, "fillHoles: expected a single-channel binary image");
    const int W = binary.width(), H = binary.height();
    ImageU8 outside(W, H, 1, 0);
    std::deque<Point> q;
    auto seed = [&](int x, int y) {
        if (!binary(x, y) && !outside(x, y)) {
            outside(x, y) = 1;
            q.push_back({x, y});
        }
    };
    for (int x = 0; x < W; ++x) {
        seed(x, 0);
        seed(x, H - 1);
    }
    for (int y = 0; y < H; ++y) {
        seed(0, y);
        seed(W - 1, y);
    }
    while (!q.empty()) {
        const Point p = q.front();
        q.pop_front();
        const Point nb[4] = {{p.x + 1, p.y}, {p.x - 1, p.y}, {p.x, p.y + 1}, {p.x, p.y - 1}};
        for (const Point& n : nb)
            if (n.x >= 0 && n.y >= 0 && n.x < W && n.y < H) seed(n.x, n.y);
    }
    ImageU8 out(W, H, 1);
    for (std::size_t i = 0; i < out.pixelCount(); ++i) out.data()[i] = outside.data()[i] ? 0 : 255;
    return out;
}

/// Inner boundary of binary shapes: foreground pixels with at least one background
/// neighbour under the structuring element (binary - erode(binary)).
inline ImageU8 boundary(const ImageU8& binary, const StructuringElement& se = getStructuringElement(MorphShape::Cross, 3)) {
    detail::require(binary.channels() == 1, "boundary: expected a single-channel binary image");
    ImageU8 bin(binary.width(), binary.height(), 1);
    for (std::size_t i = 0; i < bin.pixelCount(); ++i) bin.data()[i] = binary.data()[i] ? 255 : 0;
    const ImageU8 er = erode(bin, se);
    for (std::size_t i = 0; i < bin.pixelCount(); ++i) bin.data()[i] = (bin.data()[i] && !er.data()[i]) ? 255 : 0;
    return bin;
}

}  // namespace ipf
