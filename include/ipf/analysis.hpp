// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf analysis: connected components, contour tracing and shape descriptors, image
// moments, distance transforms and flood filling.
//
// Binary inputs: any non-zero pixel of a single-channel image is foreground.
#pragma once

#include <array>
#include <deque>

#include "core.hpp"

namespace ipf {

// ---------------------------------------------------------------------------
// Connected components
// ---------------------------------------------------------------------------

struct ComponentStats {
    int label = 0;
    long long area = 0;  ///< number of pixels
    Rect bbox;           ///< bounding box
    PointF centroid;     ///< mean pixel position
};

struct Components {
    ImageI32 labels;                   ///< 0 = background, 1..count = component labels
    int count = 0;                     ///< number of components (background excluded)
    std::vector<ComponentStats> stats;  ///< stats[i] describes label i + 1
};

/// Labels connected foreground regions (connectivity 4 or 8). Labels are numbered in
/// raster order of each component's first pixel (same as scipy.ndimage.label).
template <class T>
Components connectedComponents(const Image<T>& binary, int connectivity = 8) {
    detail::require(binary.channels() == 1, "connectedComponents: expected a single-channel image");
    detail::require(connectivity == 4 || connectivity == 8, "connectedComponents: connectivity must be 4 or 8");
    const int W = binary.width(), H = binary.height();
    Components res;
    res.labels.reset(W, H, 1, 0);
    std::vector<int> parent{0};
    auto find = [&parent](int a) {
        while (parent[a] != a) a = parent[a] = parent[parent[a]];
        return a;
    };
    auto unite = [&](int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
    };
    ImageI32& L = res.labels;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            if (binary(x, y) == T(0)) continue;
            int nb[4], n = 0;
            if (x > 0 && L(x - 1, y)) nb[n++] = L(x - 1, y);
            if (y > 0 && L(x, y - 1)) nb[n++] = L(x, y - 1);
            if (connectivity == 8 && y > 0) {
                if (x > 0 && L(x - 1, y - 1)) nb[n++] = L(x - 1, y - 1);
                if (x + 1 < W && L(x + 1, y - 1)) nb[n++] = L(x + 1, y - 1);
            }
            if (n == 0) {
                parent.push_back(static_cast<int>(parent.size()));
                L(x, y) = static_cast<int>(parent.size()) - 1;
            } else {
                int m = nb[0];
                for (int k = 1; k < n; ++k) m = std::min(m, nb[k]);
                L(x, y) = m;
                for (int k = 0; k < n; ++k) unite(m, nb[k]);
            }
        }
    // Relabel to consecutive numbers in raster order of first appearance.
    std::vector<int> remap(parent.size(), 0);
    for (int& v : L) {
        if (!v) continue;
        const int root = find(v);
        if (!remap[root]) remap[root] = ++res.count;
        v = remap[root];
    }
    res.stats.resize(static_cast<std::size_t>(res.count));
    std::vector<int> x0(res.count, W), y0(res.count, H), x1(res.count, -1), y1(res.count, -1);
    std::vector<double> sx(res.count, 0), sy(res.count, 0);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const int l = L(x, y);
            if (!l) continue;
            ComponentStats& s = res.stats[l - 1];
            ++s.area;
            sx[l - 1] += x;
            sy[l - 1] += y;
            x0[l - 1] = std::min(x0[l - 1], x);
            y0[l - 1] = std::min(y0[l - 1], y);
            x1[l - 1] = std::max(x1[l - 1], x);
            y1[l - 1] = std::max(y1[l - 1], y);
        }
    for (int i = 0; i < res.count; ++i) {
        ComponentStats& s = res.stats[i];
        s.label = i + 1;
        s.bbox = Rect(x0[i], y0[i], x1[i] - x0[i] + 1, y1[i] - y0[i] + 1);
        s.centroid = PointF(sx[i] / static_cast<double>(s.area), sy[i] / static_cast<double>(s.area));
    }
    return res;
}

/// Removes foreground components smaller than minArea pixels (returns a 0/255 mask).
template <class T>
ImageU8 removeSmallObjects(const Image<T>& binary, long long minArea, int connectivity = 8) {
    const Components cc = connectedComponents(binary, connectivity);
    ImageU8 out(binary.width(), binary.height(), 1, 0);
    for (std::size_t i = 0; i < out.pixelCount(); ++i) {
        const int l = cc.labels.data()[i];
        if (l && cc.stats[l - 1].area >= minArea) out.data()[i] = 255;
    }
    return out;
}

/// Keeps only the largest foreground component (returns a 0/255 mask; empty input -> all zero).
template <class T>
ImageU8 largestComponent(const Image<T>& binary, int connectivity = 8) {
    const Components cc = connectedComponents(binary, connectivity);
    ImageU8 out(binary.width(), binary.height(), 1, 0);
    int best = 0;
    for (const auto& s : cc.stats)
        if (!best || s.area > cc.stats[best - 1].area) best = s.label;
    for (std::size_t i = 0; i < out.pixelCount(); ++i)
        if (best && cc.labels.data()[i] == best) out.data()[i] = 255;
    return out;
}

// ---------------------------------------------------------------------------
// Contours (Suzuki & Abe border following)
// ---------------------------------------------------------------------------

struct Contour {
    std::vector<Point> points;  ///< boundary pixels in tracing order (8-connected chain)
    bool isHole = false;        ///< true for the inner border of a hole
    int parent = -1;            ///< index of the enclosing contour, -1 for top level
};

/// Traces all borders of the foreground (8-connected objects, 4-connected holes) using
/// Suzuki & Abe's algorithm (as OpenCV's findContours with RETR_TREE, CHAIN_APPROX_NONE).
/// Outer borders are traced counter-clockwise (as displayed) and hole borders clockwise.
template <class T>
std::vector<Contour> findContours(const Image<T>& binary) {
    detail::require(binary.channels() == 1, "findContours: expected a single-channel image");
    const int W = binary.width() + 2, H = binary.height() + 2;  // zero frame around the image
    std::vector<int> f(static_cast<std::size_t>(W) * H, 0);
    for (int y = 0; y < binary.height(); ++y)
        for (int x = 0; x < binary.width(); ++x)
            if (binary(x, y) != T(0)) f[static_cast<std::size_t>(y + 1) * W + x + 1] = 1;
    auto F = [&](int x, int y) -> int& { return f[static_cast<std::size_t>(y) * W + x]; };
    // 8-neighbourhood in clockwise order (y axis down): E, SE, S, SW, W, NW, N, NE
    static const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    static const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    auto dirOf = [](int ddx, int ddy) {
        for (int k = 0; k < 8; ++k)
            if (dx[k] == ddx && dy[k] == ddy) return k;
        return 0;
    };
    std::vector<Contour> contours;
    std::vector<int> nbdToContour{-1, -1};  // NBD 1 = the frame (no contour)
    int nbd = 1;
    for (int y = 1; y < H - 1; ++y) {
        int lnbd = 1;
        for (int x = 1; x < W - 1; ++x) {
            const int v = F(x, y);
            if (v == 0) continue;
            bool outer = false, start = false;
            int fromX = 0, fromY = 0;
            if (v == 1 && F(x - 1, y) == 0) {
                outer = true;
                start = true;
                fromX = x - 1;
                fromY = y;
            } else if (v >= 1 && F(x + 1, y) == 0) {
                start = true;
                fromX = x + 1;
                fromY = y;
                if (v > 1) lnbd = v;
            }
            if (start) {
                ++nbd;
                Contour c;
                c.isHole = !outer;
                const int lc = lnbd < static_cast<int>(nbdToContour.size()) ? nbdToContour[lnbd] : -1;
                if (lc >= 0)  // same border type as LNBD -> sibling, different type -> child
                    c.parent = (outer == !contours[lc].isHole) ? contours[lc].parent : lc;
                // 3.1: search clockwise around (x, y) starting at (fromX, fromY) for a non-zero pixel
                int d0 = dirOf(fromX - x, fromY - y), found = -1;
                for (int k = 0; k < 8; ++k) {
                    const int d = (d0 + k) % 8;
                    if (F(x + dx[d], y + dy[d]) != 0) {
                        found = d;
                        break;
                    }
                }
                if (found < 0) {  // isolated pixel
                    F(x, y) = -nbd;
                    c.points.push_back({x - 1, y - 1});
                } else {
                    const int x1 = x + dx[found], y1 = y + dy[found];
                    int x2 = x1, y2 = y1, x3 = x, y3 = y;
                    for (;;) {
                        // 3.3: search counter-clockwise around (x3, y3) starting after (x2, y2)
                        const int dPrev = dirOf(x2 - x3, y2 - y3);
                        int x4 = 0, y4 = 0;
                        bool eastZeroExamined = false;
                        for (int k = 1; k <= 8; ++k) {
                            const int d = (dPrev - k + 16) % 8;
                            const int nx = x3 + dx[d], ny = y3 + dy[d];
                            if (F(nx, ny) != 0) {
                                x4 = nx;
                                y4 = ny;
                                break;
                            }
                            if (d == 0) eastZeroExamined = true;
                        }
                        // 3.4
                        if (eastZeroExamined) F(x3, y3) = -nbd;
                        else if (F(x3, y3) == 1) F(x3, y3) = nbd;
                        c.points.push_back({x3 - 1, y3 - 1});
                        // 3.5
                        if (x4 == x && y4 == y && x3 == x1 && y3 == y1) break;
                        x2 = x3;
                        y2 = y3;
                        x3 = x4;
                        y3 = y4;
                    }
                }
                nbdToContour.push_back(static_cast<int>(contours.size()));
                contours.push_back(std::move(c));
            }
            const int after = F(x, y);
            if (after != 1) lnbd = std::abs(after);
        }
    }
    return contours;
}

/// Signed polygon area (shoelace formula); positive for counter-clockwise as displayed
/// (i.e. clockwise in y-down coordinates gives a negative value).
inline double contourAreaSigned(const std::vector<Point>& poly) {
    double a = 0;
    const std::size_t n = poly.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Point& p = poly[i];
        const Point& q = poly[(i + 1) % n];
        a += static_cast<double>(p.x) * q.y - static_cast<double>(q.x) * p.y;
    }
    return -a / 2.0;
}
/// Absolute polygon area.
inline double contourArea(const std::vector<Point>& poly) { return std::fabs(contourAreaSigned(poly)); }

/// Perimeter of a polyline (closed = true adds the closing segment).
inline double arcLength(const std::vector<Point>& poly, bool closed = true) {
    double len = 0;
    for (std::size_t i = 0; i + 1 < poly.size(); ++i) len += std::hypot(poly[i + 1].x - poly[i].x, poly[i + 1].y - poly[i].y);
    if (closed && poly.size() > 1) len += std::hypot(poly.front().x - poly.back().x, poly.front().y - poly.back().y);
    return len;
}

/// Axis-aligned bounding box of a point set.
inline Rect boundingRect(const std::vector<Point>& pts) {
    if (pts.empty()) return Rect();
    int x0 = pts[0].x, y0 = pts[0].y, x1 = x0, y1 = y0;
    for (const Point& p : pts) {
        x0 = std::min(x0, p.x);
        y0 = std::min(y0, p.y);
        x1 = std::max(x1, p.x);
        y1 = std::max(y1, p.y);
    }
    return Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
}

/// Convex hull (Andrew's monotone chain). Collinear points are dropped.
inline std::vector<Point> convexHull(std::vector<Point> pts) {
    std::sort(pts.begin(), pts.end(), [](Point a, Point b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
    if (pts.size() < 3) return pts;
    auto cross = [](Point o, Point a, Point b) {
        return static_cast<long long>(a.x - o.x) * (b.y - o.y) - static_cast<long long>(a.y - o.y) * (b.x - o.x);
    };
    std::vector<Point> h(2 * pts.size());
    std::size_t k = 0;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        while (k >= 2 && cross(h[k - 2], h[k - 1], pts[i]) <= 0) --k;
        h[k++] = pts[i];
    }
    for (std::size_t i = pts.size() - 1, t = k + 1; i > 0; --i) {
        while (k >= t && cross(h[k - 2], h[k - 1], pts[i - 1]) <= 0) --k;
        h[k++] = pts[i - 1];
    }
    h.resize(k - 1);
    return h;
}

/// Douglas-Peucker polyline simplification with tolerance epsilon (pixels).
inline std::vector<Point> approxPolyDP(const std::vector<Point>& poly, double epsilon, bool closed = true) {
    if (poly.size() < 3) return poly;
    std::vector<char> keep(poly.size(), 0);
    auto distToSeg = [](Point p, Point a, Point b) {
        const double vx = b.x - a.x, vy = b.y - a.y, len2 = vx * vx + vy * vy;
        if (len2 == 0) return std::hypot(p.x - a.x, p.y - a.y);
        const double t = clamp(((p.x - a.x) * vx + (p.y - a.y) * vy) / len2, 0.0, 1.0);
        return std::hypot(p.x - (a.x + t * vx), p.y - (a.y + t * vy));
    };
    std::vector<std::pair<std::size_t, std::size_t>> stack;
    std::size_t first = 0, last = poly.size() - 1;
    if (closed) {  // split at the point farthest from the first point
        double best = -1;
        for (std::size_t i = 1; i < poly.size(); ++i) {
            const double d = std::hypot(poly[i].x - poly[0].x, poly[i].y - poly[0].y);
            if (d > best) {
                best = d;
                last = i;
            }
        }
        stack.push_back({last, poly.size()});  // second half wraps to index 0 (== size)
    }
    keep[first] = keep[last] = 1;
    stack.push_back({first, last});
    while (!stack.empty()) {
        const auto [a, b] = stack.back();
        stack.pop_back();
        const Point pa = poly[a], pb = poly[b % poly.size()];
        double best = -1;
        std::size_t idx = a;
        for (std::size_t i = a + 1; i < b; ++i) {
            const double d = distToSeg(poly[i], pa, pb);
            if (d > best) {
                best = d;
                idx = i;
            }
        }
        if (best > epsilon) {
            keep[idx] = 1;
            stack.push_back({a, idx});
            stack.push_back({idx, b});
        }
    }
    std::vector<Point> out;
    for (std::size_t i = 0; i < poly.size(); ++i)
        if (keep[i]) out.push_back(poly[i]);
    return out;
}

/// Even-odd point-in-polygon test (points on the boundary may be reported either way).
inline bool pointInPolygon(const std::vector<Point>& poly, PointF p) {
    bool in = false;
    for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
        const double xi = poly[i].x, yi = poly[i].y, xj = poly[j].x, yj = poly[j].y;
        if ((yi > p.y) != (yj > p.y) && p.x < (xj - xi) * (p.y - yi) / (yj - yi) + xi) in = !in;
    }
    return in;
}

struct Circle {
    PointF center;
    double radius = 0;
};

/// Smallest circle enclosing all points (Welzl's algorithm, expected linear time).
inline Circle minEnclosingCircle(std::vector<Point> pts) {
    if (pts.empty()) return Circle();
    // deterministic shuffle for expected O(n)
    std::uint32_t s = 2463534242u;
    for (std::size_t i = pts.size(); i > 1; --i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        std::swap(pts[i - 1], pts[s % i]);
    }
    auto inside = [](const Circle& c, Point p) { return std::hypot(p.x - c.center.x, p.y - c.center.y) <= c.radius + 1e-7; };
    auto two = [](Point a, Point b) {
        return Circle{PointF((a.x + b.x) / 2.0, (a.y + b.y) / 2.0), std::hypot(a.x - b.x, a.y - b.y) / 2.0};
    };
    auto three = [&](Point a, Point b, Point c) {
        const double bx = b.x - a.x, by = b.y - a.y, cx = c.x - a.x, cy = c.y - a.y;
        const double d = 2 * (bx * cy - by * cx);
        if (std::fabs(d) < 1e-12) {  // collinear: use the widest pair
            Circle c1 = two(a, b), c2 = two(a, c), c3 = two(b, c);
            return c1.radius > c2.radius ? (c1.radius > c3.radius ? c1 : c3) : (c2.radius > c3.radius ? c2 : c3);
        }
        const double ux = (cy * (bx * bx + by * by) - by * (cx * cx + cy * cy)) / d;
        const double uy = (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by)) / d;
        return Circle{PointF(a.x + ux, a.y + uy), std::hypot(ux, uy)};
    };
    Circle c{PointF(pts[0]), 0};
    for (std::size_t i = 1; i < pts.size(); ++i) {
        if (inside(c, pts[i])) continue;
        c = Circle{PointF(pts[i]), 0};
        for (std::size_t j = 0; j < i; ++j) {
            if (inside(c, pts[j])) continue;
            c = two(pts[i], pts[j]);
            for (std::size_t k = 0; k < j; ++k)
                if (!inside(c, pts[k])) c = three(pts[i], pts[j], pts[k]);
        }
    }
    return c;
}

// ---------------------------------------------------------------------------
// Moments
// ---------------------------------------------------------------------------

struct Moments {
    double m00 = 0, m10 = 0, m01 = 0, m20 = 0, m11 = 0, m02 = 0, m30 = 0, m21 = 0, m12 = 0, m03 = 0;  ///< raw
    double mu20 = 0, mu11 = 0, mu02 = 0, mu30 = 0, mu21 = 0, mu12 = 0, mu03 = 0;                     ///< central
    double nu20 = 0, nu11 = 0, nu02 = 0, nu30 = 0, nu21 = 0, nu12 = 0, nu03 = 0;                     ///< normalized
    PointF centroid() const { return m00 != 0 ? PointF(m10 / m00, m01 / m00) : PointF(); }
    /// Orientation of the major axis in degrees (counter-clockwise as displayed), in (-90, 90].
    double orientation() const { return -0.5 * std::atan2(2 * mu11, mu20 - mu02) * 180.0 / detail::kPi; }
    /// The seven Hu invariants (translation, scale and rotation invariant).
    std::array<double, 7> hu() const {
        const double t0 = nu30 + nu12, t1 = nu21 + nu03, q0 = nu30 - 3 * nu12, q1 = 3 * nu21 - nu03;
        return {nu20 + nu02,
                (nu20 - nu02) * (nu20 - nu02) + 4 * nu11 * nu11,
                q0 * q0 + q1 * q1,
                t0 * t0 + t1 * t1,
                q0 * t0 * (t0 * t0 - 3 * t1 * t1) + q1 * t1 * (3 * t0 * t0 - t1 * t1),
                (nu20 - nu02) * (t0 * t0 - t1 * t1) + 4 * nu11 * t0 * t1,
                q1 * t0 * (t0 * t0 - 3 * t1 * t1) - q0 * t1 * (3 * t0 * t0 - t1 * t1)};
    }
};

/// Spatial moments of a single-channel image (pixel values are weights; set binary = true
/// to treat every non-zero pixel as weight 1).
template <class T>
Moments moments(const Image<T>& img, bool binary = false) {
    detail::require(img.channels() == 1, "moments: expected a single-channel image");
    Moments m;
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); ++x) {
            double v = static_cast<double>(img(x, y));
            if (binary) v = v != 0 ? 1.0 : 0.0;
            if (v == 0) continue;
            const double xx = x, yy = y;
            m.m00 += v;
            m.m10 += v * xx;
            m.m01 += v * yy;
            m.m20 += v * xx * xx;
            m.m11 += v * xx * yy;
            m.m02 += v * yy * yy;
            m.m30 += v * xx * xx * xx;
            m.m21 += v * xx * xx * yy;
            m.m12 += v * xx * yy * yy;
            m.m03 += v * yy * yy * yy;
        }
    if (m.m00 == 0) return m;
    const double cx = m.m10 / m.m00, cy = m.m01 / m.m00;
    m.mu20 = m.m20 - cx * m.m10;
    m.mu11 = m.m11 - cx * m.m01;
    m.mu02 = m.m02 - cy * m.m01;
    m.mu30 = m.m30 - 3 * cx * m.m20 + 2 * cx * cx * m.m10;
    m.mu21 = m.m21 - 2 * cx * m.m11 - cy * m.m20 + 2 * cx * cx * m.m01;
    m.mu12 = m.m12 - 2 * cy * m.m11 - cx * m.m02 + 2 * cy * cy * m.m10;
    m.mu03 = m.m03 - 3 * cy * m.m02 + 2 * cy * cy * m.m01;
    const double s2 = m.m00 * m.m00, s3 = std::pow(m.m00, 2.5);
    m.nu20 = m.mu20 / s2;
    m.nu11 = m.mu11 / s2;
    m.nu02 = m.mu02 / s2;
    m.nu30 = m.mu30 / s3;
    m.nu21 = m.mu21 / s3;
    m.nu12 = m.mu12 / s3;
    m.nu03 = m.mu03 / s3;
    return m;
}

// ---------------------------------------------------------------------------
// Distance transform
// ---------------------------------------------------------------------------

enum class DistanceType {
    L1,          ///< city-block (4-connected steps)
    L2,          ///< exact Euclidean distance
    Chessboard   ///< L-infinity (8-connected steps)
};

namespace detail {
/// 1D squared-distance transform (Felzenszwalb & Huttenlocher lower envelope of parabolas).
/// "Infinite" samples are represented by a large finite value (kFar).
constexpr double kFar = 1e20;
inline void edt1d(const double* f, double* d, int n, std::vector<int>& v, std::vector<double>& z) {
    auto inter = [&](int q, int p) {
        return ((f[q] + static_cast<double>(q) * q) - (f[p] + static_cast<double>(p) * p)) / (2.0 * q - 2.0 * p);
    };
    int k = 0;
    v[0] = 0;
    z[0] = -kFar * 10;
    z[1] = kFar * 10;
    for (int q = 1; q < n; ++q) {
        double s = inter(q, v[k]);
        while (k > 0 && s <= z[k]) s = inter(q, v[--k]);
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = kFar * 10;
    }
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < q) ++k;
        const double diff = q - v[k];
        d[q] = diff * diff + f[v[k]];
    }
}
}  // namespace detail

/// Distance from every foreground (non-zero) pixel to the nearest background (zero) pixel;
/// background pixels get 0. If the image has no background, all distances are +infinity.
template <class T>
ImageF distanceTransform(const Image<T>& binary, DistanceType type = DistanceType::L2) {
    detail::require(binary.channels() == 1, "distanceTransform: expected a single-channel image");
    const int W = binary.width(), H = binary.height();
    ImageF out(W, H, 1);
    if (binary.empty()) return out;
    const double INF = 1e300;
    if (type == DistanceType::L2) {
        std::vector<double> g(static_cast<std::size_t>(W) * H);
        const std::size_t span = static_cast<std::size_t>(std::max(W, H));
        parallelFor(0, W, [&](int x0, int x1) {  // columns are independent
            std::vector<double> col(static_cast<std::size_t>(H)), dcol(static_cast<std::size_t>(H)), z(span + 2);
            std::vector<int> v(span + 1);
            for (int x = x0; x < x1; ++x) {
                for (int y = 0; y < H; ++y) col[y] = binary(x, y) != T(0) ? detail::kFar : 0.0;
                detail::edt1d(col.data(), dcol.data(), H, v, z);
                for (int y = 0; y < H; ++y) g[static_cast<std::size_t>(y) * W + x] = dcol[y];
            }
        }, static_cast<double>(H) * 8);
        parallelFor(0, H, [&](int y0, int y1) {  // then rows
            std::vector<double> row(static_cast<std::size_t>(W)), z(span + 2);
            std::vector<int> v(span + 1);
            for (int y = y0; y < y1; ++y) {
                detail::edt1d(&g[static_cast<std::size_t>(y) * W], row.data(), W, v, z);
                for (int x = 0; x < W; ++x)
                    out(x, y) = row[x] >= detail::kFar / 2 ? std::numeric_limits<float>::infinity()
                                                           : static_cast<float>(std::sqrt(row[x]));
            }
        }, static_cast<double>(W) * 8);
        return out;
    }
    // Two-pass chamfer propagation (exact for L1 and chessboard metrics).
    const bool chess = type == DistanceType::Chessboard;
    std::vector<double> d(static_cast<std::size_t>(W) * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) d[static_cast<std::size_t>(y) * W + x] = binary(x, y) != T(0) ? INF : 0.0;
    auto D = [&](int x, int y) -> double& { return d[static_cast<std::size_t>(y) * W + x]; };
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            double& c = D(x, y);
            if (x > 0) c = std::min(c, D(x - 1, y) + 1);
            if (y > 0) c = std::min(c, D(x, y - 1) + 1);
            if (chess && y > 0 && x > 0) c = std::min(c, D(x - 1, y - 1) + 1);
            if (chess && y > 0 && x + 1 < W) c = std::min(c, D(x + 1, y - 1) + 1);
        }
    for (int y = H - 1; y >= 0; --y)
        for (int x = W - 1; x >= 0; --x) {
            double& c = D(x, y);
            if (x + 1 < W) c = std::min(c, D(x + 1, y) + 1);
            if (y + 1 < H) c = std::min(c, D(x, y + 1) + 1);
            if (chess && y + 1 < H && x + 1 < W) c = std::min(c, D(x + 1, y + 1) + 1);
            if (chess && y + 1 < H && x > 0) c = std::min(c, D(x - 1, y + 1) + 1);
        }
    for (std::size_t i = 0; i < d.size(); ++i)
        out.data()[i] = d[i] >= INF ? std::numeric_limits<float>::infinity() : static_cast<float>(d[i]);
    return out;
}

// ---------------------------------------------------------------------------
// Flood fill
// ---------------------------------------------------------------------------

/// Fills the region connected to `seed` whose pixels differ from the seed color by at most
/// `tolerance` in every channel, painting it with `newColor`. Returns the number of pixels
/// filled. Optionally writes the filled region into `mask` (0/255, resized as needed).
template <class T>
std::size_t floodFill(Image<T>& img, Point seed, const std::vector<double>& newColor, double tolerance = 0.0,
                      int connectivity = 4, ImageU8* mask = nullptr) {
    detail::require(img.inside(seed.x, seed.y), "floodFill: seed outside the image");
    detail::require(static_cast<int>(newColor.size()) == img.channels(), "floodFill: color size != channels");
    detail::require(connectivity == 4 || connectivity == 8, "floodFill: connectivity must be 4 or 8");
    const int W = img.width(), H = img.height(), cn = img.channels();
    std::vector<double> ref(static_cast<std::size_t>(cn));
    for (int c = 0; c < cn; ++c) ref[c] = img(seed.x, seed.y, c);
    ImageU8 visited(W, H, 1, 0);
    auto matches = [&](int x, int y) {
        for (int c = 0; c < cn; ++c)
            if (std::fabs(static_cast<double>(img(x, y, c)) - ref[c]) > tolerance) return false;
        return true;
    };
    std::deque<Point> q{seed};
    visited(seed.x, seed.y) = 255;
    std::vector<Point> region;
    while (!q.empty()) {
        const Point p = q.front();
        q.pop_front();
        region.push_back(p);
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if ((dx == 0 && dy == 0) || (connectivity == 4 && dx != 0 && dy != 0)) continue;
                const int nx = p.x + dx, ny = p.y + dy;
                if (nx < 0 || ny < 0 || nx >= W || ny >= H || visited(nx, ny) || !matches(nx, ny)) continue;
                visited(nx, ny) = 255;
                q.push_back({nx, ny});
            }
    }
    for (const Point& p : region)
        for (int c = 0; c < cn; ++c) img(p.x, p.y, c) = saturate_cast<T>(newColor[c]);
    if (mask) *mask = std::move(visited);
    return region.size();
}

}  // namespace ipf
