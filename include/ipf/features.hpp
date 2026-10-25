// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf features: edge detection (Canny), corners (Harris, Shi-Tomasi), Hough transforms
// (lines, circles), template matching, local binary patterns, peak finding and
// bounding-box non-maximum suppression.
#pragma once

#include <deque>

#include "color.hpp"
#include "core.hpp"
#include "fft.hpp"
#include "filter.hpp"
#include "morphology.hpp"

namespace ipf {

// ---------------------------------------------------------------------------
// Canny edge detector
// ---------------------------------------------------------------------------

/// Canny edge detection (OpenCV semantics): Sobel gradients with aperture ksize,
/// non-maximum suppression and hysteresis between `lowThreshold` and `highThreshold`
/// (in gradient-magnitude units; L1 magnitude unless l2Gradient). Color images use the
/// channel with the strongest gradient at each pixel. Returns a 0/255 edge mask.
template <class T>
ImageU8 canny(const Image<T>& src, double lowThreshold, double highThreshold, int ksize = 3, bool l2Gradient = false) {
    detail::require(lowThreshold <= highThreshold, "canny: lowThreshold must be <= highThreshold");
    const int W = src.width(), H = src.height(), cn = src.channels();
    const ImageF gx = sobel(src, 1, 0, ksize, 1.0, BorderMode::Replicate);
    const ImageF gy = sobel(src, 0, 1, ksize, 1.0, BorderMode::Replicate);
    ImageF mag(W, H, 1), dx(W, H, 1), dy(W, H, 1);
    parallelFor(0, H, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x) {
                float best = -1;
                for (int c = 0; c < cn; ++c) {
                    const float a = gx(x, y, c), b = gy(x, y, c);
                    const float m = l2Gradient ? std::sqrt(a * a + b * b) : std::fabs(a) + std::fabs(b);
                    if (m > best) {
                        best = m;
                        dx(x, y) = a;
                        dy(x, y) = b;
                    }
                }
                mag(x, y) = best;
            }
    }, static_cast<double>(W) * cn * 4);
    // 0 = not an edge, 1 = weak candidate, 2 = strong edge
    ImageU8 state(W, H, 1, 0);
    const float tan22 = 0.41421356f;  // tan(22.5 deg)
    auto M = [&](int x, int y) { return (x < 0 || y < 0 || x >= W || y >= H) ? 0.f : mag(x, y); };
    parallelFor(0, H, [&](int y0, int y1) {  // non-maximum suppression (each pixel independent)
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x) {
                const float m = mag(x, y);
                if (m <= lowThreshold) continue;
                const float ax = std::fabs(dx(x, y)), ay = std::fabs(dy(x, y));
                bool isMax;
                if (ay <= ax * tan22) {
                    isMax = m > M(x - 1, y) && m >= M(x + 1, y);
                } else if (ay * tan22 >= ax) {
                    isMax = m > M(x, y - 1) && m >= M(x, y + 1);
                } else {
                    const int s = (dx(x, y) < 0) != (dy(x, y) < 0) ? -1 : 1;
                    isMax = m > M(x - s, y - 1) && m > M(x + s, y + 1);
                }
                if (isMax) state(x, y) = m > highThreshold ? 2 : 1;
            }
    }, static_cast<double>(W) * 6);
    std::deque<Point> q;  // hysteresis (the result does not depend on the visiting order)
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (state(x, y) == 2) q.push_back({x, y});
    while (!q.empty()) {  // hysteresis: grow strong edges through weak candidates
        const Point p = q.front();
        q.pop_front();
        for (int j = -1; j <= 1; ++j)
            for (int i = -1; i <= 1; ++i) {
                const int nx = p.x + i, ny = p.y + j;
                if (nx < 0 || ny < 0 || nx >= W || ny >= H || state(nx, ny) != 1) continue;
                state(nx, ny) = 2;
                q.push_back({nx, ny});
            }
    }
    for (auto& v : state) v = v == 2 ? 255 : 0;
    return state;
}

/// Canny with automatic thresholds (MATLAB's heuristic): Gaussian blur, high threshold at the
/// 70th percentile of the gradient magnitude, low threshold = 0.4 * high.
template <class T>
ImageU8 cannyAuto(const Image<T>& src, double blurSigma = 1.4) {
    const ImageF smooth = gaussianBlur<float>(toGray(src), blurSigma);
    const ImageF g = gradientMagnitude(smooth);
    std::vector<float> v(g.begin(), g.end());
    if (v.empty()) return ImageU8(src.width(), src.height(), 1, 0);
    const std::size_t k = v.size() * 7 / 10;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
    const double high = std::max(1e-6, static_cast<double>(v[k]));
    return canny(smooth, 0.4 * high, high, 3, true);
}

// ---------------------------------------------------------------------------
// Corners
// ---------------------------------------------------------------------------

namespace detail {
/// Structure tensor components (sums of dx^2, dx*dy, dy^2 over blockSize windows), OpenCV scaling.
template <class T>
void structureTensor(const Image<T>& src, int blockSize, int ksize, ImageF& a, ImageF& b, ImageF& c) {
    require(src.channels() == 1, "corner detection: expected a single-channel image");
    double scale = static_cast<double>(1 << ((ksize > 0 ? ksize : 3) - 1)) * blockSize;
    if (!std::is_floating_point<T>::value) scale *= PixelTraits<T>::max();
    scale = 1.0 / scale;
    const ImageF dx = sobel(src, 1, 0, ksize, scale), dy = sobel(src, 0, 1, ksize, scale);
    ImageF xx(src.width(), src.height(), 1), xy(src.width(), src.height(), 1), yy(src.width(), src.height(), 1);
    for (std::size_t i = 0; i < xx.sampleCount(); ++i) {
        xx.data()[i] = dx.data()[i] * dx.data()[i];
        xy.data()[i] = dx.data()[i] * dy.data()[i];
        yy.data()[i] = dy.data()[i] * dy.data()[i];
    }
    a = boxFilter(xx, blockSize, blockSize, false);
    b = boxFilter(xy, blockSize, blockSize, false);
    c = boxFilter(yy, blockSize, blockSize, false);
}
}  // namespace detail

/// Harris corner response R = det(M) - k * trace(M)^2 of the structure tensor M.
template <class T>
ImageF cornerHarris(const Image<T>& src, int blockSize = 3, int ksize = 3, double k = 0.04) {
    ImageF a, b, c;
    detail::structureTensor(src, blockSize, ksize, a, b, c);
    ImageF r(src.width(), src.height(), 1);
    for (std::size_t i = 0; i < r.sampleCount(); ++i) {
        const double A = a.data()[i], B = b.data()[i], C = c.data()[i];
        r.data()[i] = static_cast<float>(A * C - B * B - k * (A + C) * (A + C));
    }
    return r;
}

/// Minimum eigenvalue of the structure tensor (Shi-Tomasi corner strength).
template <class T>
ImageF cornerMinEigenVal(const Image<T>& src, int blockSize = 3, int ksize = 3) {
    ImageF a, b, c;
    detail::structureTensor(src, blockSize, ksize, a, b, c);
    ImageF r(src.width(), src.height(), 1);
    for (std::size_t i = 0; i < r.sampleCount(); ++i) {
        const double A = a.data()[i] * 0.5, B = b.data()[i], C = c.data()[i] * 0.5;
        r.data()[i] = static_cast<float>((A + C) - std::sqrt((A - C) * (A - C) + B * B));
    }
    return r;
}

struct Peak {
    Point pos;
    float value = 0;
};

/// Finds local maxima (3x3) above `threshold`, strongest first, keeping peaks at least
/// `minDistance` apart; at most maxPeaks results (0 = unlimited).
inline std::vector<Peak> findPeaks(const ImageF& response, float threshold, double minDistance = 1.0, int maxPeaks = 0) {
    detail::require(response.channels() == 1, "findPeaks: expected a single-channel image");
    const ImageF dil = dilate(response, getStructuringElement(MorphShape::Rect, 3));
    std::vector<Peak> cand;
    for (int y = 0; y < response.height(); ++y)
        for (int x = 0; x < response.width(); ++x) {
            const float v = response(x, y);
            if (v > threshold && v >= dil(x, y)) cand.push_back({{x, y}, v});
        }
    std::stable_sort(cand.begin(), cand.end(), [](const Peak& a, const Peak& b) { return a.value > b.value; });
    std::vector<Peak> out;
    const double md2 = minDistance * minDistance;
    for (const Peak& p : cand) {
        bool ok = true;
        for (const Peak& o : out) {
            const double ddx = p.pos.x - o.pos.x, ddy = p.pos.y - o.pos.y;
            if (ddx * ddx + ddy * ddy < md2) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;
        out.push_back(p);
        if (maxPeaks > 0 && static_cast<int>(out.size()) >= maxPeaks) break;
    }
    return out;
}

/// Shi-Tomasi / Harris "good features to track" (OpenCV semantics): corners whose strength
/// exceeds qualityLevel * strongest corner, at least minDistance apart, strongest first.
template <class T>
std::vector<Point> goodFeaturesToTrack(const Image<T>& src, int maxCorners, double qualityLevel = 0.01,
                                       double minDistance = 10.0, int blockSize = 3, bool useHarris = false,
                                       double k = 0.04) {
    const ImageF r = useHarris ? cornerHarris(toGray(src), blockSize, 3, k) : cornerMinEigenVal(toGray(src), blockSize, 3);
    const double maxVal = minMaxLoc(r).maxVal;
    if (maxVal <= 0) return {};
    std::vector<Point> pts;
    for (const Peak& p : findPeaks(r, static_cast<float>(maxVal * qualityLevel), minDistance, maxCorners))
        pts.push_back(p.pos);
    return pts;
}

// ---------------------------------------------------------------------------
// Hough transforms
// ---------------------------------------------------------------------------

struct HoughLine {
    double rho = 0;    ///< signed distance from the origin (pixels)
    double theta = 0;  ///< angle of the line normal in radians [0, pi)
    int votes = 0;
};

/// Standard Hough line transform on a binary edge image: lines x cos(theta) + y sin(theta) = rho
/// with at least `threshold` votes, strongest first.
template <class T>
std::vector<HoughLine> houghLines(const Image<T>& edges, double rhoRes, double thetaRes, int threshold, int maxLines = 0) {
    detail::require(edges.channels() == 1, "houghLines: expected a single-channel edge image");
    detail::require(rhoRes > 0 && thetaRes > 0, "houghLines: resolutions must be positive");
    const int W = edges.width(), H = edges.height();
    const int nTheta = static_cast<int>(std::round(detail::kPi / thetaRes));
    const double maxRho = std::hypot(W, H);
    const int nRho = static_cast<int>(std::round(2 * maxRho / rhoRes)) + 1;
    std::vector<int> acc(static_cast<std::size_t>(nTheta + 2) * (nRho + 2), 0);
    std::vector<double> cs(static_cast<std::size_t>(nTheta)), sn(static_cast<std::size_t>(nTheta));
    for (int t = 0; t < nTheta; ++t) {
        cs[t] = std::cos(t * thetaRes) / rhoRes;
        sn[t] = std::sin(t * thetaRes) / rhoRes;
    }
    auto A = [&](int t, int r) -> int& { return acc[static_cast<std::size_t>(t + 1) * (nRho + 2) + r + 1]; };
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            if (edges(x, y) == T(0)) continue;
            for (int t = 0; t < nTheta; ++t) {
                const int r = static_cast<int>(std::lround(x * cs[t] + y * sn[t] + (nRho - 1) / 2.0));
                if (r >= 0 && r < nRho) ++A(t, r);
            }
        }
    std::vector<HoughLine> lines;
    for (int t = 0; t < nTheta; ++t)
        for (int r = 0; r < nRho; ++r) {
            const int v = A(t, r);
            if (v >= threshold && v > A(t, r - 1) && v >= A(t, r + 1) && v > A(t - 1, r) && v >= A(t + 1, r))
                lines.push_back({(r - (nRho - 1) / 2.0) * rhoRes, t * thetaRes, v});
        }
    std::stable_sort(lines.begin(), lines.end(), [](const HoughLine& a, const HoughLine& b) { return a.votes > b.votes; });
    if (maxLines > 0 && static_cast<int>(lines.size()) > maxLines) lines.resize(static_cast<std::size_t>(maxLines));
    return lines;
}

struct LineSegment {
    Point p0, p1;
    double length() const { return std::hypot(p1.x - p0.x, p1.y - p0.y); }
};

/// Probabilistic Hough transform (Matas et al., as OpenCV's HoughLinesP): returns line
/// segments of at least minLineLength pixels whose points have at least `threshold` votes,
/// bridging gaps of up to maxLineGap pixels. Edge points are visited in a pseudo-random
/// order determined by `seed`, so results are reproducible.
template <class T>
std::vector<LineSegment> houghLinesP(const Image<T>& edges, double rhoRes, double thetaRes, int threshold,
                                     double minLineLength = 0, double maxLineGap = 0, int maxLines = 0,
                                     unsigned seed = 12345) {
    detail::require(edges.channels() == 1, "houghLinesP: expected a single-channel edge image");
    detail::require(rhoRes > 0 && thetaRes > 0, "houghLinesP: resolutions must be positive");
    const int W = edges.width(), H = edges.height();
    const int numAngle = static_cast<int>(std::lround(detail::kPi / thetaRes));
    const int numRho = static_cast<int>(std::lround(((W + H) * 2 + 1) / rhoRes));
    const double irho = 1.0 / rhoRes;
    std::vector<double> trig(static_cast<std::size_t>(numAngle) * 2);
    for (int n = 0; n < numAngle; ++n) {
        trig[2 * n] = std::cos(n * thetaRes) * irho;
        trig[2 * n + 1] = std::sin(n * thetaRes) * irho;
    }
    std::vector<int> acc(static_cast<std::size_t>(numAngle) * numRho, 0);
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(W) * H, 0);
    std::vector<Point> pts;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (edges(x, y) != T(0)) {
                mask[static_cast<std::size_t>(y) * W + x] = 1;
                pts.push_back({x, y});
            }
    std::uint64_t state = seed ? seed : 1;  // xorshift64* for a portable, reproducible shuffle
    auto rnd = [&state]() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 2685821657736338717ULL;
    };
    for (std::size_t i = pts.size(); i > 1; --i) std::swap(pts[i - 1], pts[static_cast<std::size_t>(rnd() % i)]);
    auto rhoIndex = [&](int n, int x, int y) {
        return static_cast<int>(std::lround(x * trig[2 * n] + y * trig[2 * n + 1])) + (numRho - 1) / 2;
    };
    auto asr = [](long long v, int s) { return v >= 0 ? v >> s : -((-v - 1) >> s) - 1; };  // floor(v / 2^s)
    const int shift = 16;
    const int lineGap = static_cast<int>(maxLineGap), lineLength = static_cast<int>(minLineLength);
    std::vector<LineSegment> lines;
    for (const Point& pt : pts) {
        if (!mask[static_cast<std::size_t>(pt.y) * W + pt.x]) continue;
        int maxVal = threshold - 1, maxN = 0;
        for (int n = 0; n < numAngle; ++n) {
            const int v = ++acc[static_cast<std::size_t>(n) * numRho + rhoIndex(n, pt.x, pt.y)];
            if (maxVal < v) {
                maxVal = v;
                maxN = n;
            }
        }
        if (maxVal < threshold) continue;
        // walk from the point along the strongest line in both directions (fixed point)
        const double a = -trig[2 * maxN + 1], b = trig[2 * maxN];
        long long x0 = pt.x, y0 = pt.y, dx0, dy0;
        const bool xflag = std::fabs(a) > std::fabs(b);
        if (xflag) {
            dx0 = a > 0 ? 1 : -1;
            dy0 = std::llround(b * (1 << shift) / std::fabs(a));
            y0 = (y0 << shift) + (1 << (shift - 1));
        } else {
            dy0 = b > 0 ? 1 : -1;
            dx0 = std::llround(a * (1 << shift) / std::fabs(b));
            x0 = (x0 << shift) + (1 << (shift - 1));
        }
        Point lineEnd[2] = {pt, pt};
        auto pixelAt = [&](long long x, long long y, int& j1, int& i1) {
            if (xflag) {
                j1 = static_cast<int>(x);
                i1 = static_cast<int>(asr(y, shift));
            } else {
                j1 = static_cast<int>(asr(x, shift));
                i1 = static_cast<int>(y);
            }
        };
        for (int k = 0; k < 2; ++k) {
            const long long dx = k ? -dx0 : dx0, dy = k ? -dy0 : dy0;
            int gap = 0;
            for (long long x = x0, y = y0;; x += dx, y += dy) {
                int j1, i1;
                pixelAt(x, y, j1, i1);
                if (j1 < 0 || j1 >= W || i1 < 0 || i1 >= H) break;
                if (mask[static_cast<std::size_t>(i1) * W + j1]) {
                    gap = 0;
                    lineEnd[k] = {j1, i1};
                } else if (++gap > lineGap) {
                    break;
                }
            }
        }
        const bool good = std::abs(lineEnd[1].x - lineEnd[0].x) >= lineLength ||
                          std::abs(lineEnd[1].y - lineEnd[0].y) >= lineLength;
        // second walk: consume the points (and remove their votes if the segment is accepted)
        for (int k = 0; k < 2; ++k) {
            const long long dx = k ? -dx0 : dx0, dy = k ? -dy0 : dy0;
            for (long long x = x0, y = y0;; x += dx, y += dy) {
                int j1, i1;
                pixelAt(x, y, j1, i1);
                std::uint8_t& m = mask[static_cast<std::size_t>(i1) * W + j1];
                if (m) {
                    if (good)
                        for (int n = 0; n < numAngle; ++n) --acc[static_cast<std::size_t>(n) * numRho + rhoIndex(n, j1, i1)];
                    m = 0;
                }
                if (i1 == lineEnd[k].y && j1 == lineEnd[k].x) break;
            }
        }
        if (good) {
            lines.push_back({lineEnd[0], lineEnd[1]});
            if (maxLines > 0 && static_cast<int>(lines.size()) >= maxLines) break;
        }
    }
    return lines;
}

struct HoughCircle {
    PointF center;
    double radius = 0;
    int votes = 0;  ///< number of edge pixels supporting the circle
};

/// Gradient-based Hough circle detection on a grayscale image.
///  1. Canny edges (cannyHigh / 2, cannyHigh) vote for centers along their gradient direction.
///  2. Accumulator peaks with at least `centerVotes` votes (and `minDist` apart) become candidates.
///  3. Each candidate gets the radius in [minRadius, maxRadius] best supported by edge pixels whose
///     gradient points radially, then center and radius are refined by a least-squares circle fit.
///  4. A circle is accepted when its supporting edge pixels cover at least `minCoverage` of the
///     circumference (measured in 10-degree sectors), which rejects straight edges and corners.
/// Results are sorted by support (number of supporting edge pixels).
template <class T>
std::vector<HoughCircle> houghCircles(const Image<T>& gray, int minRadius, int maxRadius, double cannyHigh = 100,
                                      int centerVotes = 20, double minDist = 10, double minCoverage = 0.6) {
    detail::require(gray.channels() == 1, "houghCircles: expected a single-channel image");
    detail::require(minRadius > 0 && maxRadius >= minRadius, "houghCircles: invalid radius range");
    const int W = gray.width(), H = gray.height();
    const ImageU8 edges = canny(gray, cannyHigh / 2, cannyHigh);
    const ImageF smoothGray = gaussianBlur<float>(gray, 1.0);  // smoother normals than raw Sobel
    const ImageF dx = sobel(smoothGray, 1, 0), dy = sobel(smoothGray, 0, 1);
    struct EdgePt {
        double x, y, nx, ny;  // position and unit gradient
    };
    std::vector<EdgePt> pts;
    ImageF acc(W, H, 1, 0.f);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            if (!edges(x, y)) continue;
            const double gx = dx(x, y), gy = dy(x, y), g = std::hypot(gx, gy);
            if (g < 1e-9) continue;
            pts.push_back({static_cast<double>(x), static_cast<double>(y), gx / g, gy / g});
            for (int sgn = -1; sgn <= 1; sgn += 2)
                for (int r = minRadius; r <= maxRadius; ++r) {
                    const int cx = static_cast<int>(std::lround(x + sgn * r * gx / g));
                    const int cy = static_cast<int>(std::lround(y + sgn * r * gy / g));
                    if (cx >= 0 && cy >= 0 && cx < W && cy < H) acc(cx, cy) += 1.f;
                }
        }
    // An edge pixel supports a circle if it lies near it and its gradient is radial.
    const double kAlign = 0.92;  // |cos| of the angle between gradient and radius (~23 degrees)
    auto radial = [](const EdgePt& e, PointF c, double& dist) {
        const double ux = e.x - c.x, uy = e.y - c.y;
        dist = std::hypot(ux, uy);
        return dist > 1e-9 ? std::fabs(ux * e.nx + uy * e.ny) / dist : 0.0;
    };
    const ImageF smooth = boxFilter(acc, 3, 3, false, BorderMode::Constant);
    std::vector<HoughCircle> circles;
    for (const Peak& p : findPeaks(smooth, static_cast<float>(centerVotes) - 0.5f, minDist)) {
        // sub-pixel center: vote-weighted centroid of the accumulator around the peak
        double sw = 0, scx = 0, scy = 0;
        for (int j = -2; j <= 2; ++j)
            for (int i = -2; i <= 2; ++i) {
                const int x = p.pos.x + i, y = p.pos.y + j;
                if (x < 0 || y < 0 || x >= W || y >= H) continue;
                const double v = acc(x, y);
                sw += v;
                scx += v * x;
                scy += v * y;
            }
        PointF c = sw > 0 ? PointF(scx / sw, scy / sw) : PointF(p.pos);
        // radius with the best radially-aligned support per unit circumference
        std::vector<int> hist(static_cast<std::size_t>(maxRadius + 2), 0);
        for (const EdgePt& e : pts) {
            double d;
            if (radial(e, c, d) < kAlign) continue;
            const int r = static_cast<int>(std::lround(d));
            if (r >= minRadius && r <= maxRadius) ++hist[r];
        }
        int bestR = 0;
        double bestScore = 0;
        for (int r = minRadius; r <= maxRadius; ++r) {
            const double score = (hist[r] + 0.5 * (hist[r - 1] + hist[r + 1])) / (2 * detail::kPi * r);
            if (score > bestScore) {
                bestScore = score;
                bestR = r;
            }
        }
        if (!bestR) continue;
        // Refine with an algebraic least-squares circle fit (Kasa) on the supporting pixels.
        double radius = bestR;
        for (int iter = 0; iter < 2; ++iter) {
            double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sz = 0, sxz = 0, syz = 0, n = 0;
            for (const EdgePt& e : pts) {
                double d;
                if (radial(e, c, d) < kAlign || std::fabs(d - radius) > 2.0) continue;
                const double x = e.x - c.x, y = e.y - c.y, z = x * x + y * y;  // centered for conditioning
                sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; sz += z; sxz += x * z; syz += y * z;
                n += 1;
            }
            if (n < 3) break;
            // Solve [sxx sxy sx; sxy syy sy; sx sy n] [D E F]^T = -[sxz syz sz]^T (Cramer's rule)
            const double m[9] = {sxx, sxy, sx, sxy, syy, sy, sx, sy, n}, rhs[3] = {-sxz, -syz, -sz};
            auto det3 = [](const double* t) {
                return t[0] * (t[4] * t[8] - t[5] * t[7]) - t[1] * (t[3] * t[8] - t[5] * t[6]) +
                       t[2] * (t[3] * t[7] - t[4] * t[6]);
            };
            const double det = det3(m);
            if (std::fabs(det) < 1e-12) break;
            double sol[3];
            for (int col = 0; col < 3; ++col) {
                double t[9];
                std::copy(m, m + 9, t);
                for (int r = 0; r < 3; ++r) t[r * 3 + col] = rhs[r];
                sol[col] = det3(t) / det;
            }
            const double ox = -sol[0] / 2, oy = -sol[1] / 2, r2 = ox * ox + oy * oy - sol[2];
            if (r2 <= 0) break;
            c = PointF(c.x + ox, c.y + oy);
            radius = std::sqrt(r2);
        }
        if (radius < minRadius - 1 || radius > maxRadius + 1) continue;
        // angular coverage of the supporting pixels (36 sectors)
        bool sector[36] = {};
        int support = 0;
        for (const EdgePt& e : pts) {
            double d;
            if (radial(e, c, d) < kAlign || std::fabs(d - radius) > 1.5) continue;
            ++support;
            double a = std::atan2(e.y - c.y, e.x - c.x) + detail::kPi;
            sector[std::min(35, static_cast<int>(a / (2 * detail::kPi) * 36))] = true;
        }
        const int covered = static_cast<int>(std::count(std::begin(sector), std::end(sector), true));
        if (covered < minCoverage * 36) continue;
        circles.push_back({c, radius, support});
    }
    std::stable_sort(circles.begin(), circles.end(), [](const HoughCircle& a, const HoughCircle& b) { return a.votes > b.votes; });
    // several accumulator peaks may refine to the same circle: keep the best-supported one
    std::vector<HoughCircle> unique;
    for (const HoughCircle& cc : circles) {
        bool dup = false;
        for (const HoughCircle& u : unique) dup |= std::hypot(cc.center.x - u.center.x, cc.center.y - u.center.y) < minDist;
        if (!dup) unique.push_back(cc);
    }
    return unique;
}

// ---------------------------------------------------------------------------
// Template matching
// ---------------------------------------------------------------------------

enum class MatchMethod { SqDiff, SqDiffNormed, CCorr, CCorrNormed, CCoeff, CCoeffNormed };

/// Slides `templ` over `img` and returns a (W - w + 1) x (H - h + 1) score map using
/// OpenCV's formulas (multi-channel images sum over channels). For SqDiff* the best match is
/// the minimum, otherwise the maximum (use minMaxLoc on the result).
template <class T>
ImageF matchTemplate(const Image<T>& img, const Image<T>& templ, MatchMethod method) {
    detail::require(img.channels() == templ.channels(), "matchTemplate: channel counts differ");
    detail::require(templ.width() <= img.width() && templ.height() <= img.height() && !templ.empty(),
                    "matchTemplate: template must be non-empty and not larger than the image");
    const int W = img.width() - templ.width() + 1, H = img.height() - templ.height() + 1;
    const int tw = templ.width(), th = templ.height(), cn = img.channels();
    const double n = static_cast<double>(tw) * th;
    std::vector<double> tMean(static_cast<std::size_t>(cn), 0.0);
    double tSq = 0, tVar = 0;
    for (int c = 0; c < cn; ++c) {
        double s = 0, s2 = 0;
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < tw; ++x) {
                const double v = templ(x, y, c);
                s += v;
                s2 += v * v;
            }
        tMean[c] = s / n;
        tSq += s2;
        tVar += s2 - s * s / n;
    }
    const ImageD I = integralImage(img), I2 = integralImage(img, true);
    // Cross term sum T * I per channel: direct for small templates, FFT for large ones
    // (chosen by estimated operation count; both give the same values up to rounding).
    const double directCost = static_cast<double>(W) * H * tw * th;
    const double fftSize = static_cast<double>(detail::nextPow2(static_cast<std::size_t>(img.width()))) *
                           static_cast<double>(detail::nextPow2(static_cast<std::size_t>(img.height())));
    const double fftCost = fftSize * std::log2(std::max(2.0, fftSize)) * 12.0;
    ImageD crossTerm;
    if (directCost > fftCost) crossTerm = detail::correlateFFT(img, templ, W, H);
    ImageF out(W, H, 1);
    parallelFor(0, H, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x) {
                double cross = 0, wSum2 = 0, wVar = 0, crossC = 0;
                for (int c = 0; c < cn; ++c) {
                    double cc = 0;
                    if (!crossTerm.empty()) {
                        cc = crossTerm(x, y, c);
                    } else {
                        for (int j = 0; j < th; ++j) {
                            const T* ir = img.row(y + j);
                            const T* tr = templ.row(j);
                            for (int i = 0; i < tw; ++i)
                                cc += static_cast<double>(ir[(x + i) * cn + c]) * static_cast<double>(tr[i * cn + c]);
                        }
                    }
                    const Rect r(x, y, tw, th);
                    const double s = integralSum(I, r, c), s2 = integralSum(I2, r, c);
                    cross += cc;
                    crossC += cc - tMean[c] * s;  // sum (T - meanT) * I == sum (T - meanT)(I - meanI)
                    wSum2 += s2;
                    wVar += s2 - s * s / n;
                }
                double v = 0;
                switch (method) {
                    case MatchMethod::SqDiff: v = tSq - 2 * cross + wSum2; break;
                    case MatchMethod::SqDiffNormed: {
                        const double d = std::sqrt(tSq * wSum2);
                        v = d > 1e-12 ? (tSq - 2 * cross + wSum2) / d : 1.0;
                        break;
                    }
                    case MatchMethod::CCorr: v = cross; break;
                    case MatchMethod::CCorrNormed: {
                        const double d = std::sqrt(tSq * wSum2);
                        v = d > 1e-12 ? cross / d : 0.0;
                        break;
                    }
                    case MatchMethod::CCoeff: v = crossC; break;
                    default: {
                        const double d = std::sqrt(std::max(0.0, tVar) * std::max(0.0, wVar));
                        v = d > 1e-9 * (tSq + wSum2 + 1) ? crossC / d : 0.0;
                        break;
                    }
                }
                if (method == MatchMethod::SqDiff || method == MatchMethod::SqDiffNormed) v = std::max(0.0, v);
                out(x, y) = static_cast<float>(v);
            }
    }, crossTerm.empty() ? static_cast<double>(tw) * th * cn * W : 20.0 * W);
    return out;
}

/// Convenience: position (top-left) and score of the best match.
struct MatchResult {
    Point location;
    double score = 0;
};
template <class T>
MatchResult findTemplate(const Image<T>& img, const Image<T>& templ, MatchMethod method = MatchMethod::CCoeffNormed) {
    const ImageF r = matchTemplate(img, templ, method);
    const MinMaxResult mm = minMaxLoc(r);
    const bool lowBest = method == MatchMethod::SqDiff || method == MatchMethod::SqDiffNormed;
    return lowBest ? MatchResult{mm.minLoc, mm.minVal} : MatchResult{mm.maxLoc, mm.maxVal};
}

// ---------------------------------------------------------------------------
// Texture and detection helpers
// ---------------------------------------------------------------------------

/// Basic 8-neighbour local binary pattern: bit k is set when neighbour k >= center, with
/// neighbours ordered clockwise from the top-left pixel (bit 7) to the left pixel (bit 0).
/// Borders replicate edge pixels.
template <class T>
ImageU8 localBinaryPattern(const Image<T>& gray) {
    detail::require(gray.channels() == 1, "localBinaryPattern: expected a single-channel image");
    static const int ox[8] = {-1, 0, 1, 1, 1, 0, -1, -1}, oy[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
    ImageU8 out(gray.width(), gray.height(), 1);
    for (int y = 0; y < gray.height(); ++y)
        for (int x = 0; x < gray.width(); ++x) {
            const T c = gray(x, y);
            int code = 0;
            for (int k = 0; k < 8; ++k)
                if (gray.atBorder(x + ox[k], y + oy[k], 0, BorderMode::Replicate) >= c) code |= 1 << (7 - k);
            out(x, y) = static_cast<std::uint8_t>(code);
        }
    return out;
}

/// Intersection over union of two rectangles.
inline double iou(const Rect& a, const Rect& b) {
    const double inter = static_cast<double>((a & b).area());
    const double uni = static_cast<double>(a.area() + b.area()) - inter;
    return uni > 0 ? inter / uni : 0.0;
}

/// Greedy non-maximum suppression of scored boxes: returns indices of kept boxes (highest
/// score first), dropping boxes whose IoU with a kept box exceeds iouThreshold.
inline std::vector<int> nmsBoxes(const std::vector<Rect>& boxes, const std::vector<double>& scores, double iouThreshold,
                                 double scoreThreshold = -1e300) {
    detail::require(boxes.size() == scores.size(), "nmsBoxes: boxes and scores sizes differ");
    std::vector<int> order;
    for (int i = 0; i < static_cast<int>(boxes.size()); ++i)
        if (scores[i] > scoreThreshold) order.push_back(i);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    std::vector<int> keep;
    for (int i : order) {
        bool ok = true;
        for (int k : keep)
            if (iou(boxes[i], boxes[k]) > iouThreshold) {
                ok = false;
                break;
            }
        if (ok) keep.push_back(i);
    }
    return keep;
}

}  // namespace ipf
