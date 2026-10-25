// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf filter: linear filtering (correlation, separable, box, Gaussian), non-linear
// smoothing (median, bilateral), derivatives (Sobel, Scharr, Laplacian, Prewitt),
// sharpening, integral images and adaptive thresholding.
//
// All filters work on any channel count (channels are filtered independently) and
// keep the image size. Pixels outside the image are synthesized with a BorderMode
// (default Reflect101, like OpenCV). Results are rounded/saturated to the output type.
// Filters taking a template parameter TOut can produce another type, e.g.
//   ImageF smooth = ipf::gaussianBlur<float>(img8, 2.0);
#pragma once

#include <memory>

#include "core.hpp"
#include "parallel.hpp"
#include "pixel_ops.hpp"

namespace ipf {

namespace detail {

template <class TOut, class T>
using OutT = std::conditional_t<std::is_void<TOut>::value, T, TOut>;

/// Source index for every padded position p in [0, len + before + after): borderIndex(p - before).
inline std::vector<int> borderTable(int len, int before, int after, BorderMode mode) {
    std::vector<int> t(static_cast<std::size_t>(len + before + after));
    for (int p = 0; p < static_cast<int>(t.size()); ++p) t[p] = borderIndex(p - before, len, mode);
    return t;
}

/// Separable correlation: horizontal kernel kx (anchor ax), then vertical ky (anchor ay).
/// Inner loops run over contiguous memory so the compiler can vectorize them; rows are
/// processed in parallel.
template <class TOut, class T>
Image<TOut> sepFilterImpl(const Image<T>& src, const std::vector<double>& kx, const std::vector<double>& ky, int ax,
                          int ay, BorderMode border, double delta, double borderValue) {
    require(!kx.empty() && !ky.empty(), "separable filter: empty kernel");
    const int W = src.width(), H = src.height(), cn = src.channels();
    Image<TOut> out(W, H, cn);
    if (src.empty()) return out;
    const int kw = static_cast<int>(kx.size()), kh = static_cast<int>(ky.size());
    const std::vector<int> xt = borderTable(W, ax, kw - 1 - ax, border);
    const std::vector<int> yt = borderTable(H, ay, kh - 1 - ay, border);
    const int PH = H + kh - 1;
    const std::size_t rowLen = static_cast<std::size_t>(W) * cn;
    // every element is written below, so skip value-initialization of the large buffer
    std::unique_ptr<double[]> tmp(new double[static_cast<std::size_t>(PH) * rowLen]);
    parallelFor(0, PH, [&](int y0, int y1) {
        std::vector<double> padded(static_cast<std::size_t>(W + kw - 1) * cn);
        for (int py = y0; py < y1; ++py) {
            const int sy = yt[py];
            for (int px = 0; px < W + kw - 1; ++px) {
                double* b = &padded[static_cast<std::size_t>(px) * cn];
                if (sy < 0 || xt[px] < 0) {
                    for (int c = 0; c < cn; ++c) b[c] = borderValue;
                } else {
                    const T* s0 = src.ptr(xt[px], sy);
                    for (int c = 0; c < cn; ++c) b[c] = static_cast<double>(s0[c]);
                }
            }
            const std::size_t len = rowLen;  // local copies keep the inner loop vectorizable
            double* t = &tmp[static_cast<std::size_t>(py) * len];
            std::fill(t, t + len, 0.0);
            for (int i = 0; i < kw; ++i) {
                const double k = kx[i];
                const double* b = &padded[static_cast<std::size_t>(i) * cn];
                for (std::size_t q = 0; q < len; ++q) t[q] += k * b[q];
            }
        }
    }, static_cast<double>(rowLen) * kw);
    parallelFor(0, H, [&](int y0, int y1) {
        const std::size_t len = rowLen;
        std::vector<double> accBuf(len);
        double* acc = accBuf.data();
        for (int y = y0; y < y1; ++y) {
            std::fill(acc, acc + len, delta);
            for (int j = 0; j < kh; ++j) {
                const double k = ky[j];
                const double* r = &tmp[static_cast<std::size_t>(y + j) * len];
                for (std::size_t q = 0; q < len; ++q) acc[q] += k * r[q];
            }
            TOut* d = out.row(y);
            for (std::size_t q = 0; q < len; ++q) d[q] = saturate_cast<TOut>(acc[q]);
        }
    }, static_cast<double>(rowLen) * kh);
    return out;
}

/// Derivative / smoothing kernels in the style of OpenCV's getDerivKernels.
inline std::vector<double> derivKernel(int order, int ksize) {
    if (ksize == 1) {  // no smoothing: plain central differences
        if (order == 0) return {1.0};
        if (order == 1) return {-1.0, 0.0, 1.0};
        if (order == 2) return {1.0, -2.0, 1.0};
        fail("derivKernel: order > 2 requires ksize >= 3");
    }
    require(ksize % 2 == 1 && ksize >= 3 && ksize <= 31, "derivKernel: ksize must be odd in [3, 31]");
    require(order >= 0 && order < ksize, "derivKernel: invalid derivative order");
    std::vector<double> k{1.0};
    auto conv = [&k](double a, double b) {
        std::vector<double> r(k.size() + 1, 0.0);
        for (std::size_t i = 0; i < k.size(); ++i) {
            r[i] += k[i] * a;
            r[i + 1] += k[i] * b;
        }
        k = r;
    };
    for (int i = 0; i < ksize - 1 - order; ++i) conv(1, 1);
    for (int i = 0; i < order; ++i) conv(-1, 1);
    return k;
}

}  // namespace detail

/// A convolution kernel is a single-channel float image.
using Kernel = ImageF;

/// Builds a kernel from row-major values.
inline Kernel makeKernel(int width, int height, std::initializer_list<float> values) {
    detail::require(static_cast<int>(values.size()) == width * height, "makeKernel: wrong number of values");
    return Kernel(width, height, 1, std::vector<float>(values));
}

/// 1D Gaussian weights (normalized). ksize <= 0 derives the size from sigma
/// (radius = round(3 sigma), as scipy's truncate=3); sigma <= 0 derives it from ksize (OpenCV rule).
inline std::vector<double> gaussianKernel1D(double sigma, int ksize = 0) {
    if (ksize <= 0) {
        detail::require(sigma > 0, "gaussianKernel1D: need sigma > 0 or ksize > 0");
        ksize = 2 * std::max(1, static_cast<int>(3.0 * sigma + 0.5)) + 1;
    }
    detail::require(ksize % 2 == 1, "gaussianKernel1D: ksize must be odd");
    if (sigma <= 0) sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8;
    std::vector<double> k(static_cast<std::size_t>(ksize));
    const int r = ksize / 2;
    double sum = 0;
    for (int i = 0; i < ksize; ++i) {
        k[i] = std::exp(-0.5 * (i - r) * (i - r) / (sigma * sigma));
        sum += k[i];
    }
    for (double& v : k) v /= sum;
    return k;
}

/// 2D Gaussian kernel (outer product of two 1D kernels).
inline Kernel gaussianKernel2D(double sigma, int ksize = 0) {
    const std::vector<double> g = gaussianKernel1D(sigma, ksize);
    const int n = static_cast<int>(g.size());
    Kernel k(n, n, 1);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) k(x, y) = static_cast<float>(g[y] * g[x]);
    return k;
}

/// Generic 2D correlation: out(x, y) = sum k(i, j) * src(x + i - ax, y + j - ay) + delta.
/// The anchor defaults to the kernel center. (For true convolution, flip the kernel.)
template <class TOut = void, class T>
Image<detail::OutT<TOut, T>> filter2D(const Image<T>& src, const Kernel& kernel,
                                      BorderMode border = BorderMode::Reflect101, double delta = 0.0,
                                      Point anchor = Point(-1, -1), double borderValue = 0.0) {
    using O = detail::OutT<TOut, T>;
    detail::require(kernel.channels() == 1 && !kernel.empty(), "filter2D: kernel must be a non-empty single-channel image");
    const int kw = kernel.width(), kh = kernel.height();
    const int ax = anchor.x < 0 ? kw / 2 : anchor.x, ay = anchor.y < 0 ? kh / 2 : anchor.y;
    detail::require(ax < kw && ay < kh, "filter2D: anchor outside kernel");
    const int W = src.width(), H = src.height(), cn = src.channels();
    Image<O> out(W, H, cn);
    if (src.empty()) return out;
    struct Tap {
        int dx, dy;
        double w;
    };
    std::vector<Tap> taps;
    for (int j = 0; j < kh; ++j)
        for (int i = 0; i < kw; ++i)
            if (kernel(i, j) != 0.f) taps.push_back({i, j, kernel(i, j)});
    const std::vector<int> xt = detail::borderTable(W, ax, kw - 1 - ax, border);
    const std::vector<int> yt = detail::borderTable(H, ay, kh - 1 - ay, border);
    parallelFor(0, H, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < cn; ++c) {
                    double s = delta;
                    for (const Tap& t : taps) {
                        const int sx = xt[x + t.dx], sy = yt[y + t.dy];
                        s += t.w * ((sx < 0 || sy < 0) ? borderValue : static_cast<double>(src(sx, sy, c)));
                    }
                    out(x, y, c) = saturate_cast<O>(s);
                }
    }, static_cast<double>(W) * cn * (taps.size() + 1));
    return out;
}

/// Separable correlation with a horizontal and a vertical 1D kernel (anchors at centers).
template <class TOut = void, class T>
Image<detail::OutT<TOut, T>> sepFilter2D(const Image<T>& src, const std::vector<double>& kx, const std::vector<double>& ky,
                                         BorderMode border = BorderMode::Reflect101, double delta = 0.0,
                                         double borderValue = 0.0) {
    return detail::sepFilterImpl<detail::OutT<TOut, T>>(src, kx, ky, static_cast<int>(kx.size()) / 2,
                                                        static_cast<int>(ky.size()) / 2, border, delta, borderValue);
}

/// Box filter (sum or mean over a kw x kh window) in O(1) per pixel using running sums.
template <class TOut = void, class T>
Image<detail::OutT<TOut, T>> boxFilter(const Image<T>& src, int kw, int kh, bool normalize = true,
                                       BorderMode border = BorderMode::Reflect101, double borderValue = 0.0) {
    using O = detail::OutT<TOut, T>;
    detail::require(kw > 0 && kh > 0, "boxFilter: kernel size must be positive");
    const int W = src.width(), H = src.height(), cn = src.channels();
    Image<O> out(W, H, cn);
    if (src.empty()) return out;
    const int ax = kw / 2, ay = kh / 2;
    const std::vector<int> xt = detail::borderTable(W, ax, kw - 1 - ax, border);
    const std::vector<int> yt = detail::borderTable(H, ay, kh - 1 - ay, border);
    const int PH = H + kh - 1;
    const std::size_t rowLen = static_cast<std::size_t>(W) * cn;
    std::vector<double> rows(static_cast<std::size_t>(PH) * rowLen);  // horizontal window sums
    parallelFor(0, PH, [&](int y0, int y1) {
        for (int py = y0; py < y1; ++py) {
            const int sy = yt[py];
            auto val = [&](int px, int c) {
                return (sy < 0 || xt[px] < 0) ? borderValue : static_cast<double>(src(xt[px], sy, c));
            };
            double* r = &rows[static_cast<std::size_t>(py) * rowLen];
            for (int c = 0; c < cn; ++c) {
                double s = 0;
                for (int i = 0; i < kw; ++i) s += val(i, c);
                r[c] = s;
                for (int x = 1; x < W; ++x) {
                    s += val(x + kw - 1, c) - val(x - 1, c);
                    r[static_cast<std::size_t>(x) * cn + c] = s;
                }
            }
        }
    }, static_cast<double>(rowLen) * 4);
    const double scale = normalize ? 1.0 / (static_cast<double>(kw) * kh) : 1.0;
    // vertical running sums: independent per column, so split the row into column blocks
    parallelFor(0, static_cast<int>(rowLen), [&](int q0, int q1) {
        std::vector<double> col(static_cast<std::size_t>(q1 - q0), 0.0);
        for (int j = 0; j < kh; ++j)
            for (int q = q0; q < q1; ++q) col[q - q0] += rows[static_cast<std::size_t>(j) * rowLen + q];
        for (int y = 0; y < H; ++y) {
            O* d = out.row(y);
            for (int q = q0; q < q1; ++q) d[q] = saturate_cast<O>(col[q - q0] * scale);
            if (y + 1 < H)
                for (int q = q0; q < q1; ++q)
                    col[q - q0] += rows[static_cast<std::size_t>(y + kh) * rowLen + q] - rows[static_cast<std::size_t>(y) * rowLen + q];
        }
    }, static_cast<double>(H) * 3);
    return out;
}

/// Mean filter with a square ksize x ksize window.
template <class TOut = void, class T>
Image<detail::OutT<TOut, T>> blur(const Image<T>& src, int ksize, BorderMode border = BorderMode::Reflect101) {
    return boxFilter<TOut>(src, ksize, ksize, true, border);
}

/// Gaussian blur. sigmaY = 0 uses sigmaX; ksize = 0 derives the window from sigma (radius round(3 sigma)).
template <class TOut = void, class T>
Image<detail::OutT<TOut, T>> gaussianBlur(const Image<T>& src, double sigmaX, double sigmaY = 0.0, int ksize = 0,
                                          BorderMode border = BorderMode::Reflect101) {
    if (sigmaY <= 0) sigmaY = sigmaX;
    return sepFilter2D<TOut>(src, gaussianKernel1D(sigmaX, ksize), gaussianKernel1D(sigmaY, ksize), border);
}

namespace detail {
/// Huang's sliding-histogram median for uint8 rows [y0, y1) with a running median: O(ksize) per pixel.
inline void medianHistogramRows(const ImageU8& src, ImageU8& out, int ksize, const std::vector<int>& xt,
                                const std::vector<int>& yt, int y0, int y1) {
    const int W = src.width(), cn = src.channels();
    const int mid = ksize * ksize / 2;  // rank of the median (0-based)
    for (int c = 0; c < cn; ++c)
        for (int y = y0; y < y1; ++y) {
            std::array<int, 256> h{};
            for (int j = 0; j < ksize; ++j)
                for (int i = 0; i < ksize; ++i) ++h[src(xt[i], yt[y + j], c)];
            // med = value of rank `mid`; lt = number of window samples below med
            int med = 0, lt = 0;
            while (lt + h[med] <= mid) lt += h[med++];
            out(0, y, c) = static_cast<std::uint8_t>(med);
            for (int x = 1; x < W; ++x) {
                for (int j = 0; j < ksize; ++j) {
                    const int a = src(xt[x - 1], yt[y + j], c), b = src(xt[x + ksize - 1], yt[y + j], c);
                    --h[a];
                    lt -= a < med;
                    ++h[b];
                    lt += b < med;
                }
                while (lt > mid) lt -= h[--med];
                while (lt + h[med] <= mid) lt += h[med++];
                out(x, y, c) = static_cast<std::uint8_t>(med);
            }
        }
}
}  // namespace detail

/// Median filter with a square ksize x ksize window (ksize odd). Default border replicates
/// edge pixels (OpenCV behavior).
template <class T>
Image<T> medianBlur(const Image<T>& src, int ksize, BorderMode border = BorderMode::Replicate) {
    detail::require(ksize >= 1 && ksize % 2 == 1, "medianBlur: ksize must be odd and positive");
    detail::require(border != BorderMode::Constant, "medianBlur: Constant border is not supported");
    const int W = src.width(), H = src.height(), cn = src.channels(), r = ksize / 2;
    if (src.empty() || ksize == 1) return src;
    Image<T> out(W, H, cn);
    const std::vector<int> xt = detail::borderTable(W, r, r, border);
    const std::vector<int> yt = detail::borderTable(H, r, r, border);
    const std::size_t n = static_cast<std::size_t>(ksize) * ksize, mid = n / 2;
    if constexpr (std::is_same<T, std::uint8_t>::value) {
        if (ksize >= 5) {
            parallelFor(0, H, [&](int y0, int y1) { detail::medianHistogramRows(src, out, ksize, xt, yt, y0, y1); },
                        static_cast<double>(W) * cn * ksize * 2);
            return out;
        }
    }
    parallelFor(0, H, [&](int y0, int y1) {
        std::vector<T> win(n);
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < cn; ++c) {
                    std::size_t k = 0;
                    for (int j = 0; j < ksize; ++j)
                        for (int i = 0; i < ksize; ++i) win[k++] = src(xt[x + i], yt[y + j], c);
                    std::nth_element(win.begin(), win.begin() + static_cast<std::ptrdiff_t>(mid), win.end());
                    out(x, y, c) = win[mid];
                }
    }, static_cast<double>(W) * cn * n * 3);
    return out;
}

/// Edge-preserving bilateral filter (OpenCV semantics): circular window of diameter d
/// (d <= 0 derives it from sigmaSpace), spatial Gaussian sigmaSpace and range Gaussian
/// sigmaColor applied to the L1 color distance between the pixels.
template <class T>
Image<T> bilateralFilter(const Image<T>& src, int d, double sigmaColor, double sigmaSpace,
                         BorderMode border = BorderMode::Reflect101) {
    detail::require(sigmaColor > 0 && sigmaSpace > 0, "bilateralFilter: sigmas must be positive");
    const int r = d <= 0 ? static_cast<int>(std::lround(sigmaSpace * 1.5)) : d / 2;
    const int W = src.width(), H = src.height(), cn = src.channels();
    Image<T> out(W, H, cn);
    if (src.empty()) return out;
    struct Tap {
        int dx, dy;
        double w;
    };
    std::vector<Tap> taps;
    for (int j = -r; j <= r; ++j)
        for (int i = -r; i <= r; ++i)
            if (i * i + j * j <= r * r) taps.push_back({i, j, std::exp(-(i * i + j * j) / (2 * sigmaSpace * sigmaSpace))});
    const std::vector<int> xt = detail::borderTable(W, r, r, border);
    const std::vector<int> yt = detail::borderTable(H, r, r, border);
    const double cc = -1.0 / (2 * sigmaColor * sigmaColor);
    // Integer images: the L1 color distance is an integer, so range weights come from a table.
    std::vector<double> rangeLut;
    if constexpr (std::is_integral<T>::value) {
        const double maxDist = PixelTraits<T>::max() * cn;
        if (maxDist <= (1 << 20)) {
            rangeLut.resize(static_cast<std::size_t>(maxDist) + 1);
            for (std::size_t k = 0; k < rangeLut.size(); ++k) rangeLut[k] = std::exp(static_cast<double>(k * k) * cc);
        }
    }
    parallelFor(0, H, [&](int y0, int y1) {
        std::vector<double> acc(static_cast<std::size_t>(cn));
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x) {
                const T* p0 = src.ptr(x, y);
                std::fill(acc.begin(), acc.end(), 0.0);
                double wsum = 0;
                for (const Tap& t : taps) {
                    const int sx = xt[x + r + t.dx], sy = yt[y + r + t.dy];
                    if (sx < 0 || sy < 0) continue;  // Constant border: skip outside pixels
                    const T* p = src.ptr(sx, sy);
                    double dist = 0;
                    for (int c = 0; c < cn; ++c) dist += std::fabs(static_cast<double>(p[c]) - static_cast<double>(p0[c]));
                    const double w = t.w * (rangeLut.empty() ? std::exp(dist * dist * cc)
                                                             : rangeLut[static_cast<std::size_t>(dist)]);
                    wsum += w;
                    for (int c = 0; c < cn; ++c) acc[c] += w * static_cast<double>(p[c]);
                }
                for (int c = 0; c < cn; ++c) out(x, y, c) = saturate_cast<T>(acc[c] / wsum);
            }
    }, static_cast<double>(W) * taps.size() * (cn + 2));
    return out;
}

// ---------------------------------------------------------------------------
// Derivatives (float output)
// ---------------------------------------------------------------------------

/// Sobel derivative of order (dx, dy) with an odd ksize in [1, 31] (ksize 1 = no smoothing).
/// `scale` multiplies the result. Output is float with the same channel count.
template <class T>
ImageF sobel(const Image<T>& src, int dx, int dy, int ksize = 3, double scale = 1.0,
             BorderMode border = BorderMode::Reflect101) {
    detail::require(dx >= 0 && dy >= 0 && dx + dy > 0, "sobel: need dx + dy > 0");
    std::vector<double> kx = detail::derivKernel(dx, ksize);
    std::vector<double> ky = detail::derivKernel(dy, ksize);
    if (ksize == 1) {  // keep both kernels the same odd length so anchors stay centered
        if (kx.size() == 1) kx = {0.0, 1.0, 0.0};
        if (ky.size() == 1) ky = {0.0, 1.0, 0.0};
    }
    for (double& v : kx) v *= scale;
    return sepFilter2D<float>(src, kx, ky, border);
}

/// Scharr derivative (more rotation-invariant 3x3 derivative); exactly one of dx, dy must be 1.
template <class T>
ImageF scharr(const Image<T>& src, int dx, int dy, double scale = 1.0, BorderMode border = BorderMode::Reflect101) {
    detail::require((dx == 1 && dy == 0) || (dx == 0 && dy == 1), "scharr: (dx, dy) must be (1,0) or (0,1)");
    const std::vector<double> d = {-scale, 0.0, scale}, s = {3.0, 10.0, 3.0};
    return dx ? sepFilter2D<float>(src, d, s, border) : sepFilter2D<float>(src, s, d, border);
}

/// Prewitt derivative (box-smoothed central difference); exactly one of dx, dy must be 1.
template <class T>
ImageF prewitt(const Image<T>& src, int dx, int dy, BorderMode border = BorderMode::Reflect101) {
    detail::require((dx == 1 && dy == 0) || (dx == 0 && dy == 1), "prewitt: (dx, dy) must be (1,0) or (0,1)");
    const std::vector<double> d = {-1.0, 0.0, 1.0}, s = {1.0, 1.0, 1.0};
    return dx ? sepFilter2D<float>(src, d, s, border) : sepFilter2D<float>(src, s, d, border);
}

/// Laplacian. ksize = 1 uses the 3x3 kernel [0 1 0; 1 -4 1; 0 1 0]; larger ksize sums
/// second-order Sobel derivatives (OpenCV semantics).
template <class T>
ImageF laplacian(const Image<T>& src, int ksize = 1, double scale = 1.0, BorderMode border = BorderMode::Reflect101) {
    if (ksize == 1) {
        Kernel k = makeKernel(3, 3, {0, 1, 0, 1, -4, 1, 0, 1, 0});
        for (float& v : k) v = static_cast<float>(v * scale);
        return filter2D<float>(src, k, border);
    }
    ImageF a = sobel(src, 2, 0, ksize, scale, border);
    const ImageF b = sobel(src, 0, 2, ksize, scale, border);
    for (std::size_t i = 0; i < a.sampleCount(); ++i) a.data()[i] += b.data()[i];
    return a;
}

/// Gradient magnitude sqrt(gx^2 + gy^2) of two derivative images.
inline ImageF magnitude(const ImageF& gx, const ImageF& gy) {
    detail::require(gx.sameShape(gy), "magnitude: shape mismatch");
    ImageF out(gx.width(), gx.height(), gx.channels());
    for (std::size_t i = 0; i < gx.sampleCount(); ++i) out.data()[i] = std::hypot(gx.data()[i], gy.data()[i]);
    return out;
}

/// Gradient direction atan2(gy, gx) in degrees [0, 360) (or radians in [0, 2 pi)).
inline ImageF phase(const ImageF& gx, const ImageF& gy, bool degrees = true) {
    detail::require(gx.sameShape(gy), "phase: shape mismatch");
    ImageF out(gx.width(), gx.height(), gx.channels());
    const double full = degrees ? 360.0 : 2 * detail::kPi;
    for (std::size_t i = 0; i < gx.sampleCount(); ++i) {
        double a = std::atan2(gy.data()[i], gx.data()[i]);
        if (degrees) a *= 180.0 / detail::kPi;
        if (a < 0) a += full;
        out.data()[i] = static_cast<float>(a);
    }
    return out;
}

/// Sobel gradient magnitude in one call (single- or multi-channel input).
template <class T>
ImageF gradientMagnitude(const Image<T>& src, int ksize = 3) {
    return magnitude(sobel(src, 1, 0, ksize), sobel(src, 0, 1, ksize));
}

// ---------------------------------------------------------------------------
// Sharpening and kernels
// ---------------------------------------------------------------------------

/// Unsharp masking: out = src + amount * (src - gaussianBlur(src, sigma)), applied only where
/// |src - blurred| > threshold (in pixel units).
template <class T>
Image<T> unsharpMask(const Image<T>& src, double sigma = 1.0, double amount = 1.0, double threshold = 0.0) {
    const ImageD blurred = gaussianBlur<double>(src, sigma);
    Image<T> out(src.width(), src.height(), src.channels());
    for (std::size_t i = 0; i < src.sampleCount(); ++i) {
        const double v = src.data()[i], diff = v - blurred.data()[i];
        out.data()[i] = saturate_cast<T>(std::fabs(diff) > threshold ? v + amount * diff : v);
    }
    return out;
}

/// Simple 3x3 sharpening: out = src + strength * (src - 4-neighbour mean) via a Laplacian kernel.
template <class T>
Image<T> sharpen(const Image<T>& src, double strength = 1.0) {
    const float s = static_cast<float>(strength);
    return filter2D(src, makeKernel(3, 3, {0, -s, 0, -s, 1 + 4 * s, -s, 0, -s, 0}));
}

/// Emboss effect (3x3 directional kernel) with a mid-gray offset.
template <class T>
Image<T> emboss(const Image<T>& src) {
    return filter2D(src, makeKernel(3, 3, {-2, -1, 0, -1, 1, 1, 0, 1, 2}));
}

/// Line kernel for motion blur: a normalized line of `length` pixels through the kernel
/// center at `angleDegrees` (0 = horizontal, counter-clockwise).
inline Kernel motionBlurKernel(int length, double angleDegrees) {
    detail::require(length >= 1, "motionBlurKernel: length must be >= 1");
    const int size = length % 2 ? length : length + 1;
    Kernel k(size, size, 1, 0.f);
    const double a = angleDegrees * detail::kPi / 180.0, c = (size - 1) / 2.0;
    const int samples = 4 * length;
    for (int i = 0; i < samples; ++i) {
        const double t = (length - 1) * (static_cast<double>(i) / (samples - 1) - 0.5);
        const int x = static_cast<int>(std::lround(c + t * std::cos(a)));
        const int y = static_cast<int>(std::lround(c - t * std::sin(a)));
        k(clamp(x, 0, size - 1), clamp(y, 0, size - 1)) = 1.f;
    }
    float n = 0;
    for (float v : k) n += v;
    for (float& v : k) v /= n;
    return k;
}

// ---------------------------------------------------------------------------
// Integral images and adaptive thresholding
// ---------------------------------------------------------------------------

/// Summed-area table with an extra zero row/column: I(x, y) = sum of src over [0,x) x [0,y).
/// Size (W+1) x (H+1), same channel count, double precision.
template <class T>
ImageD integralImage(const Image<T>& src, bool squared = false) {
    const int W = src.width(), H = src.height(), cn = src.channels();
    ImageD I(W + 1, H + 1, cn, 0.0);
    for (int y = 0; y < H; ++y) {
        std::vector<double> rowSum(static_cast<std::size_t>(cn), 0.0);
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < cn; ++c) {
                const double v = src(x, y, c);
                rowSum[c] += squared ? v * v : v;
                I(x + 1, y + 1, c) = I(x + 1, y, c) + rowSum[c];
            }
    }
    return I;
}

/// Sum over the rectangle r using an integral image from integralImage().
inline double integralSum(const ImageD& I, Rect r, int channel = 0) {
    return I(r.right(), r.bottom(), channel) - I(r.x, r.bottom(), channel) - I(r.right(), r.y, channel) +
           I(r.x, r.y, channel);
}

enum class AdaptiveMethod { Mean, Gaussian };

/// Adaptive threshold of a single-channel image (OpenCV semantics): the threshold is the local
/// mean (or Gaussian-weighted mean) of a blockSize x blockSize neighbourhood minus C.
/// type must be ThresholdType::Binary or ThresholdType::BinaryInv.
template <class T>
Image<T> adaptiveThreshold(const Image<T>& src, double maxval, AdaptiveMethod method, ThresholdType type, int blockSize,
                           double C) {
    detail::require(src.channels() == 1, "adaptiveThreshold: expected a single-channel image");
    detail::require(blockSize >= 3 && blockSize % 2 == 1, "adaptiveThreshold: blockSize must be odd and >= 3");
    detail::require(type == ThresholdType::Binary || type == ThresholdType::BinaryInv,
                    "adaptiveThreshold: type must be Binary or BinaryInv");
    const Image<T> mean = method == AdaptiveMethod::Mean
                              ? boxFilter(src, blockSize, blockSize, true, BorderMode::Replicate)
                              : gaussianBlur(src, 0.0, 0.0, blockSize, BorderMode::Replicate);
    Image<T> out(src.width(), src.height(), 1);
    for (std::size_t i = 0; i < src.sampleCount(); ++i) {
        const bool above = static_cast<double>(src.data()[i]) - static_cast<double>(mean.data()[i]) > -C;
        out.data()[i] = saturate_cast<T>((above == (type == ThresholdType::Binary)) ? maxval : 0.0);
    }
    return out;
}

}  // namespace ipf
