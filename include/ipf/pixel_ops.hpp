// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf pixel_ops: per-pixel arithmetic, blending, masks, thresholding,
// intensity transforms, histograms, equalization (global and CLAHE).
#pragma once

#include <array>

#include "core.hpp"

namespace ipf {

namespace detail {
template <class T, class Fn>
Image<T> binaryOp(const Image<T>& a, const Image<T>& b, Fn fn, const char* name) {
    if (!a.sameShape(b)) fail(std::string(name) + ": images must have the same size and channel count");
    Image<T> out(a.width(), a.height(), a.channels());
    const T* pa = a.data();
    const T* pb = b.data();
    T* d = out.data();
    parallelForN(a.sampleCount(), [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i)
            d[i] = saturate_cast<T>(fn(static_cast<double>(pa[i]), static_cast<double>(pb[i])));
    });
    return out;
}
template <class T, class Fn>
Image<T> unaryOp(const Image<T>& a, Fn fn) {
    Image<T> out(a.width(), a.height(), a.channels());
    const T* pa = a.data();
    T* d = out.data();
    parallelForN(a.sampleCount(), [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i) d[i] = saturate_cast<T>(fn(static_cast<double>(pa[i])));
    });
    return out;
}
template <class T>
void requireMask(const Image<T>& img, const ImageU8& mask, const char* name) {
    if (mask.channels() != 1 || mask.width() != img.width() || mask.height() != img.height())
        fail(std::string(name) + ": mask must be single-channel and the same size as the image");
}
}  // namespace detail

// ---------------------------------------------------------------------------
// Arithmetic (results saturate to the pixel type's range)
// ---------------------------------------------------------------------------

template <class T>
Image<T> add(const Image<T>& a, const Image<T>& b) {
    return detail::binaryOp(a, b, [](double x, double y) { return x + y; }, "add");
}
template <class T>
Image<T> subtract(const Image<T>& a, const Image<T>& b) {
    return detail::binaryOp(a, b, [](double x, double y) { return x - y; }, "subtract");
}
/// Per-sample product times `scale` (e.g. scale = 1/255 to multiply two uint8 images as [0,1] values).
template <class T>
Image<T> multiply(const Image<T>& a, const Image<T>& b, double scale = 1.0) {
    return detail::binaryOp(a, b, [scale](double x, double y) { return x * y * scale; }, "multiply");
}
/// Per-sample quotient times `scale`; division by zero yields 0.
template <class T>
Image<T> divide(const Image<T>& a, const Image<T>& b, double scale = 1.0) {
    return detail::binaryOp(a, b, [scale](double x, double y) { return y != 0 ? x * scale / y : 0.0; }, "divide");
}
template <class T>
Image<T> absDiff(const Image<T>& a, const Image<T>& b) {
    return detail::binaryOp(a, b, [](double x, double y) { return std::fabs(x - y); }, "absDiff");
}
template <class T>
Image<T> minImage(const Image<T>& a, const Image<T>& b) {
    return detail::binaryOp(a, b, [](double x, double y) { return std::min(x, y); }, "minImage");
}
template <class T>
Image<T> maxImage(const Image<T>& a, const Image<T>& b) {
    return detail::binaryOp(a, b, [](double x, double y) { return std::max(x, y); }, "maxImage");
}
/// out = a * alpha + b * beta + gamma (OpenCV's addWeighted).
template <class T>
Image<T> addWeighted(const Image<T>& a, double alpha, const Image<T>& b, double beta, double gamma = 0.0) {
    return detail::binaryOp(a, b, [=](double x, double y) { return x * alpha + y * beta + gamma; }, "addWeighted");
}
/// Linear interpolation: t = 0 gives a, t = 1 gives b.
template <class T>
Image<T> blend(const Image<T>& a, const Image<T>& b, double t) {
    return addWeighted(a, 1.0 - t, b, t);
}
/// Adds a constant to every sample.
template <class T>
Image<T> addScalar(const Image<T>& a, double v) {
    return detail::unaryOp(a, [v](double x) { return x + v; });
}
/// Multiplies every sample by a constant.
template <class T>
Image<T> multiplyScalar(const Image<T>& a, double v) {
    return detail::unaryOp(a, [v](double x) { return x * v; });
}
/// Photographic negative: max - v for every channel (alpha included unless keepAlpha).
template <class T>
Image<T> invert(const Image<T>& a, bool keepAlpha = true) {
    const double mx = PixelTraits<T>::max();
    Image<T> out = detail::unaryOp(a, [mx](double x) { return mx - x; });
    if (keepAlpha && (a.channels() == 2 || a.channels() == 4)) insertChannel(out, extractChannel(a, a.channels() - 1), a.channels() - 1);
    return out;
}
/// out = saturate(alpha * v + beta): alpha controls contrast, beta brightness.
template <class T>
Image<T> brightnessContrast(const Image<T>& a, double alpha, double beta) {
    return detail::unaryOp(a, [=](double x) { return alpha * x + beta; });
}
/// Clamps all samples to [lo, hi].
template <class T>
Image<T> clampValues(const Image<T>& a, double lo, double hi) {
    return detail::unaryOp(a, [=](double x) { return clamp(x, lo, hi); });
}

/// Alpha-composites `fg` (RGBA or GA) over `bg` (same size; RGB/RGBA or gray/GA).
/// Uses the standard "over" operator; the result has the channel count of bg.
template <class T>
Image<T> alphaComposite(const Image<T>& fg, const Image<T>& bg) {
    detail::require(fg.width() == bg.width() && fg.height() == bg.height(), "alphaComposite: size mismatch");
    detail::require(fg.channels() == 2 || fg.channels() == 4, "alphaComposite: foreground needs an alpha channel");
    const int fc = fg.channels(), bc = bg.channels();
    const int colors = fc - 1;
    detail::require(bc == colors || bc == colors + 1, "alphaComposite: incompatible channel counts");
    const double mx = PixelTraits<T>::max();
    Image<T> out(bg.width(), bg.height(), bc);
    for (std::size_t i = 0; i < out.pixelCount(); ++i) {
        const T* f = fg.data() + i * fc;
        const T* b = bg.data() + i * bc;
        T* o = out.data() + i * bc;
        const double af = f[colors] / mx;
        const double ab = bc > colors ? b[colors] / mx : 1.0;
        const double ao = af + ab * (1.0 - af);
        for (int c = 0; c < colors; ++c) {
            const double v = ao > 0 ? (f[c] * af + b[c] * ab * (1.0 - af)) / ao : 0.0;
            o[c] = saturate_cast<T>(v);
        }
        if (bc > colors) o[colors] = saturate_cast<T>(ao * mx);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Masks and bitwise operations
// ---------------------------------------------------------------------------

enum class CmpOp { Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual };

namespace detail {
inline bool cmp(double a, double b, CmpOp op) {
    switch (op) {
        case CmpOp::Equal: return a == b;
        case CmpOp::NotEqual: return a != b;
        case CmpOp::Less: return a < b;
        case CmpOp::LessEqual: return a <= b;
        case CmpOp::Greater: return a > b;
        default: return a >= b;
    }
}
}  // namespace detail

/// Per-sample comparison producing a uint8 mask (255 = true, 0 = false), same channel count.
template <class T>
ImageU8 compare(const Image<T>& a, const Image<T>& b, CmpOp op) {
    detail::require(a.sameShape(b), "compare: images must have the same shape");
    ImageU8 out(a.width(), a.height(), a.channels());
    for (std::size_t i = 0; i < a.sampleCount(); ++i)
        out.data()[i] = detail::cmp(a.data()[i], b.data()[i], op) ? 255 : 0;
    return out;
}
/// Compares every sample with a scalar.
template <class T>
ImageU8 compare(const Image<T>& a, double v, CmpOp op) {
    ImageU8 out(a.width(), a.height(), a.channels());
    for (std::size_t i = 0; i < a.sampleCount(); ++i) out.data()[i] = detail::cmp(a.data()[i], v, op) ? 255 : 0;
    return out;
}

/// Single-channel mask: 255 where lower[c] <= pixel[c] <= upper[c] for every channel.
template <class T>
ImageU8 inRange(const Image<T>& img, const std::vector<double>& lower, const std::vector<double>& upper) {
    const int cn = img.channels();
    detail::require(static_cast<int>(lower.size()) == cn && static_cast<int>(upper.size()) == cn,
                    "inRange: bounds must have one value per channel");
    ImageU8 out(img.width(), img.height(), 1);
    for (std::size_t i = 0; i < img.pixelCount(); ++i) {
        bool in = true;
        for (int c = 0; c < cn && in; ++c) {
            const double v = img.data()[i * cn + c];
            in = v >= lower[c] && v <= upper[c];
        }
        out.data()[i] = in ? 255 : 0;
    }
    return out;
}

template <class T>
Image<T> bitwiseAnd(const Image<T>& a, const Image<T>& b) {
    static_assert(std::is_integral<T>::value, "bitwise operations require integer images");
    detail::require(a.sameShape(b), "bitwiseAnd: shape mismatch");
    Image<T> out = a;
    for (std::size_t i = 0; i < a.sampleCount(); ++i) out.data()[i] = static_cast<T>(a.data()[i] & b.data()[i]);
    return out;
}
template <class T>
Image<T> bitwiseOr(const Image<T>& a, const Image<T>& b) {
    static_assert(std::is_integral<T>::value, "bitwise operations require integer images");
    detail::require(a.sameShape(b), "bitwiseOr: shape mismatch");
    Image<T> out = a;
    for (std::size_t i = 0; i < a.sampleCount(); ++i) out.data()[i] = static_cast<T>(a.data()[i] | b.data()[i]);
    return out;
}
template <class T>
Image<T> bitwiseXor(const Image<T>& a, const Image<T>& b) {
    static_assert(std::is_integral<T>::value, "bitwise operations require integer images");
    detail::require(a.sameShape(b), "bitwiseXor: shape mismatch");
    Image<T> out = a;
    for (std::size_t i = 0; i < a.sampleCount(); ++i) out.data()[i] = static_cast<T>(a.data()[i] ^ b.data()[i]);
    return out;
}
template <class T>
Image<T> bitwiseNot(const Image<T>& a) {
    static_assert(std::is_integral<T>::value, "bitwise operations require integer images");
    Image<T> out = a;
    for (auto& v : out) v = static_cast<T>(~v);
    return out;
}

/// Sets all pixels where mask != 0 to `color` (one value per channel).
template <class T>
void setTo(Image<T>& img, const std::vector<T>& color, const ImageU8& mask) {
    detail::requireMask(img, mask, "setTo");
    detail::require(static_cast<int>(color.size()) == img.channels(), "setTo: color size != channels");
    for (std::size_t i = 0; i < img.pixelCount(); ++i)
        if (mask.data()[i]) std::copy(color.begin(), color.end(), img.data() + i * img.channels());
}
/// Copies src pixels into dst where mask != 0.
template <class T>
void copyMasked(const Image<T>& src, Image<T>& dst, const ImageU8& mask) {
    detail::require(src.sameShape(dst), "copyMasked: src and dst shapes differ");
    detail::requireMask(src, mask, "copyMasked");
    const int cn = src.channels();
    for (std::size_t i = 0; i < src.pixelCount(); ++i)
        if (mask.data()[i]) std::copy(src.data() + i * cn, src.data() + (i + 1) * cn, dst.data() + i * cn);
}

// ---------------------------------------------------------------------------
// Thresholding
// ---------------------------------------------------------------------------

/// OpenCV-compatible threshold types.
enum class ThresholdType {
    Binary,     ///< v > t ? maxval : 0
    BinaryInv,  ///< v > t ? 0 : maxval
    Trunc,      ///< v > t ? t : v
    ToZero,     ///< v > t ? v : 0
    ToZeroInv   ///< v > t ? 0 : v
};

/// Applies a fixed threshold to every sample.
template <class T>
Image<T> threshold(const Image<T>& img, double thresh, double maxval, ThresholdType type = ThresholdType::Binary) {
    return detail::unaryOp(img, [=](double v) {
        const bool above = v > thresh;
        switch (type) {
            case ThresholdType::Binary: return above ? maxval : 0.0;
            case ThresholdType::BinaryInv: return above ? 0.0 : maxval;
            case ThresholdType::Trunc: return above ? thresh : v;
            case ThresholdType::ToZero: return above ? v : 0.0;
            default: return above ? 0.0 : v;
        }
    });
}

/// 256-bin histogram of one channel of a uint8 image.
inline std::array<std::uint64_t, 256> histogram(const ImageU8& img, int channel = 0) {
    detail::require(channel >= 0 && channel < img.channels(), "histogram: invalid channel");
    std::array<std::uint64_t, 256> h{};
    const int cn = img.channels();
    const std::uint8_t* p = img.data() + channel;
    for (std::size_t i = 0; i < img.pixelCount(); ++i) ++h[p[i * cn]];
    return h;
}

/// Histogram with `bins` equal-width bins over [lo, hi) of one channel of any image type
/// (values outside the range are ignored; hi itself is counted in the last bin).
template <class T>
std::vector<std::uint64_t> histogram(const Image<T>& img, int bins, double lo, double hi, int channel = 0) {
    detail::require(bins > 0 && hi > lo, "histogram: invalid bins or range");
    detail::require(channel >= 0 && channel < img.channels(), "histogram: invalid channel");
    std::vector<std::uint64_t> h(static_cast<std::size_t>(bins), 0);
    const int cn = img.channels();
    const double scale = bins / (hi - lo);
    for (std::size_t i = 0; i < img.pixelCount(); ++i) {
        const double v = img.data()[i * cn + channel];
        if (v < lo || v > hi) continue;
        const int b = std::min(bins - 1, static_cast<int>((v - lo) * scale));
        ++h[b];
    }
    return h;
}

/// Otsu's threshold of a 256-bin histogram (maximizes between-class variance).
inline int otsuThreshold(const std::array<std::uint64_t, 256>& h) {
    double total = 0, sum = 0;
    for (int i = 0; i < 256; ++i) {
        total += static_cast<double>(h[i]);
        sum += i * static_cast<double>(h[i]);
    }
    double wB = 0, sumB = 0, best = -1;
    int t = 0;
    for (int i = 0; i < 256; ++i) {
        wB += static_cast<double>(h[i]);
        if (wB == 0) continue;
        const double wF = total - wB;
        if (wF == 0) break;
        sumB += i * static_cast<double>(h[i]);
        const double mB = sumB / wB, mF = (sum - sumB) / wF;
        const double between = wB * wF * (mB - mF) * (mB - mF);
        if (between > best) {
            best = between;
            t = i;
        }
    }
    return t;
}

/// Otsu's threshold of a single-channel uint8 image. Use with threshold(img, t, 255).
inline int otsuThreshold(const ImageU8& gray) { return otsuThreshold(histogram(gray, 0)); }

/// Otsu's threshold for any pixel type (computed on 256 bins between the image min and max).
template <class T>
double otsuThresholdValue(const Image<T>& gray) {
    if constexpr (std::is_same<T, std::uint8_t>::value) {
        return otsuThreshold(gray);
    } else {
        const MinMaxResult mm = minMaxLoc(gray, 0);
        if (mm.maxVal <= mm.minVal) return mm.minVal;
        const auto hv = histogram(gray, 256, mm.minVal, mm.maxVal, 0);
        std::array<std::uint64_t, 256> h{};
        std::copy(hv.begin(), hv.end(), h.begin());
        const int t = otsuThreshold(h);
        return mm.minVal + (t + 1) * (mm.maxVal - mm.minVal) / 256.0;
    }
}

/// Binarizes a single-channel image with Otsu's threshold (255 / maxval = foreground above threshold).
template <class T>
Image<T> thresholdOtsu(const Image<T>& gray, double maxval = PixelTraits<T>::max(), bool inverse = false) {
    detail::require(gray.channels() == 1, "thresholdOtsu: expected a single-channel image");
    return threshold(gray, otsuThresholdValue(gray), maxval, inverse ? ThresholdType::BinaryInv : ThresholdType::Binary);
}

// ---------------------------------------------------------------------------
// Lookup tables and intensity transforms
// ---------------------------------------------------------------------------

/// Applies a 256-entry lookup table to every sample of a uint8 image.
inline ImageU8 applyLUT(const ImageU8& img, const std::array<std::uint8_t, 256>& lut) {
    ImageU8 out(img.width(), img.height(), img.channels());
    for (std::size_t i = 0; i < img.sampleCount(); ++i) out.data()[i] = lut[img.data()[i]];
    return out;
}
/// Applies one 256-entry lookup table per channel.
inline ImageU8 applyLUT(const ImageU8& img, const std::vector<std::array<std::uint8_t, 256>>& luts) {
    detail::require(static_cast<int>(luts.size()) == img.channels(), "applyLUT: need one LUT per channel");
    ImageU8 out(img.width(), img.height(), img.channels());
    const int cn = img.channels();
    for (std::size_t i = 0; i < img.sampleCount(); ++i) out.data()[i] = luts[i % cn][img.data()[i]];
    return out;
}

/// Gamma correction: out = max * (v / max)^(1 / gamma). gamma > 1 brightens mid-tones,
/// gamma < 1 darkens them. All channels (including alpha) are processed.
template <class T>
Image<T> gammaCorrect(const Image<T>& img, double gamma) {
    detail::require(gamma > 0, "gammaCorrect: gamma must be positive");
    const double mx = PixelTraits<T>::max(), inv = 1.0 / gamma;
    if constexpr (std::is_same<T, std::uint8_t>::value) {
        std::array<std::uint8_t, 256> lut;
        for (int i = 0; i < 256; ++i) lut[i] = saturate_cast<std::uint8_t>(mx * std::pow(i / mx, inv));
        return applyLUT(img, lut);
    } else {
        return detail::unaryOp(img, [=](double v) { return mx * std::pow(std::max(0.0, v) / mx, inv); });
    }
}

/// Linearly rescales values so that the image min maps to newMin and max to newMax.
/// perChannel = true normalizes every channel independently.
template <class T>
Image<T> normalize(const Image<T>& img, double newMin = 0, double newMax = PixelTraits<T>::max(), bool perChannel = false) {
    Image<T> out(img.width(), img.height(), img.channels());
    const int cn = img.channels();
    for (int c = 0; c < (perChannel ? cn : 1); ++c) {
        const MinMaxResult mm = minMaxLoc(img, perChannel ? c : -1);
        const double range = mm.maxVal - mm.minVal;
        const double scale = range > 0 ? (newMax - newMin) / range : 0.0;
        for (std::size_t i = 0; i < img.sampleCount(); ++i) {
            if (perChannel && static_cast<int>(i % cn) != c) continue;
            out.data()[i] = saturate_cast<T>((img.data()[i] - mm.minVal) * scale + newMin);
        }
    }
    return out;
}

/// Maximizes contrast per channel (PIL ImageOps.autocontrast semantics): `cutoffPercent`
/// of the darkest and brightest pixels are clipped before stretching to [0, 255].
inline ImageU8 autoContrast(const ImageU8& img, double cutoffPercent = 0.0) {
    std::vector<std::array<std::uint8_t, 256>> luts;
    for (int c = 0; c < img.channels(); ++c) {
        auto h = histogram(img, c);
        std::uint64_t n = 0;
        for (auto v : h) n += v;
        if (cutoffPercent > 0) {
            auto cutEnd = [&](bool low) {
                auto cut = static_cast<std::uint64_t>(std::floor(static_cast<double>(n) * cutoffPercent / 100.0));
                for (int k = 0; k < 256 && cut > 0; ++k) {
                    auto& bin = h[low ? k : 255 - k];
                    if (cut > bin) {
                        cut -= bin;
                        bin = 0;
                    } else {
                        bin -= cut;
                        cut = 0;
                    }
                }
            };
            cutEnd(true);
            cutEnd(false);
        }
        int lo = 0, hi = 255;
        while (lo < 256 && !h[lo]) ++lo;
        while (hi >= 0 && !h[hi]) --hi;
        std::array<std::uint8_t, 256> lut;
        if (hi <= lo) {
            for (int i = 0; i < 256; ++i) lut[i] = static_cast<std::uint8_t>(i);
        } else {
            const double scale = 255.0 / (hi - lo), offset = -lo * scale;
            for (int i = 0; i < 256; ++i) lut[i] = static_cast<std::uint8_t>(clamp(static_cast<int>(i * scale + offset), 0, 255));
        }
        luts.push_back(lut);
    }
    return applyLUT(img, luts);
}

/// Histogram equalization of every channel of a uint8 image (OpenCV's equalizeHist formula).
/// For color images consider equalizing only the luma (see color.hpp: rgbToYCbCr).
inline ImageU8 equalizeHist(const ImageU8& img) {
    std::vector<std::array<std::uint8_t, 256>> luts;
    for (int c = 0; c < img.channels(); ++c) {
        const auto h = histogram(img, c);
        std::array<std::uint8_t, 256> lut{};
        int first = 0;
        while (first < 256 && !h[first]) ++first;
        const double total = static_cast<double>(img.pixelCount());
        if (first == 256 || static_cast<double>(h[first]) == total) {  // constant image: unchanged
            for (int i = 0; i < 256; ++i) lut[i] = static_cast<std::uint8_t>(i);
        } else {
            const double scale = 255.0 / (total - static_cast<double>(h[first]));
            double cdf = 0;
            for (int i = first + 1; i < 256; ++i) {
                cdf += static_cast<double>(h[i]);
                lut[i] = saturate_cast<std::uint8_t>(cdf * scale);
            }
            lut[first] = 0;
        }
        luts.push_back(lut);
    }
    return applyLUT(img, luts);
}

/// Contrast Limited Adaptive Histogram Equalization of a uint8 image (each channel separately).
/// clipLimit is relative to a uniform histogram (OpenCV semantics, typical 2-4); the image is
/// divided into tilesX x tilesY regions whose mappings are bilinearly interpolated.
inline ImageU8 clahe(const ImageU8& img, double clipLimit = 2.0, int tilesX = 8, int tilesY = 8) {
    detail::require(tilesX > 0 && tilesY > 0, "clahe: tile counts must be positive");
    if (img.empty()) return img;
    const int W = img.width(), H = img.height(), cn = img.channels();
    tilesX = std::min(tilesX, W);
    tilesY = std::min(tilesY, H);
    ImageU8 out(W, H, cn);
    std::vector<std::array<float, 256>> luts(static_cast<std::size_t>(tilesX) * tilesY);
    const double tw = static_cast<double>(W) / tilesX, th = static_cast<double>(H) / tilesY;
    for (int c = 0; c < cn; ++c) {
        for (int ty = 0; ty < tilesY; ++ty)
            for (int tx = 0; tx < tilesX; ++tx) {
                const int x0 = tx * W / tilesX, x1 = (tx + 1) * W / tilesX;
                const int y0 = ty * H / tilesY, y1 = (ty + 1) * H / tilesY;
                std::array<std::uint64_t, 256> h{};
                for (int y = y0; y < y1; ++y)
                    for (int x = x0; x < x1; ++x) ++h[img(x, y, c)];
                const std::uint64_t area = static_cast<std::uint64_t>(x1 - x0) * (y1 - y0);
                if (clipLimit > 0) {
                    const auto limit = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(clipLimit * area / 256.0));
                    std::uint64_t excess = 0;
                    for (auto& v : h)
                        if (v > limit) {
                            excess += v - limit;
                            v = limit;
                        }
                    const std::uint64_t add = excess / 256, residual = excess % 256;
                    for (auto& v : h) v += add;
                    if (residual) {
                        const std::uint64_t step = std::max<std::uint64_t>(1, 256 / residual);
                        std::uint64_t left = residual;
                        for (std::uint64_t i = 0; i < 256 && left; i += step, --left) ++h[i];
                    }
                }
                auto& lut = luts[static_cast<std::size_t>(ty) * tilesX + tx];
                double cdf = 0;
                for (int i = 0; i < 256; ++i) {
                    cdf += static_cast<double>(h[i]);
                    lut[i] = static_cast<float>(cdf * 255.0 / static_cast<double>(area));
                }
            }
        for (int y = 0; y < H; ++y) {
            const double fy = (y + 0.5) / th - 0.5;
            const int ty0 = clamp(static_cast<int>(std::floor(fy)), 0, tilesY - 1);
            const int ty1 = std::min(ty0 + 1, tilesY - 1);
            const double wy = clamp(fy - ty0, 0.0, 1.0);
            for (int x = 0; x < W; ++x) {
                const double fx = (x + 0.5) / tw - 0.5;
                const int tx0 = clamp(static_cast<int>(std::floor(fx)), 0, tilesX - 1);
                const int tx1 = std::min(tx0 + 1, tilesX - 1);
                const double wx = clamp(fx - tx0, 0.0, 1.0);
                const int v = img(x, y, c);
                const double a = luts[static_cast<std::size_t>(ty0) * tilesX + tx0][v];
                const double b = luts[static_cast<std::size_t>(ty0) * tilesX + tx1][v];
                const double cc = luts[static_cast<std::size_t>(ty1) * tilesX + tx0][v];
                const double d = luts[static_cast<std::size_t>(ty1) * tilesX + tx1][v];
                const double top = a + (b - a) * wx, bottom = cc + (d - cc) * wx;
                out(x, y, c) = saturate_cast<std::uint8_t>(top + (bottom - top) * wy);
            }
        }
    }
    return out;
}

/// Remaps each channel of `src` so that its histogram matches that of `reference`
/// (classic CDF matching). Images may differ in size but must have equal channel counts.
inline ImageU8 matchHistogram(const ImageU8& src, const ImageU8& reference) {
    detail::require(src.channels() == reference.channels(), "matchHistogram: channel counts differ");
    std::vector<std::array<std::uint8_t, 256>> luts;
    for (int c = 0; c < src.channels(); ++c) {
        const auto hs = histogram(src, c), hr = histogram(reference, c);
        std::array<double, 256> cs{}, cr{};
        double as = 0, ar = 0;
        for (int i = 0; i < 256; ++i) {
            as += static_cast<double>(hs[i]);
            ar += static_cast<double>(hr[i]);
            cs[i] = as;
            cr[i] = ar;
        }
        std::array<std::uint8_t, 256> lut{};
        int r = 0;
        for (int i = 0; i < 256; ++i) {
            const double target = cs[i] / std::max(1.0, as);
            while (r < 255 && cr[r] / std::max(1.0, ar) < target - 1e-12) ++r;
            lut[i] = static_cast<std::uint8_t>(r);
        }
        luts.push_back(lut);
    }
    return applyLUT(src, luts);
}

}  // namespace ipf
