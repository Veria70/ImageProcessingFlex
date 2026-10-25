// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf color: color space conversions (Gray, HSV, HSL, YCbCr, XYZ, CIE Lab),
// color adjustments and false-color maps.
//
// Conventions: RGB images have 3 channels (a 4th alpha channel is accepted on input
// and ignored unless stated). Integer images use their full range (0-255 for uint8),
// float images use [0, 1]. Hue is in degrees [0, 360); S, V, L are in [0, 1];
// CIE L*a*b* uses L in [0, 100] (D65 white point, sRGB primaries).
// All conversions process pixels in parallel for large images.
#pragma once

#include <array>

#include "codec/common.hpp"
#include "colormap_data.hpp"
#include "core.hpp"

namespace ipf {

namespace detail {
template <class T>
void requireColor(const Image<T>& img, const char* fn) {
    if (img.channels() != 3 && img.channels() != 4)
        fail(std::string(fn) + ": expected a 3-channel (RGB) or 4-channel (RGBA) image");
}
template <class T>
constexpr double chromaOffset() {  // 128 for uint8, 32768 for uint16, 0.5 for float
    return std::is_floating_point<T>::value ? 0.5 : (PixelTraits<T>::max() + 1.0) / 2.0;
}
inline double srgbToLinear(double c) { return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4); }
inline double linearToSrgb(double c) {
    return c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
}
}  // namespace detail

// ---------------------------------------------------------------------------
// Grayscale
// ---------------------------------------------------------------------------

enum class GrayMethod {
    Luma601,    ///< 0.299 R + 0.587 G + 0.114 B (JPEG / PIL / OpenCV default)
    Luma709,    ///< 0.2126 R + 0.7152 G + 0.0722 B (HDTV / sRGB luminance)
    Average,    ///< (R + G + B) / 3
    Lightness,  ///< (max + min) / 2
    Max         ///< max(R, G, B) (the HSV "value")
};

/// Converts an RGB(A) image to a single-channel gray image.
template <class T>
Image<T> rgbToGray(const Image<T>& rgb, GrayMethod method = GrayMethod::Luma601) {
    detail::requireColor(rgb, "rgbToGray");
    Image<T> out(rgb.width(), rgb.height(), 1);
    detail::forEachPixel(rgb, out, [method](const T* s, T* d) {
        const double r = s[0], g = s[1], b = s[2];
        double v;
        switch (method) {
            case GrayMethod::Luma709: v = 0.2126 * r + 0.7152 * g + 0.0722 * b; break;
            case GrayMethod::Average: v = (r + g + b) / 3.0; break;
            case GrayMethod::Lightness: v = (std::max({r, g, b}) + std::min({r, g, b})) / 2.0; break;
            case GrayMethod::Max: v = std::max({r, g, b}); break;
            default: v = 0.299 * r + 0.587 * g + 0.114 * b; break;
        }
        d[0] = saturate_cast<T>(v);
    }, 2);
    return out;
}

/// Replicates a single-channel image into 3 RGB channels.
template <class T>
Image<T> grayToRgb(const Image<T>& gray) {
    detail::require(gray.channels() == 1, "grayToRgb: expected a single-channel image");
    return convertChannels(gray, 3);
}

/// Returns a single-channel version of any image (gray stays gray, RGB(A) is converted).
template <class T>
Image<T> toGray(const Image<T>& img) {
    if (img.channels() <= 2) return extractChannel(img, 0);
    return rgbToGray(img);
}

/// Swaps the first and third channel (RGB <-> BGR, RGBA <-> BGRA).
template <class T>
Image<T> swapRB(const Image<T>& img) {
    detail::requireColor(img, "swapRB");
    Image<T> out = img;
    const int cn = img.channels();
    T* p = out.data();
    for (std::size_t i = 0; i < out.pixelCount(); ++i, p += cn) std::swap(p[0], p[2]);
    return out;
}

// ---------------------------------------------------------------------------
// HSV / HSL
// ---------------------------------------------------------------------------

/// RGB -> HSV. Output: float image with H in [0, 360), S and V in [0, 1].
template <class T>
ImageF rgbToHsv(const Image<T>& rgb) {
    detail::requireColor(rgb, "rgbToHsv");
    ImageF out(rgb.width(), rgb.height(), 3);
    const double inv = 1.0 / PixelTraits<T>::max();
    detail::forEachPixel(rgb, out, [inv](const T* s, float* d) {
        const double r = s[0] * inv, g = s[1] * inv, b = s[2] * inv;
        const double mx = std::max({r, g, b}), mn = std::min({r, g, b}), delta = mx - mn;
        double h = 0;
        if (delta > 0) {
            if (mx == r) h = 60.0 * std::fmod((g - b) / delta + 6.0, 6.0);
            else if (mx == g) h = 60.0 * ((b - r) / delta + 2.0);
            else h = 60.0 * ((r - g) / delta + 4.0);
        }
        if (h >= 360.0) h -= 360.0;
        d[0] = static_cast<float>(h);
        d[1] = static_cast<float>(mx > 0 ? delta / mx : 0.0);
        d[2] = static_cast<float>(mx);
    }, 12);
    return out;
}

namespace detail {
inline void hueToRgb(double h, double c, double m, double* rgb) {
    h = std::fmod(h, 360.0);
    if (h < 0) h += 360.0;
    const double hp = h / 60.0;
    const double x = c * (1.0 - std::fabs(std::fmod(hp, 2.0) - 1.0));
    double r = 0, g = 0, b = 0;
    switch (static_cast<int>(hp)) {
        case 0: r = c; g = x; break;
        case 1: r = x; g = c; break;
        case 2: g = c; b = x; break;
        case 3: g = x; b = c; break;
        case 4: r = x; b = c; break;
        default: r = c; b = x; break;
    }
    rgb[0] = r + m;
    rgb[1] = g + m;
    rgb[2] = b + m;
}
}  // namespace detail

/// HSV (as produced by rgbToHsv) -> RGB of pixel type T.
template <class T = std::uint8_t>
Image<T> hsvToRgb(const ImageF& hsv) {
    detail::require(hsv.channels() == 3, "hsvToRgb: expected a 3-channel HSV image");
    Image<T> out(hsv.width(), hsv.height(), 3);
    const double mx = PixelTraits<T>::max();
    detail::forEachPixel(hsv, out, [mx](const float* s, T* d) {
        double rgb[3];
        const double v = s[2], c = v * s[1];
        detail::hueToRgb(s[0], c, v - c, rgb);
        for (int k = 0; k < 3; ++k) d[k] = saturate_cast<T>(clamp(rgb[k], 0.0, 1.0) * mx);
    }, 12);
    return out;
}

/// RGB -> HSL. Output: float image with H in [0, 360), S and L in [0, 1].
template <class T>
ImageF rgbToHsl(const Image<T>& rgb) {
    detail::requireColor(rgb, "rgbToHsl");
    ImageF out = rgbToHsv(rgb);
    const double inv = 1.0 / PixelTraits<T>::max();
    detail::forEachPixel(rgb, out, [inv](const T* s, float* d) {
        const double r = s[0] * inv, g = s[1] * inv, b = s[2] * inv;
        const double mx = std::max({r, g, b}), mn = std::min({r, g, b});
        const double l = (mx + mn) / 2.0, delta = mx - mn;
        const double denom = 1.0 - std::fabs(2.0 * l - 1.0);
        d[1] = static_cast<float>(delta > 0 && denom > 1e-12 ? delta / denom : 0.0);
        d[2] = static_cast<float>(l);
    }, 8);
    return out;
}

/// HSL (as produced by rgbToHsl) -> RGB of pixel type T.
template <class T = std::uint8_t>
Image<T> hslToRgb(const ImageF& hsl) {
    detail::require(hsl.channels() == 3, "hslToRgb: expected a 3-channel HSL image");
    Image<T> out(hsl.width(), hsl.height(), 3);
    const double mx = PixelTraits<T>::max();
    detail::forEachPixel(hsl, out, [mx](const float* s, T* d) {
        double rgb[3];
        const double l = s[2], c = (1.0 - std::fabs(2.0 * l - 1.0)) * s[1];
        detail::hueToRgb(s[0], c, l - c / 2.0, rgb);
        for (int k = 0; k < 3; ++k) d[k] = saturate_cast<T>(clamp(rgb[k], 0.0, 1.0) * mx);
    }, 12);
    return out;
}

// ---------------------------------------------------------------------------
// YCbCr (full-range BT.601, as used by JPEG/JFIF)
// ---------------------------------------------------------------------------

/// RGB -> YCbCr. Chroma is offset by half the range (128 for uint8, 0.5 for float).
template <class T>
Image<T> rgbToYCbCr(const Image<T>& rgb) {
    detail::requireColor(rgb, "rgbToYCbCr");
    Image<T> out(rgb.width(), rgb.height(), 3);
    const double off = detail::chromaOffset<T>();
    detail::forEachPixel(rgb, out, [off](const T* s, T* d) {
        const double r = s[0], g = s[1], b = s[2];
        d[0] = saturate_cast<T>(0.299 * r + 0.587 * g + 0.114 * b);
        d[1] = saturate_cast<T>(off - 0.168735892 * r - 0.331264108 * g + 0.5 * b);
        d[2] = saturate_cast<T>(off + 0.5 * r - 0.418687589 * g - 0.081312411 * b);
    }, 4);
    return out;
}

/// YCbCr -> RGB (inverse of rgbToYCbCr).
template <class T>
Image<T> yCbCrToRgb(const Image<T>& ycc) {
    detail::require(ycc.channels() == 3, "yCbCrToRgb: expected a 3-channel image");
    Image<T> out(ycc.width(), ycc.height(), 3);
    const double off = detail::chromaOffset<T>();
    detail::forEachPixel(ycc, out, [off](const T* s, T* d) {
        const double y = s[0], cb = s[1] - off, cr = s[2] - off;
        d[0] = saturate_cast<T>(y + 1.402 * cr);
        d[1] = saturate_cast<T>(y - 0.344136286 * cb - 0.714136286 * cr);
        d[2] = saturate_cast<T>(y + 1.772 * cb);
        if constexpr (std::is_floating_point<T>::value)
            for (int k = 0; k < 3; ++k) d[k] = clamp(d[k], T(0), T(1));
    }, 4);
    return out;
}

// ---------------------------------------------------------------------------
// CIE XYZ and L*a*b* (sRGB, D65)
// ---------------------------------------------------------------------------

/// sRGB -> CIE XYZ (D65, Y of white = 1).
template <class T>
ImageF rgbToXyz(const Image<T>& rgb) {
    detail::requireColor(rgb, "rgbToXyz");
    ImageF out(rgb.width(), rgb.height(), 3);
    const double inv = 1.0 / PixelTraits<T>::max();
    detail::forEachPixel(rgb, out, [inv](const T* s, float* d) {
        const double r = detail::srgbToLinear(s[0] * inv), g = detail::srgbToLinear(s[1] * inv),
                     b = detail::srgbToLinear(s[2] * inv);
        d[0] = static_cast<float>(0.4124564 * r + 0.3575761 * g + 0.1804375 * b);
        d[1] = static_cast<float>(0.2126729 * r + 0.7151522 * g + 0.0721750 * b);
        d[2] = static_cast<float>(0.0193339 * r + 0.1191920 * g + 0.9503041 * b);
    }, 40);
    return out;
}

/// CIE XYZ (D65) -> sRGB of pixel type T (out-of-gamut colors are clipped).
template <class T = std::uint8_t>
Image<T> xyzToRgb(const ImageF& xyz) {
    detail::require(xyz.channels() == 3, "xyzToRgb: expected a 3-channel image");
    Image<T> out(xyz.width(), xyz.height(), 3);
    const double mx = PixelTraits<T>::max();
    detail::forEachPixel(xyz, out, [mx](const float* s, T* d) {
        const double x = s[0], y = s[1], z = s[2];
        const double lin[3] = {3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
                               -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
                               0.0556434 * x - 0.2040259 * y + 1.0572252 * z};
        for (int k = 0; k < 3; ++k)
            d[k] = saturate_cast<T>(clamp(detail::linearToSrgb(clamp(lin[k], 0.0, 1.0)), 0.0, 1.0) * mx);
    }, 40);
    return out;
}

namespace detail {
constexpr double kXn = 0.95047, kYn = 1.0, kZn = 1.08883;
inline double labF(double t) {
    const double d = 6.0 / 29.0;
    return t > d * d * d ? std::cbrt(t) : t / (3 * d * d) + 4.0 / 29.0;
}
inline double labFInv(double t) {
    const double d = 6.0 / 29.0;
    return t > d ? t * t * t : 3 * d * d * (t - 4.0 / 29.0);
}
}  // namespace detail

/// sRGB -> CIE L*a*b* (D65). L in [0, 100], a/b roughly in [-128, 127].
template <class T>
ImageF rgbToLab(const Image<T>& rgb) {
    ImageF xyz = rgbToXyz(rgb);
    ImageF lab(xyz.width(), xyz.height(), 3);
    detail::forEachPixel(xyz, lab, [](const float* p, float* d) {
        const double fx = detail::labF(p[0] / detail::kXn), fy = detail::labF(p[1] / detail::kYn),
                     fz = detail::labF(p[2] / detail::kZn);
        d[0] = static_cast<float>(116.0 * fy - 16.0);
        d[1] = static_cast<float>(500.0 * (fx - fy));
        d[2] = static_cast<float>(200.0 * (fy - fz));
    }, 30);
    return lab;
}

/// CIE L*a*b* (D65) -> sRGB of pixel type T.
template <class T = std::uint8_t>
Image<T> labToRgb(const ImageF& lab) {
    detail::require(lab.channels() == 3, "labToRgb: expected a 3-channel image");
    ImageF xyz(lab.width(), lab.height(), 3);
    detail::forEachPixel(lab, xyz, [](const float* p, float* d) {
        const double fy = (p[0] + 16.0) / 116.0, fx = fy + p[1] / 500.0, fz = fy - p[2] / 200.0;
        d[0] = static_cast<float>(detail::kXn * detail::labFInv(fx));
        d[1] = static_cast<float>(detail::kYn * detail::labFInv(fy));
        d[2] = static_cast<float>(detail::kZn * detail::labFInv(fz));
    }, 8);
    return xyzToRgb<T>(xyz);
}

/// CIE76 color difference (Euclidean distance in L*a*b*) between two Lab images.
inline ImageF deltaE76(const ImageF& lab1, const ImageF& lab2) {
    detail::require(lab1.sameShape(lab2) && lab1.channels() == 3, "deltaE76: images must be equal-size Lab images");
    ImageF out(lab1.width(), lab1.height(), 1);
    for (std::size_t i = 0; i < out.pixelCount(); ++i) {
        const float* a = lab1.data() + i * 3;
        const float* b = lab2.data() + i * 3;
        out.data()[i] = std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Color adjustments
// ---------------------------------------------------------------------------

/// Scales color saturation: 0 = grayscale, 1 = unchanged, >1 = more vivid.
/// Blends each pixel with its luma (same definition as PIL's ImageEnhance.Color). Alpha is kept.
template <class T>
Image<T> adjustSaturation(const Image<T>& rgb, double factor) {
    detail::requireColor(rgb, "adjustSaturation");
    Image<T> out = rgb;
    detail::forEachPixel(rgb, out, [factor](const T* s, T* d) {
        const double l = 0.299 * s[0] + 0.587 * s[1] + 0.114 * s[2];
        for (int k = 0; k < 3; ++k) d[k] = saturate_cast<T>(l + factor * (s[k] - l));
    }, 4);
    return out;
}

/// Rotates the hue of every pixel by `degrees` (alpha is kept).
template <class T>
Image<T> shiftHue(const Image<T>& rgb, double degrees) {
    detail::requireColor(rgb, "shiftHue");
    ImageF hsv = rgbToHsv(rgb);
    for (std::size_t i = 0; i < hsv.pixelCount(); ++i) {
        float& h = hsv.data()[i * 3];
        h = static_cast<float>(std::fmod(h + degrees + 360.0 * 16, 360.0));
    }
    Image<T> out = hsvToRgb<T>(hsv);
    if (rgb.channels() == 4) {
        out = convertChannels(out, 4);
        for (std::size_t i = 0; i < out.pixelCount(); ++i) out.data()[i * 4 + 3] = rgb.data()[i * 4 + 3];
    }
    return out;
}

/// Classic sepia tone (Microsoft's coefficients). Alpha is kept.
template <class T>
Image<T> sepia(const Image<T>& rgb) {
    detail::requireColor(rgb, "sepia");
    Image<T> out = rgb;
    const double mx = PixelTraits<T>::max();
    detail::forEachPixel(rgb, out, [mx](const T* s, T* d) {
        const double r = s[0], g = s[1], b = s[2];
        d[0] = saturate_cast<T>(std::min(mx, 0.393 * r + 0.769 * g + 0.189 * b));
        d[1] = saturate_cast<T>(std::min(mx, 0.349 * r + 0.686 * g + 0.168 * b));
        d[2] = saturate_cast<T>(std::min(mx, 0.272 * r + 0.534 * g + 0.131 * b));
    }, 4);
    return out;
}

// ---------------------------------------------------------------------------
// False-color maps
// ---------------------------------------------------------------------------

/// Viridis, Inferno, Magma, Plasma and Cividis are matplotlib's perceptually uniform maps;
/// Turbo is Google's improved rainbow map. All six use the exact 256-entry reference tables.
enum class Colormap { Gray, Jet, Hot, Viridis, Turbo, Inferno, Magma, Plasma, Cividis };

/// Color of a colormap at position t in [0, 1], as RGB in [0, 1] (tables are linearly interpolated).
inline std::array<double, 3> colormapColor(Colormap map, double t) {
    t = clamp(t, 0.0, 1.0);
    const std::uint8_t(*table)[3] = nullptr;
    switch (map) {
        case Colormap::Jet: {
            auto ramp = [](double x) { return clamp(1.5 - std::fabs(x), 0.0, 1.0); };
            return {ramp(4 * t - 3), ramp(4 * t - 2), ramp(4 * t - 1)};
        }
        case Colormap::Hot:
            return {clamp(t * 3, 0.0, 1.0), clamp(t * 3 - 1, 0.0, 1.0), clamp(t * 3 - 2, 0.0, 1.0)};
        case Colormap::Viridis: table = detail::kColormapViridis; break;
        case Colormap::Turbo: table = detail::kColormapTurbo; break;
        case Colormap::Inferno: table = detail::kColormapInferno; break;
        case Colormap::Magma: table = detail::kColormapMagma; break;
        case Colormap::Plasma: table = detail::kColormapPlasma; break;
        case Colormap::Cividis: table = detail::kColormapCividis; break;
        default: return {t, t, t};
    }
    const double pos = t * 255.0;
    const int i0 = std::min(254, static_cast<int>(pos));
    const double f = pos - i0;
    std::array<double, 3> c;
    for (int k = 0; k < 3; ++k) c[k] = (table[i0][k] * (1 - f) + table[i0 + 1][k] * f) / 255.0;
    return c;
}

/// Maps a single-channel image to an RGB uint8 false-color image. Values are scaled from
/// [lo, hi] to the colormap; when lo == hi (default) the image's min/max are used.
template <class T>
ImageU8 applyColormap(const Image<T>& img, Colormap map = Colormap::Viridis, double lo = 0, double hi = 0) {
    detail::require(img.channels() == 1, "applyColormap: expected a single-channel image");
    ImageU8 out(img.width(), img.height(), 3);
    if (img.empty()) return out;
    if (lo == hi) {
        const MinMaxResult mm = minMaxLoc(img);
        lo = mm.minVal;
        hi = mm.maxVal;
    }
    const double scale = hi > lo ? 1.0 / (hi - lo) : 0.0;
    std::array<std::array<std::uint8_t, 3>, 256> lut;
    for (int i = 0; i < 256; ++i) {
        const auto c = colormapColor(map, i / 255.0);
        for (int k = 0; k < 3; ++k) lut[i][k] = saturate_cast<std::uint8_t>(c[k] * 255.0);
    }
    detail::forEachPixel(img, out, [&](const T* s, std::uint8_t* d) {
        const double t = (static_cast<double>(*s) - lo) * scale;
        const auto& c = lut[static_cast<std::size_t>(clamp(t, 0.0, 1.0) * 255.0 + 0.5)];
        d[0] = c[0];
        d[1] = c[1];
        d[2] = c[2];
    }, 2);
    return out;
}

}  // namespace ipf
