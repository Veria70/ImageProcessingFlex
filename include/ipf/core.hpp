// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ImageProcessingFlex (ipf) - header-only C++17 image processing library.
// core.hpp: the Image container, small geometry types, border handling,
// saturating casts, channel utilities and basic statistics.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "parallel.hpp"

#define IPF_VERSION_MAJOR 1
#define IPF_VERSION_MINOR 1
#define IPF_VERSION_PATCH 0

namespace ipf {

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// All errors raised by the library are reported with this exception type.
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace detail {
[[noreturn]] inline void fail(const std::string& msg) { throw Error("ipf: " + msg); }
inline void require(bool cond, const char* msg) {
    if (!cond) fail(msg);
}
constexpr double kPi = 3.14159265358979323846;
}  // namespace detail

// ---------------------------------------------------------------------------
// Pixel value traits and saturating conversion
// ---------------------------------------------------------------------------

/// Nominal value range of a pixel type: [0, 255] for uint8, [0, 65535] for
/// uint16 and [0, 1] for floating point images.
template <class T>
struct PixelTraits {
    static constexpr bool is_float = std::is_floating_point<T>::value;
    static constexpr double max() {
        if constexpr (std::is_floating_point<T>::value)
            return 1.0;
        else if constexpr (std::is_signed<T>::value && sizeof(T) >= 4)
            return 255.0;  // int32 images are usually labels/accumulators
        else
            return static_cast<double>(std::numeric_limits<T>::max());
    }
};

/// Converts a value to type To, rounding to nearest and clamping to the range
/// of To when To is an integer type (NaN maps to 0).
template <class To, class From>
constexpr To saturate_cast(From v) {
    if constexpr (std::is_same<To, From>::value) {
        return v;
    } else if constexpr (std::is_integral<To>::value && std::is_floating_point<From>::value) {
        if (!(v == v)) return To(0);  // NaN
        const double lo = static_cast<double>(std::numeric_limits<To>::lowest());
        const double hi = static_cast<double>(std::numeric_limits<To>::max());
        double d = static_cast<double>(v);
        d = d < 0 ? d - 0.5 : d + 0.5;  // round half away from zero
        if (d <= lo) return std::numeric_limits<To>::lowest();
        if (d >= hi) return std::numeric_limits<To>::max();
        return static_cast<To>(d);
    } else if constexpr (std::is_integral<To>::value && std::is_integral<From>::value) {
        using Wide = std::conditional_t<std::is_signed<From>::value || std::is_signed<To>::value,
                                        long long, unsigned long long>;
        if constexpr (std::is_signed<From>::value && !std::is_signed<To>::value) {
            if (v < 0) return To(0);
            if (static_cast<unsigned long long>(v) > static_cast<unsigned long long>(std::numeric_limits<To>::max()))
                return std::numeric_limits<To>::max();
            return static_cast<To>(v);
        } else if constexpr (!std::is_signed<From>::value && std::is_signed<To>::value) {
            if (static_cast<unsigned long long>(v) > static_cast<unsigned long long>(std::numeric_limits<To>::max()))
                return std::numeric_limits<To>::max();
            return static_cast<To>(v);
        } else {
            const Wide w = static_cast<Wide>(v);
            if (w < static_cast<Wide>(std::numeric_limits<To>::lowest())) return std::numeric_limits<To>::lowest();
            if (w > static_cast<Wide>(std::numeric_limits<To>::max())) return std::numeric_limits<To>::max();
            return static_cast<To>(v);
        }
    } else {
        return static_cast<To>(v);
    }
}

/// Clamps v to [lo, hi].
template <class T>
constexpr T clamp(T v, T lo, T hi) {
    return v < lo ? lo : (hi < v ? hi : v);
}

// ---------------------------------------------------------------------------
// Small geometry types
// ---------------------------------------------------------------------------

struct Point {
    int x = 0, y = 0;
    constexpr Point() = default;
    constexpr Point(int x_, int y_) : x(x_), y(y_) {}
    friend constexpr bool operator==(Point a, Point b) { return a.x == b.x && a.y == b.y; }
    friend constexpr bool operator!=(Point a, Point b) { return !(a == b); }
    friend constexpr Point operator+(Point a, Point b) { return {a.x + b.x, a.y + b.y}; }
    friend constexpr Point operator-(Point a, Point b) { return {a.x - b.x, a.y - b.y}; }
};

struct PointF {
    double x = 0, y = 0;
    constexpr PointF() = default;
    constexpr PointF(double x_, double y_) : x(x_), y(y_) {}
    constexpr PointF(Point p) : x(p.x), y(p.y) {}
    friend constexpr PointF operator+(PointF a, PointF b) { return {a.x + b.x, a.y + b.y}; }
    friend constexpr PointF operator-(PointF a, PointF b) { return {a.x - b.x, a.y - b.y}; }
    friend constexpr PointF operator*(PointF a, double s) { return {a.x * s, a.y * s}; }
    friend constexpr bool operator==(PointF a, PointF b) { return a.x == b.x && a.y == b.y; }
};

struct Size {
    int width = 0, height = 0;
    constexpr Size() = default;
    constexpr Size(int w, int h) : width(w), height(h) {}
    friend constexpr bool operator==(Size a, Size b) { return a.width == b.width && a.height == b.height; }
    friend constexpr bool operator!=(Size a, Size b) { return !(a == b); }
};

struct Rect {
    int x = 0, y = 0, width = 0, height = 0;
    constexpr Rect() = default;
    constexpr Rect(int x_, int y_, int w, int h) : x(x_), y(y_), width(w), height(h) {}
    constexpr int right() const { return x + width; }    // exclusive
    constexpr int bottom() const { return y + height; }  // exclusive
    constexpr long long area() const { return static_cast<long long>(width) * height; }
    constexpr bool empty() const { return width <= 0 || height <= 0; }
    constexpr bool contains(Point p) const { return p.x >= x && p.y >= y && p.x < right() && p.y < bottom(); }
    /// Intersection of two rectangles (empty rect if they do not overlap).
    constexpr Rect operator&(const Rect& o) const {
        const int x0 = std::max(x, o.x), y0 = std::max(y, o.y);
        const int x1 = std::min(right(), o.right()), y1 = std::min(bottom(), o.bottom());
        return (x1 > x0 && y1 > y0) ? Rect(x0, y0, x1 - x0, y1 - y0) : Rect();
    }
    /// Smallest rectangle containing both rectangles.
    constexpr Rect operator|(const Rect& o) const {
        if (empty()) return o;
        if (o.empty()) return *this;
        const int x0 = std::min(x, o.x), y0 = std::min(y, o.y);
        const int x1 = std::max(right(), o.right()), y1 = std::max(bottom(), o.bottom());
        return Rect(x0, y0, x1 - x0, y1 - y0);
    }
    friend constexpr bool operator==(const Rect& a, const Rect& b) {
        return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
    }
    friend constexpr bool operator!=(const Rect& a, const Rect& b) { return !(a == b); }
};

// ---------------------------------------------------------------------------
// Border handling
// ---------------------------------------------------------------------------

/// How pixels outside the image are synthesized (names follow OpenCV):
///   Constant   : iiiiii|abcdefgh|iiiiiii  (uses a border value)
///   Replicate  : aaaaaa|abcdefgh|hhhhhhh
///   Reflect    : fedcba|abcdefgh|hgfedcb
///   Reflect101 : gfedcb|abcdefgh|gfedcba  (default for filters)
///   Wrap       : cdefgh|abcdefgh|abcdefg
enum class BorderMode { Constant, Replicate, Reflect, Reflect101, Wrap };

/// Maps a possibly out-of-range coordinate p into [0, len) according to the
/// border mode. Returns -1 for BorderMode::Constant when p is outside.
inline int borderIndex(int p, int len, BorderMode mode) {
    if (p >= 0 && p < len) return p;
    if (len <= 0) return -1;
    switch (mode) {
        case BorderMode::Constant:
            return -1;
        case BorderMode::Replicate:
            return p < 0 ? 0 : len - 1;
        case BorderMode::Reflect: {
            const long long period = 2LL * len;
            long long q = p % period;
            if (q < 0) q += period;
            return static_cast<int>(q < len ? q : period - 1 - q);
        }
        case BorderMode::Reflect101: {
            if (len == 1) return 0;
            const long long period = 2LL * len - 2;
            long long q = p % period;
            if (q < 0) q += period;
            return static_cast<int>(q < len ? q : period - q);
        }
        case BorderMode::Wrap: {
            int q = p % len;
            return q < 0 ? q + len : q;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Image container
// ---------------------------------------------------------------------------

/// A dense 2D image with interleaved channels stored row-major
/// (pixel (x, y) channel c lives at data()[(y * width + x) * channels + c]).
/// This is the same layout as a numpy array of shape (height, width, channels).
/// Images have value semantics: copying an Image copies its pixels.
template <class T>
class Image {
    static_assert(std::is_arithmetic<T>::value, "ipf::Image pixel type must be arithmetic");

public:
    using value_type = T;

    Image() = default;

    /// Creates a width x height image with the given channel count, filled with `fill`.
    Image(int width, int height, int channels = 1, T fill = T{}) { reset(width, height, channels, fill); }

    /// Creates an image by copying width*height*channels interleaved values from `src`
    /// (e.g. a buffer from another library). `src` must not be null for non-empty images.
    static Image fromData(int width, int height, int channels, const T* src) {
        Image img(width, height, channels);
        if (!img.data_.empty()) {
            detail::require(src != nullptr, "Image::fromData: null source pointer");
            std::memcpy(img.data_.data(), src, img.data_.size() * sizeof(T));
        }
        return img;
    }

    /// Creates an image taking ownership of a pixel vector of the right size.
    Image(int width, int height, int channels, std::vector<T> pixels) {
        checkDims(width, height, channels);
        detail::require(pixels.size() == static_cast<std::size_t>(width) * height * channels,
                        "Image: pixel vector size does not match dimensions");
        w_ = width;
        h_ = height;
        c_ = channels;
        data_ = std::move(pixels);
    }

    /// Re-allocates the image with new dimensions; all pixels are set to `fill`.
    void reset(int width, int height, int channels = 1, T fill = T{}) {
        checkDims(width, height, channels);
        w_ = width;
        h_ = height;
        c_ = channels;
        data_.assign(static_cast<std::size_t>(width) * height * channels, fill);
    }

    int width() const { return w_; }
    int height() const { return h_; }
    int channels() const { return c_; }
    Size size() const { return {w_, h_}; }
    Rect rect() const { return {0, 0, w_, h_}; }
    bool empty() const { return data_.empty(); }
    std::size_t pixelCount() const { return static_cast<std::size_t>(w_) * h_; }
    std::size_t sampleCount() const { return data_.size(); }
    std::size_t stride() const { return static_cast<std::size_t>(w_) * c_; }  // samples per row

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    std::vector<T>& vector() { return data_; }
    const std::vector<T>& vector() const { return data_; }

    T* row(int y) { return data_.data() + static_cast<std::size_t>(y) * stride(); }
    const T* row(int y) const { return data_.data() + static_cast<std::size_t>(y) * stride(); }
    T* ptr(int x, int y) { return row(y) + static_cast<std::size_t>(x) * c_; }
    const T* ptr(int x, int y) const { return row(y) + static_cast<std::size_t>(x) * c_; }

    /// Unchecked access.
    T& operator()(int x, int y, int c = 0) { return data_[index(x, y, c)]; }
    const T& operator()(int x, int y, int c = 0) const { return data_[index(x, y, c)]; }

    /// Bounds-checked access (throws ipf::Error).
    T& at(int x, int y, int c = 0) {
        checkBounds(x, y, c);
        return data_[index(x, y, c)];
    }
    const T& at(int x, int y, int c = 0) const {
        checkBounds(x, y, c);
        return data_[index(x, y, c)];
    }

    bool inside(int x, int y) const { return x >= 0 && y >= 0 && x < w_ && y < h_; }

    /// Reads a sample with border extrapolation for out-of-range coordinates.
    T atBorder(int x, int y, int c, BorderMode mode, T borderValue = T{}) const {
        const int xx = borderIndex(x, w_, mode);
        const int yy = borderIndex(y, h_, mode);
        if (xx < 0 || yy < 0) return borderValue;
        return data_[index(xx, yy, c)];
    }

    /// Sets every sample to v.
    void fill(T v) { std::fill(data_.begin(), data_.end(), v); }

    /// Sets every pixel to the given per-channel color (size must equal channels()).
    void fill(std::initializer_list<T> color) { fill(std::vector<T>(color)); }
    void fill(const std::vector<T>& color) {
        detail::require(static_cast<int>(color.size()) == c_, "Image::fill: color size != channels");
        for (std::size_t i = 0; i < data_.size(); i += c_)
            for (int c = 0; c < c_; ++c) data_[i + c] = color[c];
    }

    /// Writes a pixel (all channels) from an array of channels() values.
    void setPixel(int x, int y, const T* color) {
        T* p = ptr(x, y);
        for (int c = 0; c < c_; ++c) p[c] = color[c];
    }

    /// Returns a copy of the given region; the region is clipped to the image.
    Image crop(Rect r) const {
        r = r & rect();
        if (r.empty()) return Image();
        Image out(r.width, r.height, c_);
        for (int y = 0; y < r.height; ++y)
            std::memcpy(out.row(y), ptr(r.x, r.y + y), static_cast<std::size_t>(r.width) * c_ * sizeof(T));
        return out;
    }

    /// Copies `src` into this image with its top-left corner at (x, y).
    /// Parts falling outside are clipped. Channel counts must match.
    void paste(const Image& src, int x, int y) {
        detail::require(src.channels() == c_, "Image::paste: channel count mismatch");
        const Rect dst = Rect(x, y, src.width(), src.height()) & rect();
        for (int yy = dst.y; yy < dst.bottom(); ++yy)
            std::memcpy(ptr(dst.x, yy), src.ptr(dst.x - x, yy - y),
                        static_cast<std::size_t>(dst.width) * c_ * sizeof(T));
    }

    bool sameShape(int w, int h, int c) const { return w_ == w && h_ == h && c_ == c; }
    template <class U>
    bool sameShape(const Image<U>& o) const {
        return w_ == o.width() && h_ == o.height() && c_ == o.channels();
    }

    /// Converts to another pixel type: out = saturate(in * scale + offset).
    template <class U>
    Image<U> convertTo(double scale = 1.0, double offset = 0.0) const {
        Image<U> out(w_, h_, c_);
        U* d = out.data();
        const T* s = data_.data();
        parallelForN(data_.size(), [&](std::size_t i0, std::size_t i1) {
            if (scale == 1.0 && offset == 0.0) {
                for (std::size_t i = i0; i < i1; ++i) d[i] = saturate_cast<U>(s[i]);
            } else {
                for (std::size_t i = i0; i < i1; ++i) d[i] = saturate_cast<U>(static_cast<double>(s[i]) * scale + offset);
            }
        }, 0.5);
        return out;
    }

    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }

    friend bool operator==(const Image& a, const Image& b) {
        return a.w_ == b.w_ && a.h_ == b.h_ && a.c_ == b.c_ && a.data_ == b.data_;
    }
    friend bool operator!=(const Image& a, const Image& b) { return !(a == b); }

private:
    static void checkDims(int w, int h, int c) {
        detail::require(w >= 0 && h >= 0, "Image: negative dimensions");
        detail::require(c >= 1, "Image: channel count must be >= 1");
        const unsigned long long n = static_cast<unsigned long long>(w) * static_cast<unsigned long long>(h) *
                                     static_cast<unsigned long long>(c);
        detail::require(n <= (std::numeric_limits<std::size_t>::max() / sizeof(T)) && n < (1ULL << 40),
                        "Image: dimensions too large");
    }
    std::size_t index(int x, int y, int c) const {
        return (static_cast<std::size_t>(y) * w_ + x) * c_ + c;
    }
    void checkBounds(int x, int y, int c) const {
        if (x < 0 || y < 0 || x >= w_ || y >= h_ || c < 0 || c >= c_)
            detail::fail("Image::at: coordinate (" + std::to_string(x) + ", " + std::to_string(y) + ", " +
                         std::to_string(c) + ") out of range for " + std::to_string(w_) + "x" +
                         std::to_string(h_) + "x" + std::to_string(c_) + " image");
    }

    int w_ = 0, h_ = 0, c_ = 1;
    std::vector<T> data_;
};

using ImageU8 = Image<std::uint8_t>;
using ImageU16 = Image<std::uint16_t>;
using ImageI16 = Image<std::int16_t>;
using ImageI32 = Image<std::int32_t>;
using ImageF = Image<float>;
using ImageD = Image<double>;

// ---------------------------------------------------------------------------
// Conversions between pixel types with range normalization
// ---------------------------------------------------------------------------

/// Converts between pixel types, rescaling the nominal range
/// (e.g. uint8 [0,255] -> float [0,1] -> uint16 [0,65535]).
template <class U, class T>
Image<U> convertRange(const Image<T>& src) {
    const double scale = PixelTraits<U>::max() / PixelTraits<T>::max();
    return src.template convertTo<U>(scale, 0.0);
}

/// uint8/uint16/... -> float in [0, 1].
template <class T>
ImageF toFloat(const Image<T>& src) {
    return convertRange<float>(src);
}

/// float in [0, 1] (or any type) -> uint8 in [0, 255], with saturation.
template <class T>
ImageU8 toU8(const Image<T>& src) {
    return convertRange<std::uint8_t>(src);
}

/// Applies fn to every sample, producing an image of fn's return type.
template <class T, class Fn>
auto mapSamples(const Image<T>& src, Fn fn) -> Image<std::decay_t<decltype(fn(std::declval<T>()))>> {
    using U = std::decay_t<decltype(fn(std::declval<T>()))>;
    Image<U> out(src.width(), src.height(), src.channels());
    const T* s = src.data();
    U* d = out.data();
    for (std::size_t i = 0; i < src.sampleCount(); ++i) d[i] = fn(s[i]);  // serial: fn may have state
    return out;
}

namespace detail {
/// Calls fn(const T* srcPixel, U* dstPixel) for every pixel (in parallel for large images).
/// src and dst must have the same width and height; costPerPixel guides the threading decision.
template <class T, class U, class Fn>
void forEachPixel(const Image<T>& src, Image<U>& dst, Fn fn, double costPerPixel = 4.0) {
    const std::size_t sc = static_cast<std::size_t>(src.channels()), dc = static_cast<std::size_t>(dst.channels());
    const T* s = src.data();
    U* d = dst.data();
    parallelForN(src.pixelCount(), [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i) fn(s + i * sc, d + i * dc);
    }, costPerPixel);
}
}  // namespace detail

// ---------------------------------------------------------------------------
// Channel utilities
// ---------------------------------------------------------------------------

/// Extracts one channel as a single-channel image.
template <class T>
Image<T> extractChannel(const Image<T>& src, int channel) {
    detail::require(channel >= 0 && channel < src.channels(), "extractChannel: channel out of range");
    Image<T> out(src.width(), src.height(), 1);
    const int cn = src.channels();
    const T* s = src.data() + channel;
    T* d = out.data();
    for (std::size_t i = 0; i < src.pixelCount(); ++i) d[i] = s[i * cn];
    return out;
}

/// Overwrites one channel of dst with the single-channel image src.
template <class T>
void insertChannel(Image<T>& dst, const Image<T>& src, int channel) {
    detail::require(channel >= 0 && channel < dst.channels(), "insertChannel: channel out of range");
    detail::require(src.channels() == 1 && src.width() == dst.width() && src.height() == dst.height(),
                    "insertChannel: src must be single-channel and the same size as dst");
    const int cn = dst.channels();
    T* d = dst.data() + channel;
    const T* s = src.data();
    for (std::size_t i = 0; i < src.pixelCount(); ++i) d[i * cn] = s[i];
}

/// Splits a multi-channel image into single-channel planes.
template <class T>
std::vector<Image<T>> splitChannels(const Image<T>& src) {
    std::vector<Image<T>> planes;
    planes.reserve(src.channels());
    for (int c = 0; c < src.channels(); ++c) planes.push_back(extractChannel(src, c));
    return planes;
}

/// Merges single-channel planes of identical size into one interleaved image.
template <class T>
Image<T> mergeChannels(const std::vector<Image<T>>& planes) {
    detail::require(!planes.empty(), "mergeChannels: no planes given");
    const int w = planes[0].width(), h = planes[0].height();
    for (const auto& p : planes)
        detail::require(p.channels() == 1 && p.width() == w && p.height() == h,
                        "mergeChannels: planes must be single-channel and equally sized");
    Image<T> out(w, h, static_cast<int>(planes.size()));
    for (int c = 0; c < static_cast<int>(planes.size()); ++c) insertChannel(out, planes[c], c);
    return out;
}

/// Reorders channels: out channel i = src channel order[i]. E.g. {2,1,0} swaps RGB<->BGR.
template <class T>
Image<T> reorderChannels(const Image<T>& src, const std::vector<int>& order) {
    for (int o : order) detail::require(o >= 0 && o < src.channels(), "reorderChannels: invalid channel index");
    detail::require(!order.empty(), "reorderChannels: empty order");
    Image<T> out(src.width(), src.height(), static_cast<int>(order.size()));
    const int sc = src.channels(), dc = out.channels();
    const T* s = src.data();
    T* d = out.data();
    for (std::size_t i = 0; i < src.pixelCount(); ++i)
        for (int c = 0; c < dc; ++c) d[i * dc + c] = s[i * sc + order[c]];
    return out;
}

/// Pads an image by the given number of pixels on each side, synthesizing the new
/// pixels with the border mode (Constant uses `value` for every channel).
template <class T>
Image<T> copyMakeBorder(const Image<T>& src, int top, int bottom, int left, int right, BorderMode mode,
                        T value = T{}) {
    detail::require(top >= 0 && bottom >= 0 && left >= 0 && right >= 0, "copyMakeBorder: negative padding");
    detail::require(!src.empty() || mode == BorderMode::Constant, "copyMakeBorder: empty source image");
    const int W = src.width(), H = src.height(), cn = src.channels();
    Image<T> out(W + left + right, H + top + bottom, cn, value);
    std::vector<int> xs(static_cast<std::size_t>(out.width()));
    for (int x = 0; x < out.width(); ++x) xs[x] = borderIndex(x - left, W, mode);
    for (int y = 0; y < out.height(); ++y) {
        const int sy = borderIndex(y - top, H, mode);
        if (sy < 0) continue;
        T* d = out.row(y);
        const T* s = src.row(sy);
        for (int x = 0; x < out.width(); ++x)
            if (xs[x] >= 0)
                for (int c = 0; c < cn; ++c) d[x * cn + c] = s[xs[x] * cn + c];
    }
    return out;
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

struct MinMaxResult {
    double minVal = 0, maxVal = 0;
    Point minLoc, maxLoc;
};

/// Minimum and maximum value (and their first locations in raster order).
/// For multi-channel images pass the channel to inspect, or -1 to scan all samples.
template <class T>
MinMaxResult minMaxLoc(const Image<T>& img, int channel = -1) {
    detail::require(!img.empty(), "minMaxLoc: empty image");
    detail::require(channel >= -1 && channel < img.channels(), "minMaxLoc: invalid channel");
    MinMaxResult r;
    bool first = true;
    const int cn = img.channels();
    for (int y = 0; y < img.height(); ++y) {
        const T* row = img.row(y);
        for (int x = 0; x < img.width(); ++x) {
            for (int c = (channel < 0 ? 0 : channel); c < (channel < 0 ? cn : channel + 1); ++c) {
                const double v = static_cast<double>(row[x * cn + c]);
                if (first || v < r.minVal) { r.minVal = v; r.minLoc = {x, y}; }
                if (first || v > r.maxVal) { r.maxVal = v; r.maxLoc = {x, y}; }
                first = false;
            }
        }
    }
    return r;
}

/// Per-channel mean.
template <class T>
std::vector<double> mean(const Image<T>& img) {
    std::vector<double> sum(img.channels(), 0.0);
    const int cn = img.channels();
    const T* p = img.data();
    for (std::size_t i = 0; i < img.pixelCount(); ++i)
        for (int c = 0; c < cn; ++c) sum[c] += static_cast<double>(p[i * cn + c]);
    if (img.pixelCount() > 0)
        for (double& s : sum) s /= static_cast<double>(img.pixelCount());
    return sum;
}

/// Per-channel population standard deviation.
template <class T>
std::vector<double> stddev(const Image<T>& img) {
    const std::vector<double> m = mean(img);
    std::vector<double> acc(img.channels(), 0.0);
    const int cn = img.channels();
    const T* p = img.data();
    for (std::size_t i = 0; i < img.pixelCount(); ++i)
        for (int c = 0; c < cn; ++c) {
            const double d = static_cast<double>(p[i * cn + c]) - m[c];
            acc[c] += d * d;
        }
    for (double& a : acc) a = img.pixelCount() > 0 ? std::sqrt(a / static_cast<double>(img.pixelCount())) : 0.0;
    return acc;
}

/// Per-channel sum.
template <class T>
std::vector<double> sum(const Image<T>& img) {
    std::vector<double> s = mean(img);
    for (double& v : s) v *= static_cast<double>(img.pixelCount());
    return s;
}

/// Number of non-zero samples (use on single-channel masks).
template <class T>
std::size_t countNonZero(const Image<T>& img) {
    return static_cast<std::size_t>(std::count_if(img.begin(), img.end(), [](T v) { return v != T(0); }));
}

}  // namespace ipf
