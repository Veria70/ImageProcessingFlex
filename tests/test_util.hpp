// Shared helpers for the ipf unit tests.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>

#include <ipf/core.hpp>

namespace ipftest {

/// Uniform random image covering the full nominal range of T.
template <class T>
ipf::Image<T> randomImage(int w, int h, int c, unsigned seed = 1) {
    std::mt19937 rng(seed);
    ipf::Image<T> img(w, h, c);
    for (auto& v : img) {
        if constexpr (std::is_floating_point<T>::value)
            v = static_cast<T>(std::uniform_real_distribution<double>(0.0, 1.0)(rng));
        else
            v = static_cast<T>(std::uniform_int_distribution<long long>(0, static_cast<long long>(ipf::PixelTraits<T>::max()))(rng));
    }
    return img;
}

/// Smooth "natural-looking" 8-bit test image (gradients + low-frequency waves).
inline ipf::ImageU8 smoothImage(int w, int h, int c) {
    ipf::ImageU8 img(w, h, c);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int k = 0; k < c; ++k) {
                const double v = 128 + 60 * std::sin(x * 0.07 + k) + 50 * std::cos(y * 0.05 - k * 0.5) +
                                 10 * std::sin((x + y) * 0.21);
                img(x, y, k) = ipf::saturate_cast<std::uint8_t>(v);
            }
    return img;
}

template <class T>
double psnrOf(const ipf::Image<T>& a, const ipf::Image<T>& b) {
    double mse = 0;
    for (std::size_t i = 0; i < a.sampleCount(); ++i) {
        const double d = static_cast<double>(a.data()[i]) - static_cast<double>(b.data()[i]);
        mse += d * d;
    }
    mse /= static_cast<double>(a.sampleCount());
    const double peak = ipf::PixelTraits<T>::max();
    return mse == 0 ? 999.0 : 10.0 * std::log10(peak * peak / mse);
}

template <class T>
double maxAbsDiff(const ipf::Image<T>& a, const ipf::Image<T>& b) {
    double m = 0;
    for (std::size_t i = 0; i < a.sampleCount(); ++i)
        m = std::max(m, std::fabs(static_cast<double>(a.data()[i]) - static_cast<double>(b.data()[i])));
    return m;
}

}  // namespace ipftest
