// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf fft: fast Fourier transforms (any length), 2D DFT of images, frequency-domain
// filtering, FFT-based correlation and phase correlation (sub-pixel image registration).
//
// Spectra are stored as 2-channel double images (channel 0 = real, 1 = imaginary).
// The forward transform is unnormalized; the inverse divides by the number of samples,
// so idft(dft(img)) == img (as numpy.fft).
#pragma once

#include <complex>

#include "core.hpp"

namespace ipf {

using Complex = std::complex<double>;

namespace detail {

inline bool isPow2(std::size_t n) { return n && !(n & (n - 1)); }
inline std::size_t nextPow2(std::size_t n) {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/// In-place iterative radix-2 FFT (n must be a power of two). sign = -1 forward, +1 inverse.
inline void fftPow2(Complex* a, std::size_t n, int sign) {
    for (std::size_t i = 1, j = 0; i < n; ++i) {  // bit reversal permutation
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::size_t half = len / 2;
        // twiddles computed directly (not by repeated multiplication) for accuracy
        std::vector<Complex> w(half);
        for (std::size_t k = 0; k < half; ++k) {
            const double ang = sign * 2.0 * kPi * static_cast<double>(k) / static_cast<double>(len);
            w[k] = Complex(std::cos(ang), std::sin(ang));
        }
        for (std::size_t i = 0; i < n; i += len)
            for (std::size_t k = 0; k < half; ++k) {
                const Complex u = a[i + k], v = a[i + k + half] * w[k];
                a[i + k] = u + v;
                a[i + k + half] = u - v;
            }
    }
}

/// Bluestein's algorithm: FFT of arbitrary length via a power-of-two convolution.
inline void fftBluestein(Complex* a, std::size_t n, int sign) {
    const std::size_t m = nextPow2(2 * n - 1);
    std::vector<Complex> chirp(n), A(m), B(m);
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t k2 = (k * k) % (2 * n);  // keep the angle argument small for accuracy
        const double ang = sign * kPi * static_cast<double>(k2) / static_cast<double>(n);
        chirp[k] = Complex(std::cos(ang), std::sin(ang));
    }
    for (std::size_t k = 0; k < n; ++k) A[k] = a[k] * chirp[k];
    B[0] = std::conj(chirp[0]);
    for (std::size_t k = 1; k < n; ++k) B[k] = B[m - k] = std::conj(chirp[k]);
    fftPow2(A.data(), m, -1);
    fftPow2(B.data(), m, -1);
    for (std::size_t k = 0; k < m; ++k) A[k] *= B[k];
    fftPow2(A.data(), m, +1);
    for (std::size_t k = 0; k < n; ++k) a[k] = A[k] * chirp[k] / static_cast<double>(m);
}

inline void fftAny(Complex* a, std::size_t n, int sign) {
    if (n <= 1) return;
    if (isPow2(n)) fftPow2(a, n, sign);
    else fftBluestein(a, n, sign);
}

/// 2D FFT in place on a row-major W x H complex buffer (rows, then columns), unnormalized.
inline void fft2D(std::vector<Complex>& data, int W, int H, int sign) {
    parallelFor(0, H, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) fftAny(&data[static_cast<std::size_t>(y) * W], static_cast<std::size_t>(W), sign);
    }, static_cast<double>(W) * 20);
    parallelFor(0, W, [&](int x0, int x1) {
        std::vector<Complex> col(static_cast<std::size_t>(H));
        for (int x = x0; x < x1; ++x) {
            for (int y = 0; y < H; ++y) col[y] = data[static_cast<std::size_t>(y) * W + x];
            fftAny(col.data(), static_cast<std::size_t>(H), sign);
            for (int y = 0; y < H; ++y) data[static_cast<std::size_t>(y) * W + x] = col[y];
        }
    }, static_cast<double>(H) * 20);
}

}  // namespace detail

/// 1D FFT of any length (forward unnormalized; inverse = true divides by n).
inline std::vector<Complex> fft(std::vector<Complex> a, bool inverse = false) {
    detail::fftAny(a.data(), a.size(), inverse ? +1 : -1);
    if (inverse && !a.empty())
        for (Complex& v : a) v /= static_cast<double>(a.size());
    return a;
}

/// 2D DFT of a single-channel image (any size). Returns a 2-channel ImageD (re, im).
template <class T>
ImageD dft(const Image<T>& img) {
    detail::require(img.channels() == 1 || img.channels() == 2, "dft: expected a real (1-channel) or complex (2-channel) image");
    const int W = img.width(), H = img.height();
    std::vector<Complex> buf(img.pixelCount());
    for (std::size_t i = 0; i < buf.size(); ++i)
        buf[i] = img.channels() == 1 ? Complex(static_cast<double>(img.data()[i]), 0.0)
                                     : Complex(static_cast<double>(img.data()[2 * i]), static_cast<double>(img.data()[2 * i + 1]));
    detail::fft2D(buf, W, H, -1);
    ImageD out(W, H, 2);
    for (std::size_t i = 0; i < buf.size(); ++i) {
        out.data()[2 * i] = buf[i].real();
        out.data()[2 * i + 1] = buf[i].imag();
    }
    return out;
}

/// Inverse 2D DFT of a 2-channel spectrum (divides by W*H). Returns the real part as a
/// 1-channel image, or the full complex result (2 channels) with realOutput = false.
inline ImageD idft(const ImageD& spectrum, bool realOutput = true) {
    detail::require(spectrum.channels() == 2, "idft: expected a 2-channel (re, im) spectrum");
    const int W = spectrum.width(), H = spectrum.height();
    std::vector<Complex> buf(spectrum.pixelCount());
    for (std::size_t i = 0; i < buf.size(); ++i) buf[i] = Complex(spectrum.data()[2 * i], spectrum.data()[2 * i + 1]);
    detail::fft2D(buf, W, H, +1);
    const double s = 1.0 / static_cast<double>(buf.size());
    ImageD out(W, H, realOutput ? 1 : 2);
    for (std::size_t i = 0; i < buf.size(); ++i) {
        if (realOutput) {
            out.data()[i] = buf[i].real() * s;
        } else {
            out.data()[2 * i] = buf[i].real() * s;
            out.data()[2 * i + 1] = buf[i].imag() * s;
        }
    }
    return out;
}

/// Moves the zero-frequency component to the image center (numpy.fft.fftshift).
template <class T>
Image<T> fftShift(const Image<T>& img, bool inverse = false) {
    const int W = img.width(), H = img.height(), cn = img.channels();
    const int sx = inverse ? W - W / 2 : W / 2, sy = inverse ? H - H / 2 : H / 2;
    Image<T> out(W, H, cn);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) std::memcpy(out.ptr((x + sx) % W, (y + sy) % H), img.ptr(x, y), sizeof(T) * cn);
    return out;
}

/// log(1 + |F|) of the centered spectrum, for display (e.g. with applyColormap).
template <class T>
ImageF magnitudeSpectrum(const Image<T>& img) {
    const ImageD f = fftShift(dft(img));
    ImageF out(f.width(), f.height(), 1);
    for (std::size_t i = 0; i < out.pixelCount(); ++i)
        out.data()[i] = static_cast<float>(std::log1p(std::hypot(f.data()[2 * i], f.data()[2 * i + 1])));
    return out;
}

namespace detail {
/// Linear cross-correlation of every channel of `img` with `kernel` (single channel or
/// matching channels) for output positions (x, y) in [0, outW) x [0, outH):
/// out(x, y, c) = sum k(i, j, c) * img(x + i, y + j, c). Computed with FFTs.
template <class T, class K>
ImageD correlateFFT(const Image<T>& img, const Image<K>& kernel, int outW, int outH) {
    const int W = img.width(), H = img.height(), cn = img.channels(), kc = kernel.channels();
    const int fw = static_cast<int>(nextPow2(static_cast<std::size_t>(W)));
    const int fh = static_cast<int>(nextPow2(static_cast<std::size_t>(H)));
    ImageD out(outW, outH, cn, 0.0);
    std::vector<Complex> A(static_cast<std::size_t>(fw) * fh), B(A.size());
    for (int c = 0; c < cn; ++c) {
        // pack image (real part) and kernel (imaginary part) into one complex FFT
        std::fill(A.begin(), A.end(), Complex(0, 0));
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) A[static_cast<std::size_t>(y) * fw + x].real(static_cast<double>(img(x, y, c)));
        for (int y = 0; y < kernel.height(); ++y)
            for (int x = 0; x < kernel.width(); ++x)
                A[static_cast<std::size_t>(y) * fw + x].imag(static_cast<double>(kernel(x, y, kc == 1 ? 0 : c)));
        fft2D(A, fw, fh, -1);
        // separate the two real spectra: F_img = (Z(k) + conj Z(-k)) / 2, F_ker = (Z(k) - conj Z(-k)) / 2i,
        // then form F_img * conj(F_ker)
        for (int v = 0; v < fh; ++v)
            for (int u = 0; u < fw; ++u) {
                const Complex z = A[static_cast<std::size_t>(v) * fw + u];
                const Complex zc = std::conj(A[static_cast<std::size_t>((fh - v) % fh) * fw + (fw - u) % fw]);
                const Complex fi = (z + zc) * 0.5, fk = (z - zc) * Complex(0, -0.5);
                B[static_cast<std::size_t>(v) * fw + u] = fi * std::conj(fk);
            }
        fft2D(B, fw, fh, +1);
        const double s = 1.0 / (static_cast<double>(fw) * fh);
        for (int y = 0; y < outH; ++y)
            for (int x = 0; x < outW; ++x) out(x, y, c) = B[static_cast<std::size_t>(y) * fw + x].real() * s;
    }
    return out;
}
}  // namespace detail

/// filter2D computed in the frequency domain: same result as ipf::filter2D (any border mode),
/// much faster for large kernels (roughly beyond 15x15).
template <class TOut = void, class T>
Image<std::conditional_t<std::is_void<TOut>::value, T, TOut>> filter2DFFT(
    const Image<T>& src, const ImageF& kernel, BorderMode border = BorderMode::Reflect101, double delta = 0.0,
    double borderValue = 0.0) {
    using O = std::conditional_t<std::is_void<TOut>::value, T, TOut>;
    detail::require(kernel.channels() == 1 && !kernel.empty(), "filter2DFFT: kernel must be single-channel");
    const int kw = kernel.width(), kh = kernel.height(), ax = kw / 2, ay = kh / 2;
    const ImageD padded = copyMakeBorder(src.template convertTo<double>(), ay, kh - 1 - ay, ax, kw - 1 - ax, border,
                                         borderValue);
    const ImageD r = detail::correlateFFT(padded, kernel, src.width(), src.height());
    Image<O> out(src.width(), src.height(), src.channels());
    for (std::size_t i = 0; i < out.sampleCount(); ++i) out.data()[i] = saturate_cast<O>(r.data()[i] + delta);
    return out;
}

struct PhaseCorrelation {
    PointF shift;     ///< translation of `b` relative to `a` (b(x, y) ~ a(x - shift.x, y - shift.y))
    double response;  ///< peak height in [0, 1]; higher = more reliable
};

/// Estimates the translation between two equally sized single-channel images with sub-pixel
/// accuracy (normalized cross-power spectrum; a Hann window reduces edge effects).
template <class T>
PhaseCorrelation phaseCorrelate(const Image<T>& a, const Image<T>& b, bool hannWindow = true) {
    detail::require(a.sameShape(b) && a.channels() == 1 && !a.empty(), "phaseCorrelate: need equal-size single-channel images");
    const int W = a.width(), H = a.height();
    ImageD wa(W, H, 1), wb(W, H, 1);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            double w = 1;
            if (hannWindow)
                w = (0.5 - 0.5 * std::cos(2 * detail::kPi * x / std::max(1, W - 1))) *
                    (0.5 - 0.5 * std::cos(2 * detail::kPi * y / std::max(1, H - 1)));
            wa(x, y) = w * static_cast<double>(a(x, y));
            wb(x, y) = w * static_cast<double>(b(x, y));
        }
    const ImageD fa = dft(wa), fb = dft(wb);
    ImageD cross(W, H, 2);
    for (std::size_t i = 0; i < cross.pixelCount(); ++i) {
        const Complex p = Complex(fb.data()[2 * i], fb.data()[2 * i + 1]) * std::conj(Complex(fa.data()[2 * i], fa.data()[2 * i + 1]));
        const double m = std::abs(p);
        const Complex q = m > 1e-15 ? p / m : Complex(0, 0);
        cross.data()[2 * i] = q.real();
        cross.data()[2 * i + 1] = q.imag();
    }
    const ImageD r = idft(cross);
    const MinMaxResult mm = minMaxLoc(r);
    // sub-pixel refinement: weighted centroid of the 3x3 neighbourhood (with wrap-around)
    double sw = 0, sx = 0, sy = 0;
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i) {
            const double v = std::max(0.0, r((mm.maxLoc.x + i + W) % W, (mm.maxLoc.y + j + H) % H));
            sw += v;
            sx += v * i;
            sy += v * j;
        }
    double dx = mm.maxLoc.x + (sw > 0 ? sx / sw : 0), dy = mm.maxLoc.y + (sw > 0 ? sy / sw : 0);
    if (dx > W / 2.0) dx -= W;
    if (dy > H / 2.0) dy -= H;
    return {PointF(dx, dy), mm.maxVal};
}

}  // namespace ipf
