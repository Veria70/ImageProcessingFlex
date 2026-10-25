#include <ipf/draw.hpp>
#include <ipf/features.hpp>
#include <ipf/fft.hpp>
#include <ipf/geometry.hpp>

#include "test_framework.hpp"
#include "test_util.hpp"

using namespace ipf;
using namespace ipftest;

TEST(fft_1d_matches_naive_dft) {
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> u(-1, 1);
    for (std::size_t n : {1u, 2u, 3u, 5u, 7u, 8u, 12u, 17u, 64u, 100u, 127u, 256u}) {
        std::vector<Complex> a(n);
        for (auto& v : a) v = Complex(u(rng), u(rng));
        const std::vector<Complex> f = fft(a);
        double err = 0;
        for (std::size_t k = 0; k < n; ++k) {
            Complex s(0, 0);
            for (std::size_t j = 0; j < n; ++j) s += a[j] * std::polar(1.0, -2 * detail::kPi * static_cast<double>(j * k) / static_cast<double>(n));
            err = std::max(err, std::abs(s - f[k]));
        }
        CHECK(err < 1e-9 * static_cast<double>(n));
        const std::vector<Complex> back = fft(f, true);
        for (std::size_t k = 0; k < n; ++k) CHECK(std::abs(back[k] - a[k]) < 1e-12 * static_cast<double>(n));
    }
}

TEST(fft_2d_roundtrip_and_shift) {
    const ImageF img = randomImage<float>(37, 22, 1, 3);
    const ImageD spec = dft(img);
    CHECK_NEAR(spec(0, 0, 0), sum(img)[0], 1e-6);  // DC term = sum
    const ImageD back = idft(spec);
    CHECK(maxAbsDiff(back, img.convertTo<double>()) < 1e-9);
    ImageI32 idx(4, 3, 1);
    for (int i = 0; i < 12; ++i) idx.data()[i] = i;
    const ImageI32 sh = fftShift(idx);
    CHECK_EQ(sh(2, 1), 0);  // origin moves to the center
    CHECK(fftShift(sh, true) == idx);
    const ImageF ms = magnitudeSpectrum(img);
    CHECK(ms.sameShape(img));
}

TEST(fft_filter2d_equals_spatial) {
    const ImageF img = randomImage<float>(53, 41, 2, 9);
    const Kernel k = gaussianKernel2D(3.0);
    for (BorderMode m : {BorderMode::Constant, BorderMode::Replicate, BorderMode::Reflect, BorderMode::Reflect101, BorderMode::Wrap}) {
        const ImageF a = filter2D(img, k, m, 0.5, Point(-1, -1), 0.25);
        const ImageF b = filter2DFFT(img, k, m, 0.5, 0.25);
        CHECK(maxAbsDiff(a, b) < 1e-5);
    }
    const ImageU8 u = randomImage<std::uint8_t>(40, 30, 3, 1);
    CHECK(maxAbsDiff(filter2D(u, k), filter2DFFT(u, k)) <= 1.0);
}

TEST(fft_template_matching_large_template) {
    const ImageU8 img = smoothImage(320, 220, 1);
    ImageU8 noisy = add(img, randomImage<std::uint8_t>(320, 220, 1, 4));
    const ImageU8 templ = noisy.crop(Rect(150, 90, 70, 60));  // large template -> FFT path
    for (MatchMethod m : {MatchMethod::SqDiff, MatchMethod::SqDiffNormed, MatchMethod::CCorrNormed, MatchMethod::CCoeffNormed}) {
        const MatchResult r = findTemplate(noisy, templ, m);
        CHECK(r.location == Point(150, 90));
    }
    // spot-check the FFT cross term against a direct computation
    const ImageF cc = matchTemplate(noisy, templ, MatchMethod::CCorr);
    for (Point p : {Point(0, 0), Point(37, 101), Point(250, 160)}) {
        double direct = 0;
        for (int j = 0; j < templ.height(); ++j)
            for (int i = 0; i < templ.width(); ++i) direct += double(noisy(p.x + i, p.y + j)) * templ(i, j);
        CHECK_NEAR(cc(p.x, p.y), direct, direct * 1e-6);
    }
    CHECK_NEAR(matchTemplate(noisy, templ, MatchMethod::SqDiff)(150, 90), 0.0, 1e-3);  // FFT rounding (~1e-15 relative)
}

TEST(fft_phase_correlation) {
    const ImageF base = gaussianBlur<float>(randomImage<float>(128, 96, 1, 6), 1.5);
    // integer circular shift is recovered exactly
    ImageF shifted(128, 96, 1);
    for (int y = 0; y < 96; ++y)
        for (int x = 0; x < 128; ++x) shifted((x + 7) % 128, (y + 93) % 96) = base(x, y);
    const PhaseCorrelation pc = phaseCorrelate(base, shifted, false);
    CHECK_NEAR(pc.shift.x, 7, 0.05);
    CHECK_NEAR(pc.shift.y, -3, 0.05);
    CHECK(pc.response > 0.9);
    // sub-pixel translation of a smooth image
    const ImageF sub = warpAffine(base, AffineMatrix{1, 0, 4.4, 0, 1, -2.6}, Size(), Interpolation::Bicubic, BorderMode::Reflect);
    const PhaseCorrelation ps = phaseCorrelate(base, sub);
    CHECK_NEAR(ps.shift.x, 4.4, 0.3);
    CHECK_NEAR(ps.shift.y, -2.6, 0.3);
}

TEST(features_hough_lines_p) {
    ImageU8 e(120, 90, 1, 0);
    drawLine(e, PointF(10, 20), PointF(100, 20), Color(255));  // horizontal, 91 px
    drawLine(e, PointF(30, 30), PointF(30, 50), Color(255));   // vertical with a 5 px gap
    drawLine(e, PointF(30, 56), PointF(30, 80), Color(255));
    drawLine(e, PointF(50, 80), PointF(110, 40), Color(255));  // diagonal
    e(70, 70) = 255;                                          // isolated noise
    auto close = [](Point a, Point b) { return std::abs(a.x - b.x) <= 2 && std::abs(a.y - b.y) <= 2; };
    auto has = [&](const std::vector<LineSegment>& ls, Point a, Point b) {
        for (const auto& l : ls)
            if ((close(l.p0, a) && close(l.p1, b)) || (close(l.p0, b) && close(l.p1, a))) return true;
        return false;
    };
    const auto bridged = houghLinesP(e, 1, detail::kPi / 180, 15, 15, 8);
    CHECK(has(bridged, {10, 20}, {100, 20}));
    CHECK(has(bridged, {30, 30}, {30, 80}));  // gap of 5 bridged
    CHECK(has(bridged, {50, 80}, {110, 40}));
    CHECK_EQ(bridged.size(), 3u);
    // with a 1-pixel gap tolerance the 5-pixel gap must not be bridged
    const auto strict = houghLinesP(e, 1, detail::kPi / 180, 15, 15, 1);
    CHECK(has(strict, {10, 20}, {100, 20}));
    for (const auto& l : strict) CHECK(!(std::min(l.p0.y, l.p1.y) <= 50 && std::max(l.p0.y, l.p1.y) >= 56 && l.p0.x == 30));
    CHECK(houghLinesP(e, 1, detail::kPi / 180, 15, 200, 3).empty());   // nothing that long
    CHECK(houghLinesP(e, 1, detail::kPi / 180, 15, 15, 8) .size() == bridged.size());  // reproducible
}
