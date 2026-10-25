#include <ipf/filter.hpp>

#include "test_framework.hpp"
#include "test_util.hpp"

using namespace ipf;
using namespace ipftest;

TEST(filter_kernels) {
    const auto g = gaussianKernel1D(1.0);
    CHECK_EQ(g.size(), 7u);
    double s = 0;
    for (double v : g) s += v;
    CHECK_NEAR(s, 1.0, 1e-12);
    CHECK_NEAR(g[3] / g[4], std::exp(0.5), 1e-12);
    CHECK_EQ(gaussianKernel1D(0, 5).size(), 5u);
    const auto d = detail::derivKernel(1, 5);
    const double expect[] = {-1, -2, 0, 2, 1};
    for (int i = 0; i < 5; ++i) CHECK_NEAR(d[i], expect[i], 0);
    const auto d2 = detail::derivKernel(2, 3);
    CHECK_NEAR(d2[0], 1, 0);
    CHECK_NEAR(d2[1], -2, 0);
    const Kernel k2 = gaussianKernel2D(1.5);
    double s2 = 0;
    for (float v : k2) s2 += v;
    CHECK_NEAR(s2, 1.0, 1e-5);
    const Kernel mb = motionBlurKernel(9, 0);
    double s3 = 0;
    for (float v : mb) s3 += v;
    CHECK_NEAR(s3, 1.0, 1e-5);
    CHECK_NEAR(mb(0, 4), 1.0 / 9, 1e-6);
}

TEST(filter_impulse_response) {
    // Correlating an impulse with a kernel reproduces the flipped kernel.
    ImageF impulse(7, 7, 1, 0.f);
    impulse(3, 3) = 1.f;
    const Kernel k = makeKernel(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    const ImageF r = filter2D(impulse, k, BorderMode::Constant);
    CHECK_NEAR(r(2, 2), 9, 0);
    CHECK_NEAR(r(4, 4), 1, 0);
    CHECK_NEAR(r(3, 2), 8, 0);
    const ImageF sep = sepFilter2D(impulse, {1, 2, 3}, {4, 5, 6}, BorderMode::Constant);
    CHECK_NEAR(sep(2, 2), 3 * 6, 0);
    CHECK_NEAR(sep(4, 2), 1 * 6, 0);
    CHECK_NEAR(sep(2, 4), 3 * 4, 0);
}

TEST(filter_constant_invariance) {
    const ImageU8 c(20, 15, 3, 77);
    CHECK(gaussianBlur(c, 2.0) == c);
    CHECK(blur(c, 5) == c);
    CHECK(medianBlur(c, 5) == c);
    CHECK(medianBlur(c, 9) == c);
    CHECK(bilateralFilter(c, 7, 30, 3) == c);
    CHECK(unsharpMask(c, 2.0, 1.5) == c);
    CHECK(sharpen(c) == c);
    for (BorderMode m : {BorderMode::Replicate, BorderMode::Reflect, BorderMode::Reflect101, BorderMode::Wrap})
        CHECK(boxFilter(c, 4, 6, true, m) == c);
    const ImageF gx = sobel(c, 1, 0);
    CHECK_NEAR(minMaxLoc(gx).maxVal, 0, 0);
    CHECK_NEAR(minMaxLoc(laplacian(c, 3)).maxVal, 0, 0);
}

TEST(filter_box_matches_generic) {
    const ImageU8 img = randomImage<std::uint8_t>(31, 17, 2, 9);
    for (BorderMode m : {BorderMode::Constant, BorderMode::Replicate, BorderMode::Reflect, BorderMode::Reflect101, BorderMode::Wrap}) {
        const ImageF fast = boxFilter<float>(img, 5, 3, true, m, 3.0);
        const ImageF ref = filter2D<float>(img, Kernel(5, 3, 1, 1.f / 15), m, 0.0, Point(-1, -1), 3.0);
        CHECK(maxAbsDiff(fast, ref) < 1e-3);
        const ImageF unnorm = boxFilter<float>(img, 4, 4, false, m);  // even size, anchor (2, 2)
        const ImageF ref2 = filter2D<float>(img, Kernel(4, 4, 1, 1.f), m);
        CHECK(maxAbsDiff(unnorm, ref2) < 1e-2);
    }
}

TEST(filter_gradients) {
    // Linear ramp f = 3x + 2y: Sobel 3x3 = 8 * slope, central difference (ksize 1) = 2 * slope.
    ImageF ramp(16, 16, 1);
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) ramp(x, y) = static_cast<float>(3 * x + 2 * y);
    const ImageF gx = sobel(ramp, 1, 0), gy = sobel(ramp, 0, 1);
    CHECK_NEAR(gx(8, 8), 24, 1e-4);
    CHECK_NEAR(gy(8, 8), 16, 1e-4);
    CHECK_NEAR(sobel(ramp, 1, 0, 1)(8, 8), 6, 1e-4);
    CHECK_NEAR(sobel(ramp, 1, 0, 5)(8, 8), 3 * 128, 1e-3);
    CHECK_NEAR(scharr(ramp, 1, 0)(8, 8), 3 * 32, 1e-4);
    CHECK_NEAR(prewitt(ramp, 0, 1)(8, 8), 2 * 6, 1e-4);
    CHECK_NEAR(magnitude(gx, gy)(8, 8), std::hypot(24.0, 16.0), 1e-4);
    CHECK_NEAR(phase(gx, gy)(8, 8), std::atan2(16.0, 24.0) * 180 / detail::kPi, 1e-3);
    CHECK_NEAR(gradientMagnitude(ramp)(5, 5), std::hypot(24.0, 16.0), 1e-4);
    // Laplacian of a quadratic x^2 + y^2 is 4.
    ImageF q(16, 16, 1);
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) q(x, y) = static_cast<float>(x * x + y * y);
    CHECK_NEAR(laplacian(q)(8, 8), 4, 1e-3);
    CHECK_NEAR(laplacian(q, 3)(8, 8), 4 * 4, 1e-3);  // Sobel second derivative scale 4 at ksize 3
}

TEST(filter_median_and_bilateral) {
    ImageU8 salt(9, 9, 1, 100);
    salt(4, 4) = 255;
    salt(2, 6) = 0;
    const ImageU8 m3 = medianBlur(salt, 3);
    CHECK_EQ(m3(4, 4), 100);
    CHECK_EQ(m3(2, 6), 100);
    // u8 histogram path (ksize >= 7) agrees with the generic sort path (float)
    const ImageU8 r = randomImage<std::uint8_t>(23, 19, 2, 4);
    const ImageU8 fastMed = medianBlur(r, 7, BorderMode::Reflect);
    const ImageF slowMed = medianBlur(r.convertTo<float>(), 7, BorderMode::Reflect);
    CHECK(maxAbsDiff(fastMed.convertTo<float>(), slowMed) == 0.0);
    // bilateral keeps a strong step edge
    ImageU8 step(20, 10, 1);
    for (int y = 0; y < 10; ++y)
        for (int x = 0; x < 20; ++x) step(x, y) = x < 10 ? 20 : 220;
    const ImageU8 bf = bilateralFilter(step, 9, 20, 5);
    CHECK_EQ(bf(9, 5), 20);
    CHECK_EQ(bf(10, 5), 220);
    const ImageU8 gb = gaussianBlur(step, 2.0);
    CHECK(gb(9, 5) > 60);  // whereas Gaussian blur smears it
}

TEST(filter_integral_and_adaptive) {
    const ImageU8 img = randomImage<std::uint8_t>(13, 11, 1, 2);
    const ImageD I = integralImage(img);
    double direct = 0;
    for (int y = 2; y < 9; ++y)
        for (int x = 3; x < 10; ++x) direct += img(x, y);
    CHECK_NEAR(integralSum(I, Rect(3, 2, 7, 7)), direct, 1e-9);
    CHECK_NEAR(integralSum(I, img.rect()), sum(img)[0], 1e-6);
    // adaptive threshold separates text-like dark strokes under an illumination gradient
    ImageU8 page(60, 30, 1);
    for (int y = 0; y < 30; ++y)
        for (int x = 0; x < 60; ++x) page(x, y) = static_cast<std::uint8_t>(80 + 2 * x + ((x % 10 == 5) ? -50 : 0));
    const ImageU8 bin = adaptiveThreshold(page, 255, AdaptiveMethod::Mean, ThresholdType::BinaryInv, 7, 10);
    CHECK_EQ(bin(15, 10), 255);  // stroke
    CHECK_EQ(bin(18, 10), 0);    // background
    CHECK_EQ(bin(55, 10), 255);
    const ImageU8 bing = adaptiveThreshold(page, 255, AdaptiveMethod::Gaussian, ThresholdType::BinaryInv, 7, 10);
    CHECK_EQ(bing(45, 3), 255);
    CHECK_EQ(bing(47, 3), 0);
}
