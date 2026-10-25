#include <ipf/pixel_ops.hpp>

#include "test_framework.hpp"
#include "test_util.hpp"

using namespace ipf;
using namespace ipftest;

TEST(pixel_arithmetic) {
    ImageU8 a(2, 1, 1), b(2, 1, 1);
    a(0, 0) = 200; a(1, 0) = 10;
    b(0, 0) = 100; b(1, 0) = 20;
    CHECK_EQ(add(a, b)(0, 0), 255);
    CHECK_EQ(subtract(a, b)(1, 0), 0);
    CHECK_EQ(absDiff(a, b)(1, 0), 10);
    CHECK_EQ(multiply(a, b, 1.0 / 255)(0, 0), 78);
    CHECK_EQ(divide(a, b)(0, 0), 2);
    CHECK_EQ(addWeighted(a, 0.5, b, 0.5, 1)(0, 0), 151);
    CHECK_EQ(blend(a, b, 0.25)(0, 0), 175);
    CHECK_EQ(minImage(a, b)(0, 0), 100);
    CHECK_EQ(maxImage(a, b)(1, 0), 20);
    CHECK_EQ(addScalar(a, -15)(1, 0), 0);
    CHECK_EQ(multiplyScalar(a, 1.5)(1, 0), 15);
    CHECK_EQ(invert(a)(0, 0), 55);
    CHECK_EQ(brightnessContrast(a, 2.0, -100)(0, 0), 255);
    CHECK_EQ(clampValues(a, 20, 150)(0, 0), 150);
    CHECK_THROWS(add(a, ImageU8(3, 1, 1)));
    ImageF f(1, 1, 1, 0.25f);
    CHECK_NEAR(invert(f)(0, 0), 0.75, 1e-7);
    ImageU8 ga(1, 1, 2);
    ga(0, 0, 0) = 10;
    ga(0, 0, 1) = 99;
    CHECK_EQ(invert(ga)(0, 0, 1), 99);  // alpha kept
    CHECK_EQ(invert(ga, false)(0, 0, 1), 156);
}

TEST(pixel_alpha_composite) {
    ImageU8 fg(1, 1, 4), bg(1, 1, 3);
    fg.fill({255, 0, 0, 128});
    bg.fill({0, 0, 255});
    const ImageU8 o = alphaComposite(fg, bg);
    CHECK_EQ(o.channels(), 3);
    CHECK_EQ(o(0, 0, 0), 128);
    CHECK_EQ(o(0, 0, 2), 127);
    ImageU8 bg4(1, 1, 4);
    bg4.fill({0, 0, 255, 0});
    const ImageU8 o4 = alphaComposite(fg, bg4);  // over transparent: fg color, fg alpha
    CHECK_EQ(o4(0, 0, 0), 255);
    CHECK_EQ(o4(0, 0, 3), 128);
}

TEST(pixel_masks) {
    ImageU8 a(3, 1, 1);
    a(0, 0) = 1; a(1, 0) = 5; a(2, 0) = 9;
    const ImageU8 m = compare(a, 5.0, CmpOp::GreaterEqual);
    CHECK_EQ(m(0, 0), 0);
    CHECK_EQ(m(1, 0), 255);
    CHECK_EQ(countNonZero(m), 2u);
    ImageU8 rgb(2, 1, 3);
    rgb(0, 0, 0) = 200; rgb(0, 0, 1) = 50; rgb(0, 0, 2) = 50;
    const ImageU8 r = inRange(rgb, {150, 0, 0}, {255, 100, 100});
    CHECK_EQ(r(0, 0), 255);
    CHECK_EQ(r(1, 0), 0);
    CHECK_EQ(bitwiseAnd(m, compare(a, 5.0, CmpOp::LessEqual))(1, 0), 255);
    CHECK_EQ(bitwiseOr(m, compare(a, 5.0, CmpOp::Less))(0, 0), 255);
    CHECK_EQ(bitwiseXor(m, m)(1, 0), 0);
    CHECK_EQ(bitwiseNot(m)(0, 0), 255);
    setTo(rgb, {1, 2, 3}, r);
    CHECK_EQ(rgb(0, 0, 2), 3);
    ImageU8 dst(2, 1, 3, 0);
    copyMasked(rgb, dst, r);
    CHECK_EQ(dst(0, 0, 1), 2);
    CHECK_EQ(dst(1, 0, 1), 0);
}

TEST(pixel_threshold) {
    ImageU8 a(5, 1, 1);
    for (int i = 0; i < 5; ++i) a(i, 0) = static_cast<std::uint8_t>(i * 50);  // 0 50 100 150 200
    const ImageU8 bin = threshold(a, 100, 255);
    CHECK_EQ(bin(2, 0), 0);
    CHECK_EQ(bin(3, 0), 255);
    CHECK_EQ(threshold(a, 100, 255, ThresholdType::BinaryInv)(2, 0), 255);
    CHECK_EQ(threshold(a, 100, 255, ThresholdType::Trunc)(4, 0), 100);
    CHECK_EQ(threshold(a, 100, 255, ThresholdType::ToZero)(4, 0), 200);
    CHECK_EQ(threshold(a, 100, 255, ThresholdType::ToZeroInv)(1, 0), 50);
    // bimodal image: Otsu threshold must separate the two modes
    ImageU8 bi(100, 100, 1);
    std::mt19937 rng(3);
    for (int y = 0; y < 100; ++y)
        for (int x = 0; x < 100; ++x)
            bi(x, y) = static_cast<std::uint8_t>((x < 40 ? 60 : 180) + static_cast<int>(rng() % 21) - 10);
    const int t = otsuThreshold(bi);
    CHECK(t >= 70 && t < 170);
    const ImageU8 seg = thresholdOtsu(bi);
    CHECK_EQ(countNonZero(seg), 6000u);
    ImageF bf = toFloat(bi);
    CHECK_EQ(countNonZero(thresholdOtsu(bf)), 6000u);
}

TEST(pixel_histogram_equalize) {
    ImageU8 a(4, 1, 1);
    a(0, 0) = 0; a(1, 0) = 0; a(2, 0) = 255; a(3, 0) = 7;
    const auto h = histogram(a);
    CHECK_EQ(h[0], 2u);
    CHECK_EQ(h[255], 1u);
    const auto h2 = histogram(toFloat(a), 4, 0.0, 1.0);
    CHECK_EQ(h2[0], 3u);
    CHECK_EQ(h2[3], 1u);
    // low-contrast ramp gets stretched to the full range
    ImageU8 ramp(64, 1, 1);
    for (int x = 0; x < 64; ++x) ramp(x, 0) = static_cast<std::uint8_t>(100 + x / 2);
    const ImageU8 eq = equalizeHist(ramp);
    CHECK_EQ(eq(0, 0), 0);
    CHECK_EQ(eq(63, 0), 255);
    const ImageU8 constant(8, 8, 1, 42);
    CHECK(equalizeHist(constant) == constant);
    const ImageU8 ac = autoContrast(ramp);
    CHECK_EQ(ac(0, 0), 0);
    CHECK_EQ(ac(63, 0), 255);
    const ImageU8 n = normalize(ramp);
    CHECK_EQ(n(63, 0), 255);
    CHECK_EQ(n(0, 0), 0);
    const ImageU8 g = gammaCorrect(ramp, 2.0);
    CHECK(g(10, 0) > ramp(10, 0));
    // CLAHE: increases contrast, keeps size, constant stays constant-ish
    const ImageU8 sm = smoothImage(80, 60, 1);
    const ImageU8 cl = clahe(sm, 2.0, 4, 4);
    CHECK(cl.sameShape(sm));
    CHECK(stddev(cl)[0] > stddev(sm)[0] * 0.9);
    // histogram matching to itself is the identity
    CHECK(matchHistogram(sm, sm) == sm);
    const ImageU8 matched = matchHistogram(sm, eq);
    CHECK(matched.sameShape(sm));
}
