#include <ipf/draw.hpp>
#include <ipf/filter.hpp>
#include <ipf/metrics.hpp>

#include "test_framework.hpp"
#include "test_util.hpp"

using namespace ipf;
using namespace ipftest;

TEST(draw_lines_and_shapes) {
    ImageU8 img(20, 10, 1, 0);
    drawLine(img, PointF(2, 3), PointF(12, 3), Color(255));
    CHECK_EQ(countNonZero(img), 11u);
    img.fill(0);
    drawLine(img, PointF(0, 0), PointF(9, 9), Color(255));
    CHECK_EQ(countNonZero(img), 10u);
    CHECK_EQ(img(5, 5), 255);
    img.fill(0);
    drawLine(img, PointF(-50, -50), PointF(100, 100), Color(7));  // clipped, no crash
    CHECK_EQ(img(9, 9), 7);
    img.fill(0);
    drawLine(img, PointF(2, 5), PointF(17, 5), Color(255), 3);
    CHECK_EQ(img(10, 4), 255);
    CHECK_EQ(img(10, 6), 255);
    CHECK_EQ(img(10, 7), 0);
    ImageU8 aa(20, 10, 1, 0);
    drawLine(aa, PointF(2, 2), PointF(17, 7), Color(255), 1, true);
    bool partial = false;
    for (std::uint8_t v : aa) partial |= v > 0 && v < 255;
    CHECK(partial);

    ImageU8 r(12, 12, 1, 0);
    drawRectangle(r, Rect(2, 2, 8, 6), Color(255));
    CHECK_EQ(countNonZero(r), 24u);
    drawRectangle(r, Rect(2, 2, 8, 6), Color(255), -1);
    CHECK_EQ(countNonZero(r), 48u);
    drawRectangle(r, Rect(-5, -5, 30, 30), Color(9), 2);  // clipped outline
    CHECK_EQ(r(0, 0), 0);

    ImageU8 c(41, 41, 1, 0);
    drawCircle(c, PointF(20, 20), 10, Color(255), -1);
    CHECK(std::fabs(static_cast<double>(countNonZero(c)) - detail::kPi * 100) < 15);
    c.fill(0);
    drawCircle(c, PointF(20, 20), 10, Color(255));
    CHECK_EQ(c(30, 20), 255);
    CHECK_EQ(c(20, 10), 255);
    CHECK_EQ(c(20, 20), 0);
    c.fill(0);
    drawCircle(c, PointF(20, 20), 10, Color(255), 4);
    CHECK_EQ(c(20, 20), 0);
    CHECK_EQ(c(31, 20), 255);
    c.fill(0);
    drawEllipse(c, PointF(20, 20), PointF(15, 5), 0, Color(255), -1);
    CHECK_EQ(c(34, 20), 255);
    CHECK_EQ(c(20, 27), 0);
    c.fill(0);
    drawEllipse(c, PointF(20, 20), PointF(15, 5), 90, Color(255), -1);
    CHECK_EQ(c(20, 34), 255);

    ImageU8 tri(30, 30, 1, 0);
    fillPolygon(tri, std::vector<Point>{{0, 0}, {20, 0}, {0, 20}}, Color(255));
    const double area = static_cast<double>(countNonZero(tri));
    CHECK(area > 200 && area < 250);  // 210 + boundary pixels
    ImageU8 m(15, 15, 3, 0);
    drawMarker(m, PointF(7, 7), Color(0, 255, 0), MarkerType::Cross, 9);
    CHECK_EQ(m(7, 3, 1), 255);
    CHECK_EQ(m(3, 3, 1), 0);
}

TEST(draw_colors_and_text) {
    ImageU8 rgb(4, 4, 3, 0), gray(4, 4, 1, 0), rgba(4, 4, 4, 0);
    drawPixel(rgb, Point(1, 1), Color(100));
    CHECK_EQ(rgb(1, 1, 2), 100);
    drawPixel(gray, Point(1, 1), Color(255, 0, 0));
    CHECK_EQ(gray(1, 1), 76);
    drawPixel(rgba, Point(0, 0), Color(1, 2, 3));
    CHECK_EQ(rgba(0, 0, 3), 255);
    drawPixel(rgba, Point(0, 1), Color(1, 2, 3, 40));
    CHECK_EQ(rgba(0, 1, 3), 40);
    ImageF f(4, 4, 1, 0.f);
    drawPixel(f, Point(2, 2), Color(0.5));
    CHECK_NEAR(f(2, 2), 0.5, 0);

    const Size s = getTextSize("Hi!", 2);
    CHECK_EQ(s.width, (3 * 6 - 1) * 2);
    CHECK_EQ(s.height, 8 * 2);
    CHECK_EQ(getTextSize("ab\ncd").height, 18);
    ImageU8 t(60, 20, 1, 0);
    drawText(t, "IPF", Point(1, 1), Color(255));
    CHECK(countNonZero(t) > 30u);
    CHECK_EQ(t(3, 1), 255);  // top bar of 'I'
    drawText(t, "\x01\xff", Point(0, 0), Color(255));  // non-printable -> '?', no crash
}

TEST(metrics_basic) {
    const ImageU8 a = smoothImage(40, 30, 3);
    CHECK(std::isinf(psnr(a, a)));
    CHECK_NEAR(ssim(a, a), 1.0, 1e-9);
    ImageU8 b = a;
    for (auto& v : b) v = static_cast<std::uint8_t>(v > 250 ? v : v + 5);
    CHECK(mse(a, b) > 20 && mse(a, b) <= 25);
    CHECK(mae(a, b) <= 5);
    CHECK(psnr(a, b) > 34 && psnr(a, b) < 36);
    const ImageU8 noisy = add(a, randomImage<std::uint8_t>(40, 30, 3, 2));
    const double s1 = ssim(a, b), s2 = ssim(a, noisy);
    CHECK(s1 > 0.95);
    CHECK(s2 < s1);
    ImageF map;
    ssim(a, noisy, &map);
    CHECK(map.sameShape(a));
    ImageU8 two(8, 8, 1, 0);
    for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 8; ++y) two(x, y) = 255;
    CHECK_NEAR(entropy(two), 1.0, 1e-12);
    CHECK_NEAR(entropy(ImageU8(5, 5, 1, 3)), 0.0, 1e-12);
    const ImageU8 sharp = randomImage<std::uint8_t>(32, 32, 1, 5);
    CHECK(sharpness(gaussianBlur(sharp, 2.0)) < sharpness(sharp) / 10);
}
