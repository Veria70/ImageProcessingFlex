#include <ipf/color.hpp>

#include "test_framework.hpp"
#include "test_util.hpp"

using namespace ipf;
using namespace ipftest;

namespace {
ImageU8 pixel(int r, int g, int b) {
    ImageU8 p(1, 1, 3);
    p.fill({static_cast<std::uint8_t>(r), static_cast<std::uint8_t>(g), static_cast<std::uint8_t>(b)});
    return p;
}
}  // namespace

TEST(color_gray) {
    CHECK_EQ(rgbToGray(pixel(255, 0, 0))(0, 0), 76);
    CHECK_EQ(rgbToGray(pixel(0, 255, 0))(0, 0), 150);
    CHECK_EQ(rgbToGray(pixel(0, 0, 255))(0, 0), 29);
    CHECK_EQ(rgbToGray(pixel(10, 20, 30), GrayMethod::Average)(0, 0), 20);
    CHECK_EQ(rgbToGray(pixel(10, 20, 30), GrayMethod::Max)(0, 0), 30);
    CHECK_EQ(rgbToGray(pixel(10, 20, 30), GrayMethod::Lightness)(0, 0), 20);
    CHECK_EQ(rgbToGray(pixel(0, 255, 0), GrayMethod::Luma709)(0, 0), 182);
    const ImageU8 g = toGray(smoothImage(5, 5, 3));
    CHECK_EQ(g.channels(), 1);
    CHECK_EQ(grayToRgb(g)(3, 3, 2), g(3, 3));
    CHECK_THROWS(rgbToGray(ImageU8(2, 2, 1)));
    const ImageU8 s = swapRB(pixel(1, 2, 3));
    CHECK_EQ(s(0, 0, 0), 3);
    CHECK_EQ(s(0, 0, 2), 1);
}

TEST(color_hsv_hsl) {
    const ImageF h = rgbToHsv(pixel(255, 0, 0));
    CHECK_NEAR(h(0, 0, 0), 0, 1e-5);
    CHECK_NEAR(h(0, 0, 1), 1, 1e-6);
    CHECK_NEAR(h(0, 0, 2), 1, 1e-6);
    CHECK_NEAR(rgbToHsv(pixel(0, 255, 0))(0, 0, 0), 120, 1e-4);
    CHECK_NEAR(rgbToHsv(pixel(0, 0, 255))(0, 0, 0), 240, 1e-4);
    CHECK_NEAR(rgbToHsv(pixel(255, 0, 128))(0, 0, 0), 330 - 0.12, 0.2);
    const ImageF l = rgbToHsl(pixel(255, 128, 0));
    CHECK_NEAR(l(0, 0, 2), (255 + 0) / 2.0 / 255.0, 1e-6);
    CHECK_NEAR(l(0, 0, 1), 1.0, 1e-6);
    // round trips over the whole RGB cube (subsampled)
    ImageU8 cube(32, 32 * 32, 3);
    for (int r = 0; r < 32; ++r)
        for (int g = 0; g < 32; ++g)
            for (int b = 0; b < 32; ++b) {
                cube(b, r * 32 + g, 0) = static_cast<std::uint8_t>(r * 8 + 7 * (b & 1));
                cube(b, r * 32 + g, 1) = static_cast<std::uint8_t>(g * 8 + 3);
                cube(b, r * 32 + g, 2) = static_cast<std::uint8_t>(b * 8);
            }
    CHECK(hsvToRgb(rgbToHsv(cube)) == cube);
    CHECK(hslToRgb(rgbToHsl(cube)) == cube);
    CHECK(maxAbsDiff(yCbCrToRgb(rgbToYCbCr(cube)), cube) <= 1.0);
    const ImageF cf = toFloat(cube);
    CHECK(maxAbsDiff(hsvToRgb<float>(rgbToHsv(cf)), cf) < 1e-5);
    CHECK(maxAbsDiff(labToRgb(rgbToLab(cube)), cube) <= 1.0);
}

TEST(color_lab_reference_values) {
    // Reference values from Bruce Lindbloom's calculator (sRGB, D65).
    struct Ref {
        int r, g, b;
        double L, A, B;
    } refs[] = {{255, 255, 255, 100.0, 0.0, 0.0},
                {255, 0, 0, 53.2408, 80.0925, 67.2032},
                {0, 255, 0, 87.7347, -86.1827, 83.1793},
                {0, 0, 255, 32.2970, 79.1875, -107.8602},
                {128, 128, 128, 53.5850, 0.0, 0.0}};
    for (const Ref& r : refs) {
        const ImageF lab = rgbToLab(pixel(r.r, r.g, r.b));
        CHECK_NEAR(lab(0, 0, 0), r.L, 0.01);
        CHECK_NEAR(lab(0, 0, 1), r.A, 0.02);
        CHECK_NEAR(lab(0, 0, 2), r.B, 0.02);
    }
    const ImageF xyz = rgbToXyz(pixel(255, 255, 255));
    CHECK_NEAR(xyz(0, 0, 0), 0.95047, 1e-4);
    CHECK_NEAR(xyz(0, 0, 1), 1.0, 1e-4);
    CHECK_NEAR(xyz(0, 0, 2), 1.08883, 1e-3);
    CHECK_NEAR(deltaE76(rgbToLab(pixel(255, 0, 0)), rgbToLab(pixel(255, 0, 0)))(0, 0), 0, 1e-6);
}

TEST(color_ycbcr) {
    const ImageU8 y = rgbToYCbCr(pixel(255, 255, 255));
    CHECK_EQ(y(0, 0, 0), 255);
    CHECK_EQ(y(0, 0, 1), 128);
    CHECK_EQ(y(0, 0, 2), 128);
    const ImageU8 r = rgbToYCbCr(pixel(255, 0, 0));
    CHECK_EQ(r(0, 0, 0), 76);
    CHECK_EQ(r(0, 0, 1), 85);
    CHECK_EQ(r(0, 0, 2), 255);
}

TEST(color_adjust) {
    const ImageU8 img = smoothImage(16, 16, 3);
    CHECK(adjustSaturation(img, 1.0) == img);
    const ImageU8 gray = adjustSaturation(img, 0.0);
    for (int y = 0; y < 16; ++y) {
        CHECK_EQ(gray(3, y, 0), gray(3, y, 1));
        CHECK_EQ(gray(3, y, 1), gray(3, y, 2));
    }
    CHECK(maxAbsDiff(shiftHue(img, 360.0), img) <= 1.0);
    const ImageU8 red = shiftHue(pixel(255, 0, 0), 120);
    CHECK_EQ(red(0, 0, 1), 255);
    CHECK_EQ(red(0, 0, 0), 0);
    const ImageU8 sp = sepia(pixel(100, 100, 100));
    CHECK(sp(0, 0, 0) > sp(0, 0, 1) && sp(0, 0, 1) > sp(0, 0, 2));
    // alpha preserved
    ImageU8 rgba(2, 2, 4);
    rgba.fill({255, 0, 0, 77});
    CHECK_EQ(shiftHue(rgba, 90)(1, 1, 3), 77);
    CHECK_EQ(adjustSaturation(rgba, 0.5)(1, 1, 3), 77);
}

TEST(color_colormap) {
    ImageF ramp(256, 1, 1);
    for (int x = 0; x < 256; ++x) ramp(x, 0) = static_cast<float>(x);
    for (Colormap m : {Colormap::Gray, Colormap::Jet, Colormap::Hot, Colormap::Viridis, Colormap::Turbo, Colormap::Inferno}) {
        const ImageU8 c = applyColormap(ramp, m);
        CHECK_EQ(c.channels(), 3);
        CHECK_EQ(c.width(), 256);
    }
    const ImageU8 g = applyColormap(ramp, Colormap::Gray);
    CHECK_EQ(g(0, 0, 0), 0);
    CHECK_EQ(g(255, 0, 0), 255);
    CHECK_EQ(g(128, 0, 1), 128);
    const ImageU8 jet = applyColormap(ramp, Colormap::Jet);
    CHECK(jet(0, 0, 2) > 100 && jet(0, 0, 0) == 0);    // starts blue
    CHECK(jet(255, 0, 0) > 100 && jet(255, 0, 2) == 0);  // ends red
    const ImageU8 v = applyColormap(ramp, Colormap::Viridis);
    CHECK(std::abs(v(0, 0, 0) - 68) <= 4 && std::abs(v(0, 0, 2) - 84) <= 4);      // viridis(0) = #440154
    CHECK(std::abs(v(255, 0, 0) - 253) <= 4 && std::abs(v(255, 0, 1) - 231) <= 4);  // viridis(1) = #FDE725
}
