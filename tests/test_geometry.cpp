#include <ipf/geometry.hpp>

#include "test_framework.hpp"
#include "test_util.hpp"

using namespace ipf;
using namespace ipftest;

namespace {
ImageI32 indexImage(int w, int h) {
    ImageI32 img(w, h, 1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) img(x, y) = y * 100 + x;
    return img;
}
}  // namespace

TEST(geometry_exact_transforms) {
    const ImageI32 a = indexImage(4, 3);
    CHECK_EQ(flipHorizontal(a)(0, 1), 103);
    CHECK_EQ(flipVertical(a)(1, 0), 201);
    const ImageI32 t = transpose(a);
    CHECK_EQ(t.width(), 3);
    CHECK_EQ(t(2, 3), 203);
    const ImageI32 ccw = rotate90(a, 1);  // top-right corner moves to top-left
    CHECK_EQ(ccw.width(), 3);
    CHECK_EQ(ccw.height(), 4);
    CHECK_EQ(ccw(0, 0), 3);
    CHECK_EQ(ccw(2, 3), 200);
    const ImageI32 cw = rotate90(a, -1);  // bottom-left corner moves to top-left
    CHECK_EQ(cw(0, 0), 200);
    CHECK(rotate90(a, 4) == a);
    CHECK(rotate90(rotate90(a, 1), 3) == a);
    CHECK(rotate90(a, 2) == flipHorizontal(flipVertical(a)));
    CHECK_EQ(pad(a, 2, BorderMode::Constant, -1)(0, 0), -1);
    CHECK_EQ(pad(a, 1, BorderMode::Replicate)(0, 0), 0);
}

TEST(geometry_resize_basics) {
    const ImageU8 img = smoothImage(30, 20, 3);
    for (Interpolation m : {Interpolation::Nearest, Interpolation::Bilinear, Interpolation::Bicubic,
                            Interpolation::Lanczos, Interpolation::Area, Interpolation::Hamming}) {
        CHECK(resize(img, 30, 20, m) == img);
        const ImageU8 c(17, 9, 2, 123);
        CHECK(resize(c, 40, 5, m) == ImageU8(40, 5, 2, 123));
        CHECK(resize(c, 3, 30, m) == ImageU8(3, 30, 2, 123));
    }
    // 2x Area downscale is the exact mean of each 2x2 block
    const ImageF f = randomImage<float>(8, 6, 1, 3);
    const ImageF half = resize(f, 4, 3, Interpolation::Area);
    CHECK_NEAR(half(1, 2), (f(2, 4) + f(3, 4) + f(2, 5) + f(3, 5)) / 4, 1e-6);
    // nearest 2x upscale duplicates pixels
    const ImageI32 idx = indexImage(3, 2);
    const ImageI32 up = resize(idx, 6, 4, Interpolation::Nearest);
    CHECK_EQ(up(5, 3), 102);
    CHECK_EQ(up(4, 2), 102);
    CHECK_EQ(up(1, 1), 0);
    CHECK_EQ(rescale(img, 0.5).width(), 15);
    const ImageU8 fit = resizeToFit(img, 10, 10);
    CHECK_EQ(fit.width(), 10);
    CHECK_EQ(fit.height(), 7);
    CHECK_THROWS(resize(img, 0, 5));
}

TEST(geometry_sampling_and_warps) {
    const ImageF f = randomImage<float>(9, 7, 2, 5);
    CHECK_NEAR(sample(f, 3, 4, 1), f(3, 4, 1), 1e-7);
    CHECK_NEAR(sample(f, 3.5, 4, 0), (f(3, 4) + f(4, 4)) / 2, 1e-6);
    CHECK_NEAR(sample(f, 3, 4, 1, Interpolation::Bicubic), f(3, 4, 1), 1e-6);
    CHECK_NEAR(sample(f, -5, 4, 0, Interpolation::Bilinear, BorderMode::Constant, 9), 9, 0);
    // identity and integer translation
    const AffineMatrix id = {1, 0, 0, 0, 1, 0};
    CHECK(warpAffine(f, id) == f);
    const ImageF moved = warpAffine(f, AffineMatrix{1, 0, 2, 0, 1, 1}, Size(), Interpolation::Bilinear);
    CHECK_NEAR(moved(5, 4, 1), f(3, 3, 1), 1e-6);
    CHECK_NEAR(moved(0, 0, 0), 0, 0);  // uncovered area takes the border value
    // 90 degree rotation of a square image equals rotate90
    const ImageU8 sq = smoothImage(11, 11, 1);
    CHECK(rotate(sq, 90.0, Interpolation::Bilinear) == rotate90(sq, 1));
    CHECK(rotate(sq, -90.0, Interpolation::Nearest) == rotate90(sq, -1));
    const ImageU8 big = rotate(smoothImage(20, 10, 1), 90.0, Interpolation::Nearest, true);
    CHECK_EQ(big.width(), 10);
    CHECK_EQ(big.height(), 20);
    const Homography hid = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    CHECK(warpPerspective(f, hid) == f);
}

TEST(geometry_transform_estimation) {
    const AffineMatrix m = {1.2, -0.3, 5, 0.4, 0.9, -2};
    const std::array<PointF, 3> src = {PointF(0, 0), PointF(10, 0), PointF(3, 7)};
    std::array<PointF, 3> dst;
    for (int i = 0; i < 3; ++i) dst[i] = applyAffine(m, src[i]);
    const AffineMatrix est = getAffineTransform(src, dst);
    for (int i = 0; i < 6; ++i) CHECK_NEAR(est[i], m[i], 1e-9);
    const AffineMatrix back = composeAffine(invertAffine(m), m);
    CHECK_NEAR(back[0], 1, 1e-12);
    CHECK_NEAR(back[2], 0, 1e-12);
    const std::array<PointF, 4> s4 = {PointF(0, 0), PointF(100, 0), PointF(100, 50), PointF(0, 50)};
    const std::array<PointF, 4> d4 = {PointF(10, 5), PointF(90, 0), PointF(110, 60), PointF(-5, 45)};
    const Homography h = getPerspectiveTransform(s4, d4);
    for (int i = 0; i < 4; ++i) {
        const PointF p = applyHomography(h, s4[i]);
        CHECK_NEAR(p.x, d4[i].x, 1e-8);
        CHECK_NEAR(p.y, d4[i].y, 1e-8);
    }
    const PointF q = applyHomography(invertHomography(h), applyHomography(h, PointF(37, 21)));
    CHECK_NEAR(q.x, 37, 1e-8);
    CHECK_NEAR(q.y, 21, 1e-8);
    const AffineMatrix r = getRotationMatrix2D(PointF(5, 5), 90);
    const PointF pr = applyAffine(r, PointF(10, 5));  // right of center -> above center (CCW on screen)
    CHECK_NEAR(pr.x, 5, 1e-12);
    CHECK_NEAR(pr.y, 0, 1e-12);
    // collinear source points cannot define an affine map
    CHECK_THROWS(getAffineTransform({PointF(0, 0), PointF(1, 1), PointF(2, 2)}, dst));
}

TEST(geometry_pyramids) {
    const ImageU8 c(13, 8, 3, 90);
    const ImageU8 d = pyrDown(c);
    CHECK_EQ(d.width(), 7);
    CHECK_EQ(d.height(), 4);
    CHECK(d == ImageU8(7, 4, 3, 90));
    const ImageU8 u = pyrUp(c);
    CHECK_EQ(u.width(), 26);
    CHECK(u == ImageU8(26, 16, 3, 90));
    const auto pyr = buildPyramid(smoothImage(64, 48, 1), 4);
    CHECK_EQ(pyr.size(), 4u);
    CHECK_EQ(pyr[3].width(), 8);
}
