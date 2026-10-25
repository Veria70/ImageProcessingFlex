#include <ipf/morphology.hpp>

#include "test_framework.hpp"
#include "test_util.hpp"

using namespace ipf;
using namespace ipftest;

TEST(morph_structuring_elements) {
    const auto r = getStructuringElement(MorphShape::Rect, 3, 5);
    CHECK_EQ(countNonZero(r), 15u);
    const auto c = getStructuringElement(MorphShape::Cross, 5);
    CHECK_EQ(countNonZero(c), 9u);
    CHECK_EQ(c(2, 0), 1);
    CHECK_EQ(c(0, 0), 0);
    // OpenCV's 5x5 ellipse:
    // 0 0 1 0 0 / 1 1 1 1 1 / 1 1 1 1 1 / 1 1 1 1 1 / 0 0 1 0 0
    const auto e = getStructuringElement(MorphShape::Ellipse, 5);
    CHECK_EQ(countNonZero(e), 17u);
    CHECK_EQ(e(0, 0), 0);
    CHECK_EQ(e(2, 0), 1);
    CHECK_EQ(e(0, 1), 1);
}

TEST(morph_erode_dilate) {
    ImageU8 img(9, 9, 1, 0);
    for (int y = 3; y < 6; ++y)
        for (int x = 3; x < 6; ++x) img(x, y) = 255;  // 3x3 square
    const ImageU8 er = erode(img);
    CHECK_EQ(countNonZero(er), 1u);
    CHECK_EQ(er(4, 4), 255);
    const ImageU8 di = dilate(img);
    CHECK_EQ(countNonZero(di), 25u);
    CHECK(erode(dilate(img)) == img);  // closing a convex square is a no-op
    const ImageU8 cross = dilate(er, getStructuringElement(MorphShape::Cross, 3));
    CHECK_EQ(countNonZero(cross), 5u);
    CHECK(dilate(er, getStructuringElement(MorphShape::Rect, 3), 2) == dilate(er, getStructuringElement(MorphShape::Rect, 5)));
    // outside pixels are ignored: a full white image stays white under erosion
    const ImageU8 white(6, 4, 2, 255);
    CHECK(erode(white, getStructuringElement(MorphShape::Ellipse, 5)) == white);
    const ImageF zeros(6, 4, 1, 0.f);
    CHECK(dilate(zeros, getStructuringElement(MorphShape::Cross, 7)) == zeros);
    // rect fast path agrees with the generic path (same mask built as non-rect + center)
    const ImageU8 rnd = randomImage<std::uint8_t>(15, 11, 2, 3);
    StructuringElement almostRect = getStructuringElement(MorphShape::Rect, 5, 3);
    const ImageU8 fast = erode(rnd, almostRect);
    StructuringElement generic(7, 3, 1, 0);  // 5x3 rect embedded in a wider element with the same anchor
    for (int y = 0; y < 3; ++y)
        for (int x = 1; x < 6; ++x) generic(x, y) = 1;
    CHECK(erode(rnd, generic) == fast);
}

TEST(morph_compound) {
    ImageU8 img(20, 20, 1, 0);
    for (int y = 5; y < 15; ++y)
        for (int x = 5; x < 15; ++x) img(x, y) = 200;
    img(1, 1) = 200;  // isolated speck
    img(10, 10) = 0;  // pin hole
    const auto se = getStructuringElement(MorphShape::Rect, 3);
    const ImageU8 opened = morphologyEx(img, MorphOp::Open, se);
    CHECK_EQ(opened(1, 1), 0);
    const ImageU8 closed = morphologyEx(img, MorphOp::Close, se);
    CHECK_EQ(closed(10, 10), 200);
    const ImageU8 grad = morphologyEx(img, MorphOp::Gradient, se);
    CHECK_EQ(grad(5, 10), 200);
    CHECK_EQ(grad(7, 7), 0);
    CHECK_EQ(morphologyEx(img, MorphOp::TopHat, se)(1, 1), 200);
    CHECK_EQ(morphologyEx(img, MorphOp::BlackHat, se)(10, 10), 200);
}

TEST(morph_binary_tools) {
    // ring with a hole
    ImageU8 ring(15, 15, 1, 0);
    for (int y = 2; y < 13; ++y)
        for (int x = 2; x < 13; ++x)
            if (x < 5 || x > 9 || y < 5 || y > 9) ring(x, y) = 255;
    const ImageU8 filled = fillHoles(ring);
    CHECK_EQ(filled(7, 7), 255);
    CHECK_EQ(filled(0, 0), 0);
    CHECK_EQ(countNonZero(filled), 121u);
    const ImageU8 b = boundary(filled);
    CHECK_EQ(countNonZero(b), 40u);  // perimeter of an 11x11 square
    // thinning a thick horizontal bar gives a thin line inside it
    ImageU8 bar(30, 9, 1, 0);
    for (int y = 2; y < 7; ++y)
        for (int x = 3; x < 27; ++x) bar(x, y) = 255;
    const ImageU8 sk = thinning(bar);
    CHECK(countNonZero(sk) > 10u);
    for (int x = 0; x < 30; ++x) {
        int col = 0;
        for (int y = 0; y < 9; ++y) col += sk(x, y) ? 1 : 0;
        CHECK(col <= 1);
    }
    for (std::size_t i = 0; i < sk.pixelCount(); ++i) CHECK(!sk.data()[i] || bar.data()[i]);
    // hit-or-miss detecting isolated pixels
    ImageU8 dots(7, 7, 1, 0);
    dots(3, 3) = 255;
    dots(0, 0) = 255;
    dots(1, 0) = 255;
    StructuringElement hit(3, 3, 1, 0), miss(3, 3, 1, 1);
    hit(1, 1) = 1;
    miss(1, 1) = 0;
    const ImageU8 iso = hitOrMiss(dots, hit, miss);
    CHECK_EQ(countNonZero(iso), 1u);
    CHECK_EQ(iso(3, 3), 255);
}
