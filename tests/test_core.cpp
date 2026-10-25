#include <ipf/core.hpp>

#include "test_framework.hpp"

using namespace ipf;

TEST(core_saturate_cast) {
    CHECK_EQ(saturate_cast<std::uint8_t>(300), 255);
    CHECK_EQ(saturate_cast<std::uint8_t>(-5), 0);
    CHECK_EQ(saturate_cast<std::uint8_t>(127.5), 128);
    CHECK_EQ(saturate_cast<std::uint8_t>(127.49), 127);
    CHECK_EQ(saturate_cast<std::uint8_t>(-0.4), 0);
    CHECK_EQ(saturate_cast<std::uint8_t>(1e10), 255);
    CHECK_EQ(saturate_cast<std::uint8_t>(std::nan("")), 0);
    CHECK_EQ(saturate_cast<std::int16_t>(-40000), -32768);
    CHECK_EQ(saturate_cast<std::int16_t>(-2.5), -3);
    CHECK_EQ(saturate_cast<std::uint16_t>(70000u), 65535);
    CHECK_EQ(saturate_cast<std::int8_t>(200u), 127);
    CHECK_EQ(saturate_cast<std::uint32_t>(-1), 0u);
    CHECK_EQ(saturate_cast<std::int32_t>(4e9), 2147483647);
    CHECK_NEAR(saturate_cast<float>(0.25), 0.25, 0);
}

TEST(core_border_index) {
    // OpenCV reference sequences for len = 4 (abcd)
    const int len = 4;
    // p = -3..4
    const int expectReflect[] = {2, 1, 0, 0, 1, 2, 3, 3};     // cba|abcd|d
    const int expectReflect101[] = {3, 2, 1, 0, 1, 2, 3, 2};  // dcb|abcd|c
    const int expectWrap[] = {1, 2, 3, 0, 1, 2, 3, 0};        // bcd|abcd|a
    const int expectReplicate[] = {0, 0, 0, 0, 1, 2, 3, 3};   // aaa|abcd|d
    for (int i = 0; i < 8; ++i) {
        const int p = i - 3;
        CHECK_EQ(borderIndex(p, len, BorderMode::Reflect), expectReflect[i]);
        CHECK_EQ(borderIndex(p, len, BorderMode::Reflect101), expectReflect101[i]);
        CHECK_EQ(borderIndex(p, len, BorderMode::Wrap), expectWrap[i]);
        CHECK_EQ(borderIndex(p, len, BorderMode::Replicate), expectReplicate[i]);
    }
    CHECK_EQ(borderIndex(-1, len, BorderMode::Constant), -1);
    CHECK_EQ(borderIndex(4, len, BorderMode::Constant), -1);
    // far out of range must stay in range
    for (int p = -50; p < 50; ++p) {
        for (BorderMode m : {BorderMode::Reflect, BorderMode::Reflect101, BorderMode::Wrap, BorderMode::Replicate}) {
            for (int n = 1; n < 6; ++n) {
                const int q = borderIndex(p, n, m);
                CHECK(q >= 0 && q < n);
            }
        }
    }
}

TEST(core_image_basics) {
    ImageU8 img(4, 3, 3, 7);
    CHECK_EQ(img.width(), 4);
    CHECK_EQ(img.height(), 3);
    CHECK_EQ(img.channels(), 3);
    CHECK_EQ(img.sampleCount(), 36u);
    CHECK_EQ(img(3, 2, 2), 7);
    img(1, 2, 0) = 42;
    CHECK_EQ(img.data()[(2 * 4 + 1) * 3 + 0], 42);
    CHECK_EQ(img.at(1, 2, 0), 42);
    CHECK_THROWS(img.at(4, 0, 0));
    CHECK_THROWS(img.at(0, 0, 3));
    CHECK_THROWS(ImageU8(-1, 2));
    CHECK_THROWS(ImageU8(2, 2, 0));

    img.fill({1, 2, 3});
    CHECK_EQ(img(2, 1, 0), 1);
    CHECK_EQ(img(2, 1, 1), 2);
    CHECK_EQ(img(2, 1, 2), 3);

    ImageU8 copy = img;
    CHECK(copy == img);
    copy(0, 0, 0) = 99;
    CHECK(copy != img);  // deep copy

    ImageU8 empty;
    CHECK(empty.empty());
}

TEST(core_crop_paste) {
    ImageI32 img(5, 4, 1);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 5; ++x) img(x, y) = y * 10 + x;
    ImageI32 c = img.crop({1, 1, 3, 2});
    CHECK_EQ(c.width(), 3);
    CHECK_EQ(c.height(), 2);
    CHECK_EQ(c(0, 0), 11);
    CHECK_EQ(c(2, 1), 23);
    // crop is clipped to the image
    ImageI32 clipped = img.crop({3, 2, 10, 10});
    CHECK_EQ(clipped.width(), 2);
    CHECK_EQ(clipped.height(), 2);
    CHECK_EQ(clipped(1, 1), 34);
    CHECK(img.crop({10, 10, 2, 2}).empty());

    ImageI32 canvas(4, 4, 1, -1);
    canvas.paste(c, 2, 3);  // partially outside
    CHECK_EQ(canvas(2, 3), 11);
    CHECK_EQ(canvas(3, 3), 12);
    CHECK_EQ(canvas(1, 3), -1);
    canvas.paste(c, -2, -1);
    CHECK_EQ(canvas(0, 0), 23);
}

TEST(core_atborder) {
    ImageU8 img(3, 1, 1);
    img(0, 0) = 10;
    img(1, 0) = 20;
    img(2, 0) = 30;
    CHECK_EQ(img.atBorder(-1, 0, 0, BorderMode::Constant, 5), 5);
    CHECK_EQ(img.atBorder(-1, 0, 0, BorderMode::Replicate), 10);
    CHECK_EQ(img.atBorder(-1, 0, 0, BorderMode::Reflect), 10);
    CHECK_EQ(img.atBorder(-1, 0, 0, BorderMode::Reflect101), 20);
    CHECK_EQ(img.atBorder(3, 0, 0, BorderMode::Wrap), 10);
    CHECK_EQ(img.atBorder(1, 5, 0, BorderMode::Replicate), 20);
}

TEST(core_convert) {
    ImageU8 u(2, 1, 1);
    u(0, 0) = 0;
    u(1, 0) = 255;
    ImageF f = toFloat(u);
    CHECK_NEAR(f(0, 0), 0.0, 1e-7);
    CHECK_NEAR(f(1, 0), 1.0, 1e-7);
    ImageU16 w = convertRange<std::uint16_t>(u);
    CHECK_EQ(w(1, 0), 65535);
    ImageF f2(1, 1, 1, 1.7f);
    CHECK_EQ(toU8(f2)(0, 0), 255);
    ImageF f3(1, 1, 1, 0.5f);
    CHECK_EQ(toU8(f3)(0, 0), 128);
    ImageI32 i = u.convertTo<std::int32_t>(2.0, -10.0);
    CHECK_EQ(i(1, 0), 500);
    CHECK_EQ(i(0, 0), -10);
    auto doubled = mapSamples(u, [](std::uint8_t v) { return v * 2; });
    CHECK_EQ(doubled(1, 0), 510);
}

TEST(core_channels) {
    ImageU8 rgb(2, 2, 3);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
            rgb(x, y, 0) = static_cast<std::uint8_t>(x);
            rgb(x, y, 1) = static_cast<std::uint8_t>(10 + y);
            rgb(x, y, 2) = 200;
        }
    auto planes = splitChannels(rgb);
    CHECK_EQ(planes.size(), 3u);
    CHECK_EQ(planes[1](1, 1), 11);
    CHECK(mergeChannels(planes) == rgb);
    ImageU8 bgr = reorderChannels(rgb, {2, 1, 0});
    CHECK_EQ(bgr(1, 0, 0), 200);
    CHECK_EQ(bgr(1, 0, 2), 1);
    ImageU8 g = extractChannel(rgb, 2);
    CHECK_EQ(g.channels(), 1);
    g.fill(5);
    insertChannel(rgb, g, 0);
    CHECK_EQ(rgb(1, 1, 0), 5);
    CHECK_THROWS(extractChannel(rgb, 3));
    CHECK_THROWS(mergeChannels(std::vector<ImageU8>{ImageU8(2, 2), ImageU8(3, 2)}));
}

TEST(core_statistics) {
    ImageF img(3, 2, 1);
    float v = 0;
    for (auto& s : img) s = v++;  // 0..5
    auto mm = minMaxLoc(img);
    CHECK_NEAR(mm.minVal, 0, 0);
    CHECK_NEAR(mm.maxVal, 5, 0);
    CHECK(mm.maxLoc == Point(2, 1));
    CHECK_NEAR(mean(img)[0], 2.5, 1e-12);
    CHECK_NEAR(stddev(img)[0], std::sqrt(17.5 / 6.0), 1e-12);
    CHECK_NEAR(sum(img)[0], 15.0, 1e-12);
    CHECK_EQ(countNonZero(img), 5u);
}

TEST(core_rect_ops) {
    Rect a(0, 0, 10, 10), b(5, 5, 10, 10);
    CHECK((a & b) == Rect(5, 5, 5, 5));
    CHECK((a | b) == Rect(0, 0, 15, 15));
    CHECK((a & Rect(20, 20, 1, 1)).empty());
    CHECK(a.contains({9, 9}));
    CHECK(!a.contains({10, 9}));
}

TEST(core_from_data_and_fill_ctor) {
    ImageU8 zero(2, 2, 3, 0);  // must not be ambiguous with a pointer constructor
    CHECK_EQ(zero(1, 1, 2), 0);
    const float buf[6] = {1, 2, 3, 4, 5, 6};
    const ImageF f = ImageF::fromData(3, 2, 1, buf);
    CHECK_NEAR(f(2, 1), 6, 0);
    CHECK_THROWS(ImageF::fromData(1, 1, 1, nullptr));
    CHECK_THROWS(ImageF(2, 2, 1, std::vector<float>(3)));
}
