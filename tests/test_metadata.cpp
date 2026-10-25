#include <ipf/geometry.hpp>
#include <ipf/io.hpp>

#include <string>

#include "test_framework.hpp"
#include "test_util.hpp"

using namespace ipf;
using namespace ipftest;

#ifndef IPF_TEST_DATA_DIR
#define IPF_TEST_DATA_DIR "tests/data"
#endif

namespace {
/// Minimal little-endian EXIF (TIFF) block holding only the orientation tag.
std::vector<std::uint8_t> exifWithOrientation(int o) {
    return {'I', 'I', 42, 0, 8, 0, 0, 0,                        // header, IFD0 at offset 8
            1, 0,                                               // one entry
            0x12, 0x01, 3, 0, 1, 0, 0, 0, static_cast<std::uint8_t>(o), 0, 0, 0,  // orientation SHORT
            0, 0, 0, 0};                                        // no next IFD
}
ImageU16 readReference(const std::string& name) {
    return decodePNM<std::uint16_t>(readFile(std::string(IPF_TEST_DATA_DIR) + "/" + name));
}
}  // namespace

TEST(meta_exif_orientation) {
    const ImageI32 a = [] {
        ImageI32 img(4, 3, 1);
        for (int i = 0; i < 12; ++i) img.data()[i] = i;
        return img;
    }();
    CHECK(applyExifOrientation(a, 1) == a);
    CHECK(applyExifOrientation(a, 2) == flipHorizontal(a));
    CHECK(applyExifOrientation(a, 3) == rotate90(a, 2));
    CHECK(applyExifOrientation(a, 4) == flipVertical(a));
    CHECK(applyExifOrientation(a, 5) == transpose(a));
    CHECK(applyExifOrientation(a, 6) == rotate90(a, -1));  // 90 degrees clockwise
    CHECK(applyExifOrientation(a, 7) == rotate90(transpose(a), 2));
    CHECK(applyExifOrientation(a, 8) == rotate90(a, 1));
    for (int o = 1; o <= 8; ++o) CHECK_EQ(detail::exifOrientation(exifWithOrientation(o)), o);

    // a JPEG carrying orientation 6 is loaded upright by default
    const ImageU8 img = smoothImage(40, 24, 3);
    ImageMetadata m;
    m.exif = exifWithOrientation(6);
    JpegOptions jo;
    jo.quality = 95;
    jo.metadata = &m;
    const auto jpg = encodeJPEG(img, jo);
    ImageMetadata got;
    const ImageU8 upright = decodeImage(jpg, LoadOptions{}, &got);
    CHECK_EQ(upright.width(), 24);
    CHECK_EQ(upright.height(), 40);
    CHECK_EQ(got.orientation, 6);
    CHECK_EQ(detail::exifOrientation(got.exif), 1);  // patched so re-saving does not rotate twice
    CHECK(psnrOf(upright, rotate90(img, -1)) > 35);
    LoadOptions raw;
    raw.applyOrientation = false;
    CHECK_EQ(decodeImage(jpg, raw).width(), 40);
}

TEST(meta_png_roundtrip) {
    ImageMetadata m;
    m.text = {{"Title", "ipf test"}, {"Comment", "na\xC3\xAFve caf\xC3\xA9"}};  // ASCII + UTF-8
    m.dpiX = 300;
    m.dpiY = 150;
    m.gamma = 0.45455;
    m.iccProfile.assign(3000, 7);
    m.exif = exifWithOrientation(3);
    PngOptions o;
    o.metadata = &m;
    const ImageU8 img = randomImage<std::uint8_t>(9, 7, 3, 1);
    ImageMetadata r;
    const ImageU8 back = decodePNG(encodePNG(img, o), &r);
    CHECK(back == img);
    CHECK_EQ(r.text.size(), 2u);
    CHECK(r.text[0].first == "Title" && r.text[0].second == "ipf test");
    CHECK(r.text[1].second == m.text[1].second);
    CHECK_NEAR(r.dpiX, 300, 0.02);  // PNG stores whole pixels per meter
    CHECK_NEAR(r.dpiY, 150, 0.02);
    CHECK_NEAR(r.gamma, 0.45455, 1e-5);
    CHECK(r.iccProfile == m.iccProfile);
    CHECK_EQ(r.orientation, 3);
    CHECK_EQ(r.bitDepth, 8);
}

TEST(meta_jpeg_roundtrip) {
    ImageMetadata m;
    m.iccProfile.resize(150000);  // larger than one APP2 segment -> split into 3 chunks
    for (std::size_t i = 0; i < m.iccProfile.size(); ++i) m.iccProfile[i] = static_cast<std::uint8_t>(i * 31);
    m.text = {{"comment", "made with ipf"}};
    m.dpiX = m.dpiY = 96;
    m.exif = exifWithOrientation(1);
    JpegOptions o;
    o.metadata = &m;
    ImageMetadata r;
    decodeJPEG(encodeJPEG(smoothImage(16, 16, 3), o), &r);
    CHECK(r.iccProfile == m.iccProfile);
    CHECK_EQ(r.text.size(), 1u);
    CHECK(r.text[0].second == "made with ipf");
    CHECK_NEAR(r.dpiX, 96, 1e-9);
    CHECK(r.exif == m.exif);
}

TEST(png_encoder_optimizations) {
    // binary mask -> 1-bit gray
    ImageU8 mask(37, 11, 1, 0);
    for (int x = 5; x < 20; ++x) mask(x, 4) = 255;
    auto png = encodePNG(mask);
    CHECK_EQ(png[24], 1);  // IHDR bit depth
    CHECK_EQ(png[25], 0);  // gray
    CHECK(decodePNG(png) == mask);
    // 16 gray levels -> 4-bit
    ImageU8 g4(10, 10, 1);
    for (int i = 0; i < 100; ++i) g4.data()[i] = static_cast<std::uint8_t>((i % 16) * 17);
    png = encodePNG(g4);
    CHECK_EQ(png[24], 4);
    CHECK(decodePNG(png) == g4);
    // few colors -> palette (RGB keeps 3 channels, RGBA keeps 4 channels incl. alpha)
    ImageU8 pal(20, 20, 3);
    for (int i = 0; i < 400; ++i) pal.data()[i * 3] = static_cast<std::uint8_t>((i % 5) * 50), pal.data()[i * 3 + 2] = 9;
    png = encodePNG(pal);
    CHECK_EQ(png[25], 3);
    CHECK_EQ(png[24], 4);  // 5 colors -> 4-bit indices
    CHECK(decodePNG(png) == pal);
    ImageU8 rgba = convertChannels(pal, 4);
    rgba(3, 3, 3) = 77;
    rgba(4, 4, 3) = 0;
    png = encodePNG(rgba);
    CHECK_EQ(png[25], 3);
    CHECK(decodePNG(png) == rgba);
    const ImageU8 opaque = convertChannels(pal, 4);
    CHECK(decodePNG(encodePNG(opaque)) == opaque);
    // optimization off keeps the plain color type; big random images are never palettized
    PngOptions plain;
    plain.optimize = false;
    CHECK_EQ(encodePNG(pal, plain)[25], 2);
    CHECK_EQ(encodePNG(randomImage<std::uint8_t>(40, 40, 3, 2))[25], 2);
    // Adam7 interlaced output for every layout
    PngOptions il;
    il.interlace = true;
    for (int c = 1; c <= 4; ++c) {
        const ImageU8 a = randomImage<std::uint8_t>(13, 9, c, static_cast<unsigned>(c));
        CHECK(decodePNG(encodePNG(a, il)) == a);
    }
    CHECK(decodePNG(encodePNG(mask, il)) == mask);
    CHECK(decodePNG(encodePNG(rgba, il)) == rgba);
    const ImageU16 w = randomImage<std::uint16_t>(11, 6, 2, 3);
    CHECK(decodePNG16(encodePNG(w, il)) == w);
}

TEST(apng_roundtrip) {
    std::vector<AnimationFrame> frames;
    for (int i = 0; i < 4; ++i) {
        ImageU8 f(30, 20, 4, 0);
        for (int y = 5; y < 15; ++y)
            for (int x = 5 + i * 3; x < 15 + i * 3; ++x) {
                f(x, y, 0) = static_cast<std::uint8_t>(60 * i);
                f(x, y, 1) = 200;
                f(x, y, 3) = static_cast<std::uint8_t>(i == 2 ? 128 : 255);
            }
        frames.push_back({f, 0.05 * (i + 1)});
    }
    int loops = -1;
    const auto back = decodeAPNG(encodeAPNG(frames, 3), &loops);
    CHECK_EQ(loops, 3);
    CHECK_EQ(back.size(), 4u);
    for (std::size_t i = 0; i < back.size(); ++i) {
        CHECK(back[i].image == frames[i].image);
        CHECK_NEAR(back[i].delay, frames[i].delay, 1e-9);
    }
    // a still PNG is a one-frame animation
    const auto still = decodeAPNG(encodePNG(smoothImage(5, 5, 3)));
    CHECK_EQ(still.size(), 1u);
    CHECK_EQ(still[0].image.channels(), 4);
}

TEST(jpeg_extended_modes_reference_files) {
    // files created by libjpeg-turbo's cjpeg; references decoded by its djpeg -dct float
    struct Case {
        const char* name;
        int precision;
        int tol;
    } cases[] = {{"p12_arith_prog", 12, 3}, {"lossless16_gray", 16, 0}, {"lossless8_rgb", 8, 0}, {"arith8", 8, 3}};
    for (const Case& c : cases) {
        const std::string base = std::string(IPF_TEST_DATA_DIR) + "/" + c.name;
        int precision = 0;
        ImageMetadata meta;
        decodeJPEGRaw(readFile(base + ".jpg"), precision, &meta);
        CHECK_EQ(precision, c.precision);
        CHECK_EQ(meta.bitDepth, c.precision);
        // both sides scaled to the full 16-bit range: v * 65535 / (2^precision - 1)
        const ImageU16 got = decodeJPEG16(readFile(base + ".jpg"));
        const ImageU16 want = readReference(std::string(c.name) + ".pnm");
        CHECK(got.sameShape(want));
        const double scale = 65535.0 / ((1 << c.precision) - 1);
        if (got.sameShape(want)) CHECK(maxAbsDiff(got, want) <= c.tol * scale + 0.5);
    }
    // 16-bit loading keeps the precision; 8-bit loading scales it
    const ImageU16 l16 = loadImage16(std::string(IPF_TEST_DATA_DIR) + "/lossless16_gray.jpg");
    CHECK(maxAbsDiff(l16, readReference("lossless16_gray.pnm")) == 0.0);
    CHECK_EQ(loadImage(std::string(IPF_TEST_DATA_DIR) + "/p12_arith_prog.jpg").channels(), 3);
}

TEST(icc_profiles) {
    const IccProfile s = IccProfile::sRGB();
    CHECK(s.valid() && s.isMatrixShaper());
    double white[3] = {1, 1, 1}, xyz[3], back[3];
    s.toXYZ(white, xyz);
    CHECK_NEAR(xyz[0], 0.9642, 2e-4);  // D50 white point
    CHECK_NEAR(xyz[1], 1.0, 2e-4);
    CHECK_NEAR(xyz[2], 0.8249, 2e-4);
    const double mid[3] = {0.2, 0.5, 0.8};
    s.toXYZ(mid, xyz);
    s.fromXYZ(xyz, back);
    for (int k = 0; k < 3; ++k) CHECK_NEAR(back[k], mid[k], 1e-9);
    // sRGB -> sRGB is the identity; neutral grays stay neutral in Adobe RGB / Display P3
    const ImageU8 img = smoothImage(20, 12, 3);
    CHECK(maxAbsDiff(convertColorProfile(img, s, s), img) <= 1.0);
    ImageU8 gray(1, 1, 3);
    gray.fill({128, 128, 128});
    for (const IccProfile& p : {IccProfile::adobeRGB(), IccProfile::displayP3()}) {
        const ImageU8 g = convertColorProfile(gray, p);
        CHECK(std::abs(g(0, 0, 0) - g(0, 0, 2)) <= 1);
    }
    // saturated Display P3 green is outside sRGB: clipped to the gamut boundary
    ImageU8 p3green(1, 1, 3);
    p3green.fill({0, 255, 0});
    const ImageU8 clipped = convertColorProfile(p3green, IccProfile::displayP3());
    CHECK_EQ(clipped(0, 0, 1), 255);
    CHECK_EQ(clipped(0, 0, 0), 0);
    // alpha passes through
    ImageU8 rgba = convertChannels(img, 4);
    rgba(3, 3, 3) = 42;
    CHECK_EQ(convertColorProfile(rgba, IccProfile::adobeRGB())(3, 3, 3), 42);
    CHECK_THROWS(IccProfile::fromBytes(std::vector<std::uint8_t>(100, 0)));
    CHECK_THROWS(convertColorProfile(ImageU8(2, 2, 1), s));
}

TEST(fuzz_new_parsers_throw_not_crash) {
    // Corrupted extended JPEGs, APNGs, metadata-carrying PNGs and ICC profiles must raise
    // ipf::Error (or decode), never crash or read out of bounds.
    std::vector<std::vector<std::uint8_t>> files;
    for (const char* f : {"p12_arith_prog.jpg", "lossless16_gray.jpg", "lossless8_rgb.jpg", "arith8.jpg"})
        files.push_back(readFile(std::string(IPF_TEST_DATA_DIR) + "/" + f));
    std::vector<AnimationFrame> frames = {{smoothImage(12, 9, 4), 0.1}, {randomImage<std::uint8_t>(12, 9, 4, 2), 0.2}};
    files.push_back(encodeAPNG(frames));
    ImageMetadata m;
    m.text = {{"k", "v\xC3\xA9"}};
    m.iccProfile.assign(400, 3);
    m.exif = exifWithOrientation(6);
    PngOptions po;
    po.metadata = &m;
    po.interlace = true;
    files.push_back(encodePNG(smoothImage(15, 10, 3), po));
    std::mt19937 rng(77);
    int runs = 0;
    for (const auto& f : files) {
        for (int trial = 0; trial < 400; ++trial) {
            std::vector<std::uint8_t> t = f;
            if (trial % 5 == 0) {
                t.resize(rng() % t.size());
            } else {
                const int flips = 1 + static_cast<int>(rng() % 6);
                for (int k = 0; k < flips; ++k) t[rng() % t.size()] = static_cast<std::uint8_t>(rng());
            }
            try {
                LoadOptions o;
                o.convertToSRGB = true;
                (void)decodeImage(t, o);
            } catch (const Error&) {
            }
            try {
                (void)decodeAPNG(t);
            } catch (const Error&) {
            }
            ++runs;
        }
    }
    // ICC profile parsing: libjpeg-turbo's test3.icc (RGB, lut8 tables), randomly corrupted
    std::vector<std::vector<std::uint8_t>> profiles;
    const std::string ljt = std::string(IPF_TEST_DATA_DIR) + "/test3.icc";
    profiles.push_back(readFile(ljt));
    for (const auto& p : profiles)
        for (int trial = 0; trial < 300; ++trial) {
            std::vector<std::uint8_t> t = p;
            for (int k = 0; k < 4; ++k) t[rng() % t.size()] = static_cast<std::uint8_t>(rng());
            if (trial % 7 == 0) t.resize(rng() % t.size());
            try {
                const IccProfile prof = IccProfile::fromBytes(t);
                (void)convertColorProfile(smoothImage(6, 4, prof.channels() == 1 ? 1 : 3), prof);
            } catch (const Error&) {
            }
            ++runs;
        }
    CHECK_EQ(runs, 6 * 400 + 300);
}
