#include <ipf/io.hpp>

#include <cstdio>
#include <string>

#include "test_framework.hpp"
#include "test_util.hpp"

using namespace ipf;
using namespace ipftest;

namespace {
std::vector<std::uint8_t> bytesOf(const std::string& s) { return std::vector<std::uint8_t>(s.begin(), s.end()); }
}  // namespace

TEST(io_zlib_roundtrip) {
    std::mt19937 rng(7);
    std::vector<std::vector<std::uint8_t>> inputs;
    inputs.push_back({});
    inputs.push_back({42});
    inputs.push_back(bytesOf("abcabcabcabcabcabcabcabcabcabcabcabcabcabc"));
    std::vector<std::uint8_t> random(100000), runs(300000), text;
    for (auto& b : random) b = static_cast<std::uint8_t>(rng());
    for (std::size_t i = 0; i < runs.size(); ++i) runs[i] = static_cast<std::uint8_t>((i / 1000) % 7);
    for (int i = 0; i < 5000; ++i) {
        const std::string w = "word" + std::to_string(rng() % 300) + " ";
        text.insert(text.end(), w.begin(), w.end());
    }
    inputs.push_back(random);
    inputs.push_back(runs);
    inputs.push_back(text);
    for (const auto& in : inputs)
        for (int level : {0, 1, 4, 6, 9}) {
            const auto z = zlibCompress(in.data(), in.size(), level);
            const auto back = zlibDecompress(z.data(), z.size());
            CHECK(back == in);
        }
    // Compression must actually compress redundant data.
    const auto zr = zlibCompress(runs.data(), runs.size(), 6);
    CHECK(zr.size() < runs.size() / 50);
    const auto zt = zlibCompress(text.data(), text.size(), 9);
    CHECK(zt.size() < text.size() / 2);
    // Incompressible data must not blow up (stored block fallback).
    const auto zrand = zlibCompress(random.data(), random.size(), 6);
    CHECK(zrand.size() < random.size() + random.size() / 100 + 64);
}

TEST(io_zlib_known_stream) {
    // zlib.compress(b"hello hello hello hello", 9) from CPython
    const std::vector<std::uint8_t> z = {0x78, 0xda, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40,
                                         0x27, 0x01, 0x68, 0x03, 0x08, 0xb1};
    const auto out = zlibDecompress(z.data(), z.size());
    CHECK(out == bytesOf("hello hello hello hello"));
    CHECK_EQ(crc32(reinterpret_cast<const std::uint8_t*>("123456789"), 9), 0xCBF43926u);
    CHECK_EQ(adler32(reinterpret_cast<const std::uint8_t*>("Wikipedia"), 9), 0x11E60398u);
    // Corrupt checksum must be detected.
    auto bad = z;
    bad.back() ^= 1;
    CHECK_THROWS(zlibDecompress(bad.data(), bad.size()));
}

TEST(io_png_roundtrip) {
    for (int c = 1; c <= 4; ++c)
        for (Size s : {Size(1, 1), Size(7, 3), Size(64, 33)}) {
            const ImageU8 a = randomImage<std::uint8_t>(s.width, s.height, c, static_cast<unsigned>(c * 100 + s.width));
            CHECK(decodePNG(encodePNG(a)) == a);
            CHECK(decodePNG(encodePNG(a, 0)) == a);
            const ImageU8 sm = smoothImage(s.width, s.height, c);
            CHECK(decodePNG(encodePNG(sm, 9)) == sm);
            const ImageU16 w = randomImage<std::uint16_t>(s.width, s.height, c, 5);
            CHECK(decodePNG16(encodePNG(w)) == w);
        }
    // 8-bit PNG read as 16-bit is scaled by 257
    ImageU8 g(2, 1, 1);
    g(0, 0) = 0;
    g(1, 0) = 255;
    const ImageU16 g16 = decodePNG16(encodePNG(g));
    CHECK_EQ(g16(1, 0), 65535);
    // smooth content compresses well
    const ImageU8 big = smoothImage(256, 256, 3);
    CHECK(encodePNG(big).size() < big.sampleCount() / 2);
}

TEST(io_bmp_roundtrip) {
    for (int c : {1, 3, 4}) {
        const ImageU8 a = randomImage<std::uint8_t>(13, 6, c, static_cast<unsigned>(c));
        CHECK(decodeBMP(encodeBMP(a)) == a);
    }
    const ImageU8 ga = randomImage<std::uint8_t>(5, 5, 2, 3);
    CHECK(decodeBMP(encodeBMP(ga)) == convertChannels(ga, 4));
}

TEST(io_bmp_rle8) {
    // 4x2 RLE8 image, bottom-up: row0 (bottom) = 3x idx1 + idx2, row1 = idx0,idx1,idx2,idx0 (absolute run)
    std::vector<std::uint8_t> f = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    detail::put32le(f, 40);
    detail::put32le(f, 4);
    detail::put32le(f, 2);
    detail::put16le(f, 1);
    detail::put16le(f, 8);
    detail::put32le(f, 1);  // BI_RLE8
    for (int i = 0; i < 3; ++i) detail::put32le(f, 0);
    detail::put32le(f, 3);  // colors used
    detail::put32le(f, 0);
    const std::uint8_t pal[] = {0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 0};  // BGRx: red, green, blue
    f.insert(f.end(), pal, pal + 12);
    const std::uint32_t off = static_cast<std::uint32_t>(f.size());
    f[10] = static_cast<std::uint8_t>(off);
    const std::uint8_t data[] = {3, 1, 1, 2, 0, 0, 0, 4, 0, 1, 2, 0, 0, 1};
    f.insert(f.end(), data, data + sizeof(data));
    const ImageU8 img = decodeBMP(f);
    CHECK_EQ(img.width(), 4);
    CHECK_EQ(img.channels(), 3);
    CHECK_EQ(img(0, 1, 1), 255);  // bottom row, idx 1 = green
    CHECK_EQ(img(3, 1, 2), 255);  // idx 2 = blue
    CHECK_EQ(img(0, 0, 0), 255);  // top row starts with idx 0 = red
    CHECK_EQ(img(2, 0, 2), 255);
}

TEST(io_tga_roundtrip) {
    for (int c = 1; c <= 4; ++c)
        for (bool rle : {false, true}) {
            ImageU8 a = randomImage<std::uint8_t>(17, 9, c, static_cast<unsigned>(c));
            // add runs so RLE packets are exercised
            for (int x = 3; x < 12; ++x)
                for (int k = 0; k < c; ++k) a(x, 4, k) = 77;
            if (c == 2 || c == 4)  // make alpha non-trivial
                a(0, 0, c - 1) = 128;
            CHECK(decodeTGA(encodeTGA(a, rle)) == a);
        }
    const ImageU8 flat(100, 50, 3, 9);
    CHECK(encodeTGA(flat, true).size() < 2000);
}

TEST(io_pnm) {
    const auto p2 = decodePNM(bytesOf("P2\n# comment\n3 2\n# another\n10\n0 5 10\n10 5 0\n"));
    CHECK_EQ(p2.width(), 3);
    CHECK_EQ(p2(1, 0), 128);  // 5/10 scaled to 255
    CHECK_EQ(p2(0, 1), 255);
    const auto p1 = decodePNM(bytesOf("P1 4 1 0101"));
    CHECK_EQ(p1(0, 0), 255);
    CHECK_EQ(p1(1, 0), 0);
    std::vector<std::uint8_t> p4 = bytesOf("P4\n10 1\n");
    p4.push_back(0xA0);
    p4.push_back(0x40);
    const auto bits = decodePNM(p4);
    CHECK_EQ(bits(0, 0), 0);
    CHECK_EQ(bits(1, 0), 255);
    CHECK_EQ(bits(2, 0), 0);
    CHECK_EQ(bits(9, 0), 0);
    CHECK_EQ(bits(8, 0), 255);
    for (int c : {1, 3}) {
        const ImageU8 a = randomImage<std::uint8_t>(9, 4, c, 2);
        CHECK(decodePNM(encodePNM(a)) == a);
        CHECK(decodePNM(encodePNM(a, false)) == a);
        const ImageU16 w = randomImage<std::uint16_t>(9, 4, c, 3);
        CHECK(decodePNM<std::uint16_t>(encodePNM(w)) == w);
    }
    CHECK_THROWS(decodePNM(bytesOf("P5\n3 3\n255\n12")));  // truncated
}

TEST(io_pfm) {
    for (int c : {1, 3}) {
        ImageF a = randomImage<float>(6, 5, c, 4);
        a(0, 0, 0) = -3.5f;
        a(5, 4, 0) = 1e6f;
        CHECK(decodePFM(encodePFM(a)) == a);
    }
}

TEST(io_jpeg_roundtrip) {
    const ImageU8 rgb = smoothImage(67, 45, 3);
    for (bool sub : {true, false}) {
        const ImageU8 back = decodeJPEG(encodeJPEG(rgb, 95, sub));
        CHECK_EQ(back.channels(), 3);
        CHECK(back.size() == rgb.size());
        CHECK(psnrOf(back, rgb) > (sub ? 33.0 : 38.0));
    }
    const ImageU8 gray = smoothImage(30, 17, 1);
    const ImageU8 gback = decodeJPEG(encodeJPEG(gray, 90));
    CHECK_EQ(gback.channels(), 1);
    CHECK(psnrOf(gback, gray) > 38.0);
    // quality ordering: higher quality -> larger file, better PSNR
    const auto q30 = encodeJPEG(rgb, 30), q95 = encodeJPEG(rgb, 95);
    CHECK(q30.size() < q95.size());
    CHECK(psnrOf(decodeJPEG(q30), rgb) < psnrOf(decodeJPEG(q95), rgb));
    // flat colors survive exactly-ish
    ImageU8 flat(16, 16, 3);
    flat.fill({200, 30, 90});
    CHECK(maxAbsDiff(decodeJPEG(encodeJPEG(flat, 100, false)), flat) <= 2.0);
}

TEST(io_jpeg_tables_are_consistent) {
    // Each AC table must contain all 162 (run,size) symbols exactly once.
    for (const std::uint8_t* vals : {detail::jpg::kAcLumVals, detail::jpg::kAcChromVals}) {
        std::vector<int> seen(256, 0);
        for (int i = 0; i < 162; ++i) ++seen[vals[i]];
        int ok = 0;
        for (int r = 0; r < 16; ++r)
            for (int s = 1; s <= 10; ++s) ok += seen[(r << 4) | s] == 1;
        CHECK_EQ(ok, 160);
        CHECK_EQ(seen[0x00], 1);
        CHECK_EQ(seen[0xF0], 1);
    }
}

TEST(io_detect_and_channels) {
    CHECK(detectFormat(encodePNG(ImageU8(2, 2))) == ImageFormat::PNG);
    CHECK(detectFormat(encodeJPEG(ImageU8(2, 2))) == ImageFormat::JPEG);
    CHECK(detectFormat(encodeBMP(ImageU8(2, 2))) == ImageFormat::BMP);
    CHECK(detectFormat(encodePNM(ImageU8(2, 2))) == ImageFormat::PNM);
    CHECK(detectFormat(encodePFM(ImageF(2, 2))) == ImageFormat::PFM);
    CHECK(formatFromExtension("a/b.JPEG") == ImageFormat::JPEG);
    CHECK(formatFromExtension("x.tga") == ImageFormat::TGA);
    CHECK(formatFromExtension("noext") == ImageFormat::Unknown);

    ImageU8 rgb(1, 1, 3);
    rgb.fill({255, 0, 0});
    CHECK_EQ(convertChannels(rgb, 1)(0, 0), 76);  // 0.299 * 255
    CHECK_EQ(convertChannels(rgb, 4)(0, 0, 3), 255);
    CHECK_EQ(convertChannels(convertChannels(rgb, 1), 3)(0, 0, 2), 76);
    const ImageU8 rgba = decodeImage(encodePNG(rgb), 4);
    CHECK_EQ(rgba.channels(), 4);
}

TEST(io_file_roundtrip) {
    const ImageU8 a = smoothImage(20, 10, 3);
    const std::string base = "ipf_test_tmp";
    for (const char* ext : {".png", ".bmp", ".tga", ".ppm"}) {
        saveImage(a, base + ext);
        CHECK(loadImage(base + ext) == a);
        std::remove((base + ext).c_str());
    }
    saveImage(a, base + ".jpg", 95);
    CHECK(psnrOf(loadImage(base + ".jpg"), a) > 33);
    std::remove((base + ".jpg").c_str());
    const ImageF f = toFloat(a);
    saveImage(f, base + ".pfm");
    CHECK(maxAbsDiff(loadImageF(base + ".pfm"), f) == 0.0);
    std::remove((base + ".pfm").c_str());
    const ImageU16 w = randomImage<std::uint16_t>(5, 5, 1, 1);
    saveImage(w, base + ".png");
    CHECK(loadImage16(base + ".png") == w);
    std::remove((base + ".png").c_str());
    CHECK_THROWS(loadImage("does/not/exist.png"));
    CHECK_THROWS(saveImage(a, base + ".xyz"));
}

TEST(io_corrupt_inputs_throw_not_crash) {
    // Truncations and random byte flips must raise ipf::Error (or decode), never crash.
    const ImageU8 a = smoothImage(40, 30, 3);
    std::vector<std::vector<std::uint8_t>> files = {encodePNG(a), encodeJPEG(a, 80), encodeBMP(a), encodeTGA(a),
                                                    encodePNM(a)};
    std::mt19937 rng(123);
    int decodedOrThrew = 0;
    for (const auto& f : files) {
        for (std::size_t cut : {std::size_t(0), std::size_t(1), f.size() / 3, f.size() / 2, f.size() - 1}) {
            std::vector<std::uint8_t> t(f.begin(), f.begin() + static_cast<std::ptrdiff_t>(cut));
            try {
                (void)decodeImage(t, 0, ImageFormat::TGA);
            } catch (const Error&) {
            }
            ++decodedOrThrew;
        }
        for (int trial = 0; trial < 300; ++trial) {
            std::vector<std::uint8_t> t = f;
            const int flips = 1 + static_cast<int>(rng() % 8);
            for (int k = 0; k < flips; ++k) t[rng() % t.size()] = static_cast<std::uint8_t>(rng());
            try {
                (void)decodeImage(t, 0, ImageFormat::TGA);
            } catch (const Error&) {
            }
            ++decodedOrThrew;
        }
    }
    CHECK_EQ(decodedOrThrew, 5 * (5 + 300));
}
