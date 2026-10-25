// Cross-validation driver: runs one ipf operation on .npy / image inputs and writes .npy
// results, so tests/crossval/run_crossval.py can compare them with numpy/scipy/Pillow.
//
// Usage: ipf_crossval <command> <args...>     (see the command table below)
#include <ipf/analysis.hpp>
#include <ipf/color.hpp>
#include <ipf/features.hpp>
#include <ipf/fft.hpp>
#include <ipf/metrics.hpp>
#include <ipf/morphology.hpp>
#include <ipf/filter.hpp>
#include <ipf/geometry.hpp>
#include <ipf/icc.hpp>
#include <ipf/io.hpp>
#include <ipf/pixel_ops.hpp>

#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "npy.hpp"

using namespace ipf;
using Args = std::vector<std::string>;

namespace {

double num(const Args& a, std::size_t i, double def) { return i < a.size() ? std::stod(a[i]) : def; }

BorderMode borderOf(const std::string& s) {
    if (s == "constant") return BorderMode::Constant;
    if (s == "replicate") return BorderMode::Replicate;
    if (s == "reflect") return BorderMode::Reflect;
    if (s == "wrap") return BorderMode::Wrap;
    return BorderMode::Reflect101;
}
Interpolation interpOf(const std::string& s) {
    if (s == "nearest") return Interpolation::Nearest;
    if (s == "bicubic") return Interpolation::Bicubic;
    if (s == "lanczos") return Interpolation::Lanczos;
    if (s == "area") return Interpolation::Area;
    if (s == "hamming") return Interpolation::Hamming;
    return Interpolation::Bilinear;
}
std::vector<double> vecOf(const std::string& path) {
    const ImageD v = npy::load<double>(path);
    return std::vector<double>(v.begin(), v.end());
}

std::map<std::string, std::function<void(const Args&)>>& commands() {
    static std::map<std::string, std::function<void(const Args&)>> c;
    return c;
}
struct Reg {
    Reg(const char* name, std::function<void(const Args&)> fn) { commands()[name] = std::move(fn); }
};
#define CROSSVAL(name) \
    static void cv_##name(const Args& a); \
    static Reg reg_##name(#name, cv_##name); \
    static void cv_##name(const Args& a)

// ---- codecs ---------------------------------------------------------------
// decode <file> <out.npy>
CROSSVAL(decode) { npy::save(loadImage(a[0]), a[1]); }
// decode16 <file> <out.npy>
CROSSVAL(decode16) { npy::save(loadImage16(a[0]), a[1]); }
// encode <in.npy> <out.ext> [quality] [subsample 0/1]   (uint16 input -> 16-bit PNG/PNM)
CROSSVAL(encode) {
    const npy::Array raw = npy::loadRaw(a[0]);
    const std::string ext = a[1].substr(a[1].find_last_of('.') + 1);
    if (raw.descr == "<u2") {
        saveImage(npy::load<std::uint16_t>(a[0]), a[1]);
    } else if (ext == "jpg") {
        writeFile(a[1], encodeJPEG(npy::load<std::uint8_t>(a[0]), static_cast<int>(num(a, 2, 90)), num(a, 3, 1) != 0));
    } else if (ext == "tga") {
        writeFile(a[1], encodeTGA(npy::load<std::uint8_t>(a[0]), num(a, 2, 1) != 0));
    } else {
        saveImage(npy::load<std::uint8_t>(a[0]), a[1]);
    }
}
// zinflate <in.zlib> <out.bin>  /  zdeflate <in.bin> <out.zlib> <level>
CROSSVAL(zinflate) {
    const auto b = readFile(a[0]);
    writeFile(a[1], zlibDecompress(b.data(), b.size()));
}
CROSSVAL(zdeflate) {
    const auto b = readFile(a[0]);
    writeFile(a[1], zlibCompress(b.data(), b.size(), static_cast<int>(num(a, 2, 6))));
}

// ---- color ----------------------------------------------------------------
CROSSVAL(gray) { npy::save(rgbToGray(npy::load<std::uint8_t>(a[0])), a[1]); }
CROSSVAL(ycbcr) { npy::save(rgbToYCbCr(npy::load<std::uint8_t>(a[0])), a[1]); }
CROSSVAL(ycbcr_inv) { npy::save(yCbCrToRgb(npy::load<std::uint8_t>(a[0])), a[1]); }
CROSSVAL(hsv) { npy::save(rgbToHsv(npy::load<std::uint8_t>(a[0])), a[1]); }
CROSSVAL(hsl) { npy::save(rgbToHsl(npy::load<std::uint8_t>(a[0])), a[1]); }
CROSSVAL(lab) { npy::save(rgbToLab(npy::load<std::uint8_t>(a[0])), a[1]); }
CROSSVAL(saturation) { npy::save(adjustSaturation(npy::load<std::uint8_t>(a[0]), num(a, 2, 1)), a[1]); }

// ---- pixel ops ------------------------------------------------------------
CROSSVAL(autocontrast) { npy::save(autoContrast(npy::load<std::uint8_t>(a[0]), num(a, 2, 0)), a[1]); }
CROSSVAL(equalize) { npy::save(equalizeHist(npy::load<std::uint8_t>(a[0])), a[1]); }
CROSSVAL(otsu) {
    ImageI32 t(1, 1, 1, otsuThreshold(npy::load<std::uint8_t>(a[0])));
    npy::save(t, a[1]);
}
CROSSVAL(adaptive) {
    npy::save(adaptiveThreshold(npy::load<std::uint8_t>(a[0]), 255, num(a, 2, 0) ? AdaptiveMethod::Gaussian : AdaptiveMethod::Mean,
                                ThresholdType::Binary, static_cast<int>(num(a, 3, 11)), num(a, 4, 2)),
              a[1]);
}

// ---- filters (float in, float out unless noted) ---------------------------
// filter2d <in> <kernel.npy> <out> <border> [cval]
CROSSVAL(filter2d) {
    npy::save(filter2D(npy::load<float>(a[0]), npy::load<float>(a[1]), borderOf(a[3]), 0.0, Point(-1, -1), num(a, 4, 0)), a[2]);
}
// sep <in> <kx.npy> <ky.npy> <out> <border>
CROSSVAL(sep) { npy::save(sepFilter2D(npy::load<float>(a[0]), vecOf(a[1]), vecOf(a[2]), borderOf(a[4])), a[3]); }
// box <in> <out> <kw> <kh> <border>
CROSSVAL(box) {
    npy::save(boxFilter(npy::load<float>(a[0]), static_cast<int>(num(a, 2, 3)), static_cast<int>(num(a, 3, 3)), true, borderOf(a[4])), a[1]);
}
// gauss <in> <out> <sigmaX> <sigmaY> <border>
CROSSVAL(gauss) { npy::save(gaussianBlur(npy::load<float>(a[0]), num(a, 2, 1), num(a, 3, 0), 0, borderOf(a[4])), a[1]); }
CROSSVAL(gauss_u8) { npy::save(gaussianBlur(npy::load<std::uint8_t>(a[0]), num(a, 2, 1)), a[1]); }
// median <in u8> <out> <ksize> <border>
CROSSVAL(median) { npy::save(medianBlur(npy::load<std::uint8_t>(a[0]), static_cast<int>(num(a, 2, 3)), borderOf(a[3])), a[1]); }
CROSSVAL(median_f) { npy::save(medianBlur(npy::load<float>(a[0]), static_cast<int>(num(a, 2, 3)), borderOf(a[3])), a[1]); }
// bilateral <in u8> <out> <d> <sigmaColor> <sigmaSpace>
CROSSVAL(bilateral) {
    npy::save(bilateralFilter(npy::load<std::uint8_t>(a[0]), static_cast<int>(num(a, 2, 5)), num(a, 3, 20), num(a, 4, 3)), a[1]);
}
// sobel <in> <out> <dx> <dy> <ksize> <border>
CROSSVAL(sobel) {
    npy::save(sobel(npy::load<float>(a[0]), static_cast<int>(num(a, 2, 1)), static_cast<int>(num(a, 3, 0)),
                    static_cast<int>(num(a, 4, 3)), 1.0, borderOf(a[5])), a[1]);
}
CROSSVAL(scharr) { npy::save(scharr(npy::load<float>(a[0]), static_cast<int>(num(a, 2, 1)), static_cast<int>(num(a, 3, 0))), a[1]); }
CROSSVAL(laplacian) { npy::save(laplacian(npy::load<float>(a[0]), static_cast<int>(num(a, 2, 1))), a[1]); }

// ---- geometry ---------------------------------------------------------------
// resize <in> <out> <w> <h> <interp>     (float or uint8 input -> same type)
CROSSVAL(resize) {
    const int w = static_cast<int>(num(a, 2, 1)), h = static_cast<int>(num(a, 3, 1));
    if (npy::loadRaw(a[0]).descr == "|u1") npy::save(resize(npy::load<std::uint8_t>(a[0]), w, h, interpOf(a[4])), a[1]);
    else npy::save(resize(npy::load<float>(a[0]), w, h, interpOf(a[4])), a[1]);
}
// warpaffine <in> <out> <m.npy (6)> <w> <h> <interp> <border> <cval>
CROSSVAL(warpaffine) {
    const std::vector<double> m = vecOf(a[2]);
    npy::save(warpAffine(npy::load<float>(a[0]), AffineMatrix{m[0], m[1], m[2], m[3], m[4], m[5]},
                         Size(static_cast<int>(num(a, 3, 0)), static_cast<int>(num(a, 4, 0))), interpOf(a[5]),
                         borderOf(a[6]), num(a, 7, 0)), a[1]);
}
// warpperspective <in> <out> <h.npy (9)> <w> <h> <interp> <border> <cval>
CROSSVAL(warpperspective) {
    const std::vector<double> m = vecOf(a[2]);
    Homography hm;
    std::copy(m.begin(), m.begin() + 9, hm.begin());
    npy::save(warpPerspective(npy::load<float>(a[0]), hm, Size(static_cast<int>(num(a, 3, 0)), static_cast<int>(num(a, 4, 0))),
                              interpOf(a[5]), borderOf(a[6]), num(a, 7, 0)), a[1]);
}
// remap <in> <out> <mapx.npy> <mapy.npy> <interp> <border>
CROSSVAL(remap) {
    npy::save(remap(npy::load<float>(a[0]), npy::load<float>(a[2]), npy::load<float>(a[3]), interpOf(a[4]), borderOf(a[5])), a[1]);
}
CROSSVAL(pyrdown) { npy::save(pyrDown(npy::load<float>(a[0])), a[1]); }

// ---- morphology / analysis / features / metrics ---------------------------
MorphShape shapeOf(const std::string& s) {
    return s == "cross" ? MorphShape::Cross : s == "ellipse" ? MorphShape::Ellipse : MorphShape::Rect;
}
// erode|dilate <in u8> <out> <shape> <w> <h>
CROSSVAL(erode) {
    npy::save(erode(npy::load<std::uint8_t>(a[0]), getStructuringElement(shapeOf(a[2]), static_cast<int>(num(a, 3, 3)),
                                                                         static_cast<int>(num(a, 4, 3)))), a[1]);
}
CROSSVAL(dilate) {
    npy::save(dilate(npy::load<std::uint8_t>(a[0]), getStructuringElement(shapeOf(a[2]), static_cast<int>(num(a, 3, 3)),
                                                                          static_cast<int>(num(a, 4, 3)))), a[1]);
}
CROSSVAL(fillholes) { npy::save(fillHoles(npy::load<std::uint8_t>(a[0])), a[1]); }
CROSSVAL(label) { npy::save(connectedComponents(npy::load<std::uint8_t>(a[0]), static_cast<int>(num(a, 2, 8))).labels, a[1]); }
// dist <in u8> <out> <l1|l2|chess>
CROSSVAL(dist) {
    const DistanceType t = a[2] == "l1" ? DistanceType::L1 : a[2] == "chess" ? DistanceType::Chessboard : DistanceType::L2;
    npy::save(distanceTransform(npy::load<std::uint8_t>(a[0]), t), a[1]);
}
// moments <in float> <out>: [m00 m10 m01 m20 m11 m02 m30 m21 m12 m03 mu20 mu11 mu02 mu30 mu21 mu12 mu03 hu0..hu6]
CROSSVAL(moments) {
    const Moments m = moments(npy::load<float>(a[0]));
    const auto hu = m.hu();
    std::vector<double> v = {m.m00, m.m10, m.m01, m.m20, m.m11, m.m02, m.m30, m.m21, m.m12, m.m03,
                             m.mu20, m.mu11, m.mu02, m.mu30, m.mu21, m.mu12, m.mu03};
    v.insert(v.end(), hu.begin(), hu.end());
    npy::save(ImageD(static_cast<int>(v.size()), 1, 1, v), a[1]);
}
// match <img float> <out> <templ float> <method 0..5>
CROSSVAL(match) {
    npy::save(matchTemplate(npy::load<float>(a[0]), npy::load<float>(a[2]), static_cast<MatchMethod>(static_cast<int>(num(a, 3, 0)))), a[1]);
}
CROSSVAL(lbp) { npy::save(localBinaryPattern(npy::load<std::uint8_t>(a[0])), a[1]); }
// ssim <a u8> <out> <b u8>
CROSSVAL(ssim) {
    npy::save(ImageD(1, 1, 1, ssim(npy::load<std::uint8_t>(a[0]), npy::load<std::uint8_t>(a[2]))), a[1]);
}

// ---- fft / colormaps ---------------------------------------------------------
CROSSVAL(dft) { npy::save(dft(npy::load<double>(a[0])), a[1]); }
// colormap <ignored> <out> <index>: 256 x 1 RGB ramp of the colormap
CROSSVAL(colormap) {
    ImageF ramp(256, 1, 1);
    for (int i = 0; i < 256; ++i) ramp(i, 0) = static_cast<float>(i);
    npy::save(applyColormap(ramp, static_cast<Colormap>(static_cast<int>(num(a, 2, 3))), 0, 255), a[1]);
}

// decoderaw <file.jpg> <out.npy>: native-precision samples (uint16) of a JPEG
CROSSVAL(decoderaw) {
    int precision = 8;
    npy::save(decodeJPEGRaw(readFile(a[0]), precision), a[1]);
}

// ---- metadata / icc / apng ------------------------------------------------
// iccconvert <in.npy> <out.npy> <src.icc|srgb|p3|adobe> <dst.icc|srgb|p3|adobe> <intent>
IccProfile profileOf(const std::string& s) {
    if (s == "srgb") return IccProfile::sRGB();
    if (s == "p3") return IccProfile::displayP3();
    if (s == "adobe") return IccProfile::adobeRGB();
    return IccProfile::fromBytes(readFile(s));
}
CROSSVAL(iccconvert) {
    npy::save(convertColorProfile(npy::load<std::uint8_t>(a[0]), profileOf(a[2]), profileOf(a[3]),
                                  static_cast<RenderingIntent>(static_cast<int>(num(a, 4, 0)))), a[1]);
}
CROSSVAL(iccconvert16) {
    npy::save(convertColorProfile(npy::load<std::uint16_t>(a[0]), profileOf(a[2]), profileOf(a[3]),
                                  static_cast<RenderingIntent>(static_cast<int>(num(a, 4, 0)))), a[1]);
}
// load <file> <out.npy> <srgb 0/1> <orient 0/1>
CROSSVAL(load) {
    LoadOptions o;
    o.convertToSRGB = num(a, 2, 0) != 0;
    o.applyOrientation = num(a, 3, 1) != 0;
    npy::save(loadImage(a[0], o), a[1]);
}
// metadump <file> <out.txt>: metadata decoded by ipf as text
CROSSVAL(metadump) {
    ImageMetadata m;
    LoadOptions o;
    o.applyOrientation = false;
    loadImage(a[0], o, &m);
    std::string t = "dpi " + std::to_string(m.dpiX) + " " + std::to_string(m.dpiY) + "\n";
    t += "gamma " + std::to_string(m.gamma) + "\norientation " + std::to_string(m.orientation) + "\n";
    t += "icc " + std::to_string(m.iccProfile.size()) + " " + std::to_string(crc32(m.iccProfile.data(), m.iccProfile.size())) + "\n";
    t += "exif " + std::to_string(m.exif.size()) + "\n";
    for (const auto& kv : m.text) t += "text " + kv.first + "=" + kv.second + "\n";
    writeFile(a[1], std::vector<std::uint8_t>(t.begin(), t.end()));
}
// savemeta <in.npy> <out.png|jpg> <icc file or '-'>: writes fixed metadata
CROSSVAL(savemeta) {
    ImageMetadata m;
    m.dpiX = 300;
    m.dpiY = 300;
    m.text = {{"Title", "ipf"}, {"Description", "caf\xC3\xA9 \xE2\x9C\x93"}, {"comment", "hello"}};
    if (a[2] != "-") m.iccProfile = readFile(a[2]);
    SaveOptions o;
    o.metadata = &m;
    saveImage(npy::load<std::uint8_t>(a[0]), a[1], o);
}
// apngdecode <file> <out.npy>: frames stacked vertically (n*H, W, 4); delays (ms) to <out>.txt
CROSSVAL(apngdecode) {
    int loops = 0;
    const auto frames = decodeAPNG(readFile(a[0]), &loops);
    const int W = frames[0].image.width(), H = frames[0].image.height();
    ImageU8 all(W, H * static_cast<int>(frames.size()), 4);
    std::string t = std::to_string(loops);
    for (std::size_t i = 0; i < frames.size(); ++i) {
        all.paste(frames[i].image, 0, H * static_cast<int>(i));
        t += " " + std::to_string(std::lround(frames[i].delay * 1000));
    }
    npy::save(all, a[1]);
    writeFile(a[1] + ".txt", std::vector<std::uint8_t>(t.begin(), t.end()));
}
// apngencode <in.npy (n*H, W, c)> <out.png> <n> <loops>
CROSSVAL(apngencode) {
    const ImageU8 all = npy::load<std::uint8_t>(a[0]);
    const int n = static_cast<int>(num(a, 2, 1)), H = all.height() / n;
    std::vector<AnimationFrame> frames;
    for (int i = 0; i < n; ++i) frames.push_back({all.crop(Rect(0, i * H, all.width(), H)), 0.04 * (i + 1)});
    writeFile(a[1], encodeAPNG(frames, static_cast<int>(num(a, 3, 0))));
}
// pngopt <in.npy> <out.png> <interlace 0/1>
CROSSVAL(pngopt) {
    PngOptions o;
    o.interlace = num(a, 2, 0) != 0;
    writeFile(a[1], encodePNG(npy::load<std::uint8_t>(a[0]), o));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: ipf_crossval <command> args...\ncommands:");
        for (const auto& kv : commands()) std::printf(" %s", kv.first.c_str());
        std::printf("\n");
        return 2;
    }
    const auto it = commands().find(argv[1]);
    if (it == commands().end()) {
        std::fprintf(stderr, "unknown command %s\n", argv[1]);
        return 2;
    }
    try {
        it->second(Args(argv + 2, argv + argc));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
