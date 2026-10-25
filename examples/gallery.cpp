// Gallery: runs most ipf algorithms on one image and writes a labelled contact sheet.
//
//   gallery [input-image] [output.png]
#include <ipf/ipf.hpp>

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "scene.hpp"

using namespace ipf;

namespace {

ImageU8 asRgb(const ImageU8& img) { return convertChannels(img, 3); }

ImageU8 labelColors(const Components& cc) {
    ImageU8 out(cc.labels.width(), cc.labels.height(), 3, 0);
    for (std::size_t i = 0; i < out.pixelCount(); ++i) {
        const int l = cc.labels.data()[i];
        if (!l) continue;
        const auto c = colormapColor(Colormap::Turbo, ((l * 37) % 100) / 100.0);
        for (int k = 0; k < 3; ++k) out.data()[i * 3 + k] = saturate_cast<std::uint8_t>(c[k] * 255);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const ImageU8 src = argc > 1 && argv[1][0] ? resizeToFit(loadImage(argv[1], 3), 320, 240) : makeScene();
        const std::string outPath = argc > 2 ? argv[2] : "gallery.png";
        const ImageU8 gray = rgbToGray(src);
        const ImageU8 smooth = gaussianBlur(gray, 1.2);
        const ImageU8 bin = thresholdOtsu(smooth);

        std::vector<std::pair<std::string, std::function<ImageU8()>>> tiles = {
            {"original", [&] { return src; }},
            {"grayscale", [&] { return asRgb(gray); }},
            {"gaussian blur s=3", [&] { return gaussianBlur(src, 3.0); }},
            {"median 7x7", [&] { return medianBlur(src, 7); }},
            {"bilateral", [&] { return bilateralFilter(src, 9, 40, 4); }},
            {"unsharp mask", [&] { return unsharpMask(src, 2.0, 1.5); }},
            {"sobel magnitude", [&] { return applyColormap(gradientMagnitude(smooth), Colormap::Inferno); }},
            {"canny", [&] { return asRgb(canny(smooth, 40, 100)); }},
            {"otsu threshold", [&] { return asRgb(bin); }},
            {"adaptive threshold", [&] {
                 return asRgb(adaptiveThreshold(gray, 255, AdaptiveMethod::Gaussian, ThresholdType::Binary, 15, 4));
             }},
            {"morph gradient", [&] { return asRgb(morphologyEx(gray, MorphOp::Gradient)); }},
            {"distance transform", [&] { return applyColormap(distanceTransform(bin), Colormap::Viridis); }},
            {"connected components", [&] { return labelColors(connectedComponents(bin)); }},
            {"contours + hull", [&] {
                 ImageU8 out = src;
                 for (const Contour& c : findContours(bin)) {
                     if (c.points.size() < 30) continue;
                     drawPolyline(out, c.points, true, Color(255, 0, 255), 1);
                     drawPolyline(out, convexHull(c.points), true, Color(0, 255, 255), 1);
                 }
                 return out;
             }},
            {"corners (shi-tomasi)", [&] {
                 ImageU8 out = src;
                 for (const Point& p : goodFeaturesToTrack(gray, 60, 0.05, 8))
                     drawMarker(out, p, Color(255, 0, 0), MarkerType::Cross, 9, 2);
                 return out;
             }},
            {"hough circles", [&] {
                 ImageU8 out = src;
                 for (const HoughCircle& c : houghCircles(smooth, 15, 60, 100, 20, 20))
                     drawCircle(out, c.center, c.radius, Color(255, 0, 255), 2, true);
                 return out;
             }},
            {"rotate 30 (expand)", [&] { return rotate(src, 30.0, Interpolation::Bicubic, true); }},
            {"perspective warp", [&] {
                 const int W = src.width(), H = src.height();
                 const Homography h = getPerspectiveTransform(
                     {PointF(0, 0), PointF(W - 1, 0), PointF(W - 1, H - 1), PointF(0, H - 1)},
                     {PointF(40, 20), PointF(W - 30, 0), PointF(W - 1, H - 1), PointF(0, H - 40)});
                 return warpPerspective(src, h, Size(), Interpolation::Bilinear, BorderMode::Constant, 0);
             }},
            {"lanczos 1/4 -> 4x", [&] {
                 return resize(resize(src, src.width() / 4, src.height() / 4, Interpolation::Lanczos), src.width(),
                               src.height(), Interpolation::Nearest);
             }},
            {"hue shift +120", [&] { return shiftHue(src, 120); }},
            {"sepia", [&] { return sepia(src); }},
            {"CLAHE (luma)", [&] {
                 ImageU8 ycc = rgbToYCbCr(src);
                 insertChannel(ycc, clahe(extractChannel(ycc, 0), 3.0), 0);
                 return yCbCrToRgb(ycc);
             }},
            {"FFT magnitude spectrum", [&] { return applyColormap(magnitudeSpectrum(gray), Colormap::Magma); }},
            {"hough line segments", [&] {
                 ImageU8 out = src;
                 for (const LineSegment& l : houghLinesP(canny(smooth, 40, 100), 1, 3.14159265 / 180, 30, 25, 4))
                     drawLine(out, l.p0, l.p1, Color(255, 0, 255), 2);
                 return out;
             }},
            {"Display P3 -> sRGB", [&] { return convertColorProfile(src, IccProfile::displayP3()); }},
            {"emboss", [&] { return asRgb(emboss(gray)); }},
            {"LBP texture", [&] { return asRgb(localBinaryPattern(gray)); }},
        };

        const int tw = 320, th = 240, cols = 4, pad = 6, labelH = 20;
        const int rows = static_cast<int>((tiles.size() + cols - 1) / cols);
        ImageU8 sheet(cols * (tw + pad) + pad, rows * (th + labelH + pad) + pad, 3, 32);
        for (std::size_t i = 0; i < tiles.size(); ++i) {
            const int cx = pad + static_cast<int>(i % cols) * (tw + pad);
            const int cy = pad + static_cast<int>(i / cols) * (th + labelH + pad);
            ImageU8 tile = tiles[i].second();
            if (tile.width() != tw || tile.height() != th) {
                ImageU8 fitted = resizeToFit(tile, tw, th);
                tile = ImageU8(tw, th, 3, 0);
                tile.paste(fitted, (tw - fitted.width()) / 2, (th - fitted.height()) / 2);
            }
            drawText(sheet, tiles[i].first, Point(cx + 2, cy + 4), Color(235, 235, 235), 2);
            sheet.paste(tile, cx, cy + labelH);
            std::printf("  %-22s done\n", tiles[i].first.c_str());
        }
        saveImage(sheet, outPath);
        std::printf("wrote %s (%dx%d)\n", outPath.c_str(), sheet.width(), sheet.height());
    } catch (const Error& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
