// Benchmark: times common operations on a Full-HD RGB image.
//
//   benchmark [width] [height]
#include <ipf/ipf.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

#include "scene.hpp"

using namespace ipf;

namespace {
double timeMs(const std::function<void()>& fn) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
// Times an operation with 1 thread and with all hardware threads.
void bench(const char* name, const std::function<void()>& fn) {
    setNumThreads(1);
    const double serial = timeMs(fn);
    setNumThreads(0);
    const double parallel = timeMs(fn);
    std::printf("  %-36s %9.1f ms %9.1f ms %6.1fx\n", name, serial, parallel, serial / parallel);
}
}  // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 1920, H = argc > 2 ? std::atoi(argv[2]) : 1080;
    const ImageU8 img = resize(makeScene(), W, H, Interpolation::Bicubic);
    const ImageU8 gray = rgbToGray(img);
    std::printf("image %dx%d RGB, %d hardware threads\n", W, H, getNumThreads());
    std::printf("  %-36s %12s %12s %7s\n", "operation", "1 thread", "all threads", "speedup");
    std::vector<std::uint8_t> png, jpg;
    bench("encode PNG", [&] { png = encodePNG(img); });
    bench("decode PNG", [&] { (void)decodePNG(png); });
    bench("encode JPEG q90", [&] { jpg = encodeJPEG(img, 90); });
    bench("decode JPEG", [&] { (void)decodeJPEG(jpg); });
    bench("rgbToGray", [&] { (void)rgbToGray(img); });
    bench("rgbToHsv", [&] { (void)rgbToHsv(img); });
    bench("gaussianBlur sigma 2 (RGB)", [&] { (void)gaussianBlur(img, 2.0); });
    bench("boxFilter 15x15 (RGB)", [&] { (void)blur(img, 15); });
    bench("medianBlur 5 (gray)", [&] { (void)medianBlur(gray, 5); });
    bench("medianBlur 9 (gray, histogram)", [&] { (void)medianBlur(gray, 9); });
    bench("bilateralFilter d=9 (gray)", [&] { (void)bilateralFilter(gray, 9, 30, 3); });
    bench("sobel dx (gray)", [&] { (void)sobel(gray, 1, 0); });
    bench("canny (gray)", [&] { (void)canny(gray, 50, 150); });
    bench("resize 1/2 lanczos (RGB)", [&] { (void)resize(img, W / 2, H / 2, Interpolation::Lanczos); });
    bench("resize 2x bilinear (RGB)", [&] { (void)resize(img, W * 2, H * 2, Interpolation::Bilinear); });
    bench("rotate 15 deg bilinear (RGB)", [&] { (void)rotate(img, 15.0); });
    bench("erode 5x5 rect (gray)", [&] { (void)erode(gray, getStructuringElement(MorphShape::Rect, 5)); });
    bench("dilate 5x5 ellipse (gray)", [&] { (void)dilate(gray, getStructuringElement(MorphShape::Ellipse, 5)); });
    const ImageU8 bin = thresholdOtsu(gray);
    bench("connectedComponents", [&] { (void)connectedComponents(bin); });
    bench("findContours", [&] { (void)findContours(bin); });
    bench("distanceTransform L2", [&] { (void)distanceTransform(bin); });
    bench("equalizeHist (gray)", [&] { (void)equalizeHist(gray); });
    bench("clahe (gray)", [&] { (void)clahe(gray); });
    bench("ssim (gray)", [&] { (void)ssim(gray, gray); });
    const ImageU8 templ = gray.crop(Rect(W / 3, H / 3, 128, 96));
    bench("matchTemplate 128x96 (FFT)", [&] { (void)matchTemplate(gray, templ, MatchMethod::CCoeffNormed); });
    bench("dft (gray)", [&] { (void)dft(gray); });
    bench("adobeRGB -> sRGB (ICC)", [&] { (void)convertColorProfile(img, IccProfile::adobeRGB()); });
    bench("houghLinesP on Canny edges", [&] { (void)houghLinesP(canny(gray, 50, 150), 1, 3.14159265 / 180, 80, 50, 5); });
    return 0;
}
