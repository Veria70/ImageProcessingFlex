#include <ipf/ipf.hpp>

#include <atomic>
#include <functional>
#include <stdexcept>

#include "test_framework.hpp"
#include "test_util.hpp"

using namespace ipf;
using namespace ipftest;

namespace {
template <class Fn>
auto withThreads(int n, Fn fn) {
    setNumThreads(n);
    auto r = fn();
    setNumThreads(0);
    return r;
}
}  // namespace

TEST(parallel_for_basics) {
    setNumThreads(4);
    std::vector<int> hits(100000, 0);
    parallelFor(0, 100000, [&](int a, int b) {
        for (int i = a; i < b; ++i) ++hits[i];
    }, 10.0);
    CHECK(std::all_of(hits.begin(), hits.end(), [](int v) { return v == 1; }));
    // exceptions propagate to the caller
    CHECK_THROWS(parallelFor(0, 100000, [](int a, int) {
        if (a > 0) throw std::runtime_error("boom");
    }, 10.0));
    // nested regions run serially inside the worker (no oversubscription, still correct)
    std::atomic<long long> total{0};
    parallelFor(0, 64, [&](int a, int b) {
        for (int i = a; i < b; ++i)
            parallelFor(0, 1000, [&](int c, int d) { total += d - c; }, 1e6);
    }, 1e6);
    CHECK_EQ(total.load(), 64000LL);
    std::size_t covered = 0;
    std::mutex m;
    parallelForN(1000003, [&](std::size_t a, std::size_t b) {
        std::lock_guard<std::mutex> lock(m);
        covered += b - a;
    });
    CHECK_EQ(covered, 1000003u);
    setNumThreads(0);
    CHECK(getNumThreads() >= 1);
}

TEST(parallel_results_independent_of_thread_count) {
    const ImageU8 img = smoothImage(301, 211, 3);
    const ImageU8 noisy = add(img, randomImage<std::uint8_t>(301, 211, 3, 3));
    const ImageU8 gray = rgbToGray(noisy);
    const ImageU8 bin = thresholdOtsu(gray);
    std::vector<std::pair<const char*, std::function<ImageF()>>> ops = {
        {"gaussian", [&] { return gaussianBlur<float>(noisy, 2.5); }},
        {"box", [&] { return boxFilter<float>(noisy, 7, 5); }},
        {"filter2D", [&] { return filter2D<float>(noisy, gaussianKernel2D(1.2)); }},
        {"median5", [&] { return medianBlur(noisy, 5).convertTo<float>(); }},
        {"median3", [&] { return medianBlur(noisy, 3).convertTo<float>(); }},
        {"bilateral", [&] { return bilateralFilter(noisy, 7, 30, 3).convertTo<float>(); }},
        {"sobel", [&] { return sobel(gray, 1, 1, 5); }},
        {"resize", [&] { return resize(noisy, 457, 123, Interpolation::Lanczos).convertTo<float>(); }},
        {"rotate", [&] { return rotate(noisy, 17.0, Interpolation::Bicubic, true).convertTo<float>(); }},
        {"perspective", [&] {
             return warpPerspective(noisy, Homography{1, 0.1, 3, 0.05, 1, 2, 0.0005, 0.0002, 1}).convertTo<float>();
         }},
        {"erode", [&] { return erode(noisy, getStructuringElement(MorphShape::Ellipse, 7)).convertTo<float>(); }},
        {"dilate", [&] { return dilate(noisy, getStructuringElement(MorphShape::Rect, 5)).convertTo<float>(); }},
        {"hsv", [&] { return rgbToHsv(noisy); }},
        {"lab", [&] { return rgbToLab(noisy); }},
        {"edt", [&] { return distanceTransform(bin); }},
        {"canny", [&] { return canny(gray, 30, 90).convertTo<float>(); }},
        {"blend", [&] { return addWeighted(noisy, 0.3, img, 0.7).convertTo<float>(); }},
        {"match", [&] { return matchTemplate(gray, gray.crop(Rect(40, 30, 21, 17)), MatchMethod::CCoeffNormed); }},
    };
    detail::threadSetting() = 0;
    for (auto& op : ops) {
        const ImageF one = withThreads(1, op.second);
        const ImageF many = withThreads(8, op.second);
        if (!(one == many)) std::printf("    mismatch in %s\n", op.first);
        CHECK(one == many);
    }
}
