// Shared helper for the examples: renders a synthetic test scene so the examples run
// without any input files.
#pragma once

#include <ipf/ipf.hpp>

#include <random>

inline ipf::ImageU8 makeScene(int width = 320, int height = 240) {
    using namespace ipf;
    ImageU8 img(width, height, 3);
    for (int y = 0; y < height; ++y)  // sky-like vertical gradient
        for (int x = 0; x < width; ++x) {
            img(x, y, 0) = saturate_cast<std::uint8_t>(70 + 0.25 * y);
            img(x, y, 1) = saturate_cast<std::uint8_t>(110 + 0.30 * y);
            img(x, y, 2) = saturate_cast<std::uint8_t>(190 - 0.20 * y + 0.1 * x);
        }
    drawRectangle(img, Rect(20, 150, 110, 70), Color(200, 60, 40), -1);
    drawRectangle(img, Rect(20, 150, 110, 70), Color(90, 20, 10), 3);
    drawCircle(img, PointF(230, 80), 42, Color(250, 210, 40), -1, true);
    drawCircle(img, PointF(150, 70), 22, Color(40, 170, 80), -1, true);
    fillPolygon(img, std::vector<Point>{{170, 225}, {300, 225}, {250, 140}}, Color(120, 60, 160));
    drawLine(img, PointF(10, 120), PointF(310, 105), Color(30, 30, 30), 3, true);
    drawText(img, "ipf 1.0", Point(18, 18), Color(255, 255, 255), 3);
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 6.0);
    for (auto& v : img) v = saturate_cast<std::uint8_t>(v + noise(rng));
    return img;
}
