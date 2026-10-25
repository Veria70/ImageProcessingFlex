// Object counting: segment bright objects, clean the mask with morphology, label the
// connected components and report their measurements.
//
//   count_objects [input-image] [output.png]
#include <ipf/ipf.hpp>

#include <cstdio>
#include <random>
#include <string>

namespace {
// Synthetic "cells under a microscope" image used when no input is given.
ipf::ImageU8 makeCells() {
    ipf::ImageU8 img(400, 300, 1, 40);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> px(30, 370), py(30, 270), pr(8, 22);
    for (int i = 0; i < 25; ++i)
        ipf::drawCircle(img, ipf::PointF(px(rng), py(rng)), pr(rng), ipf::Color(190), -1, true);
    img = ipf::gaussianBlur(img, 2.0);
    std::normal_distribution<double> noise(0, 12);
    for (auto& v : img) v = ipf::saturate_cast<std::uint8_t>(v + noise(rng));
    return img;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const ipf::ImageU8 gray = argc > 1 && argv[1][0] ? ipf::loadImage(argv[1], 1) : makeCells();
        const std::string outPath = argc > 2 ? argv[2] : "objects.png";

        // Segment: denoise, Otsu threshold, open to remove specks, fill holes.
        const ipf::ImageU8 smooth = ipf::medianBlur(gray, 5);
        ipf::ImageU8 mask = ipf::thresholdOtsu(smooth);
        mask = ipf::morphologyEx(mask, ipf::MorphOp::Open, ipf::getStructuringElement(ipf::MorphShape::Ellipse, 5));
        mask = ipf::fillHoles(mask);
        mask = ipf::removeSmallObjects(mask, 50);

        // Measure each object.
        const ipf::Components cc = ipf::connectedComponents(mask);
        ipf::ImageU8 vis = ipf::convertChannels(gray, 3);
        std::printf("found %d objects\n  id   area   centroid      bbox\n", cc.count);
        for (const auto& s : cc.stats) {
            std::printf("  %2d  %5lld  (%5.1f,%5.1f)  %dx%d@(%d,%d)\n", s.label, s.area, s.centroid.x, s.centroid.y,
                        s.bbox.width, s.bbox.height, s.bbox.x, s.bbox.y);
            ipf::drawRectangle(vis, s.bbox, ipf::Color(0, 255, 0), 1);
            ipf::drawText(vis, std::to_string(s.label), ipf::Point(s.bbox.x, s.bbox.y - 9), ipf::Color(255, 255, 0));
        }
        for (const ipf::Contour& c : ipf::findContours(mask))
            if (!c.isHole) ipf::drawPolyline(vis, c.points, true, ipf::Color(255, 0, 0));
        ipf::saveImage(vis, outPath);
        std::printf("wrote %s\n", outPath.c_str());
    } catch (const ipf::Error& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
