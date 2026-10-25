// Quick start: load an image (or render a test scene), process it, save the results.
//
//   quickstart [input-image] [output-directory]
#include <ipf/ipf.hpp>

#include <cstdio>
#include <string>

#include "scene.hpp"

int main(int argc, char** argv) {
    try {
        const std::string outDir = argc > 2 ? std::string(argv[2]) + "/" : "";
        // 1. Load any PNG/JPEG/BMP/TGA/PNM file as RGB (or use the built-in scene).
        ipf::ImageU8 img = argc > 1 && argv[1][0] ? ipf::loadImage(argv[1], 3) : makeScene();
        std::printf("image: %dx%d, %d channels\n", img.width(), img.height(), img.channels());

        // 2. Classic processing steps.
        const ipf::ImageU8 gray = ipf::rgbToGray(img);
        const ipf::ImageU8 smooth = ipf::gaussianBlur(gray, 1.5);
        const ipf::ImageU8 edges = ipf::canny(smooth, 40, 100);
        const ipf::ImageU8 half = ipf::resize(img, img.width() / 2, img.height() / 2, ipf::Interpolation::Lanczos);
        const ipf::ImageU8 mask = ipf::thresholdOtsu(smooth);

        // 3. Measurements.
        const ipf::Components cc = ipf::connectedComponents(mask);
        std::printf("Otsu threshold: %d, components: %d, sharpness: %.1f\n", ipf::otsuThreshold(smooth), cc.count,
                    ipf::sharpness(gray));

        // 4. Annotate and save (format chosen from the extension).
        ipf::ImageU8 annotated = img;
        for (const auto& s : cc.stats)
            if (s.area > 200) ipf::drawRectangle(annotated, s.bbox, ipf::Color(255, 0, 255), 2);
        ipf::saveImage(edges, outDir + "edges.png");
        ipf::saveImage(half, outDir + "half.jpg", 92);
        ipf::saveImage(annotated, outDir + "annotated.png");
        std::printf("wrote edges.png, half.jpg, annotated.png\n");
    } catch (const ipf::Error& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
