#include <ipf/analysis.hpp>
#include <ipf/draw.hpp>
#include <ipf/geometry.hpp>

#include "test_framework.hpp"
#include "test_util.hpp"

using namespace ipf;
using namespace ipftest;

namespace {
ImageU8 rectMask(int w, int h, Rect r) {
    ImageU8 m(w, h, 1, 0);
    drawRectangle(m, r, Color(255), -1);
    return m;
}
}  // namespace

TEST(analysis_connected_components) {
    ImageU8 m(10, 6, 1, 0);
    drawRectangle(m, Rect(0, 0, 3, 2), Color(255), -1);  // blob 1: 6 px
    drawRectangle(m, Rect(6, 3, 4, 3), Color(255), -1);  // blob 2: 12 px
    m(3, 2) = 255;                                         // touches blob 1 only diagonally
    const Components c4 = connectedComponents(m, 4), c8 = connectedComponents(m, 8);
    CHECK_EQ(c4.count, 3);
    CHECK_EQ(c8.count, 2);
    CHECK_EQ(c8.stats[0].area, 7);
    CHECK(c8.stats[1].bbox == Rect(6, 3, 4, 3));
    CHECK_NEAR(c8.stats[1].centroid.x, 7.5, 1e-12);
    CHECK_NEAR(c8.stats[1].centroid.y, 4.0, 1e-12);
    CHECK_EQ(c8.labels(9, 5), 2);
    CHECK_EQ(c8.labels(5, 0), 0);
    CHECK_EQ(countNonZero(removeSmallObjects(m, 10)), 12u);
    CHECK_EQ(countNonZero(largestComponent(m)), 12u);
    // U shape: labels merge correctly through union-find
    ImageU8 u(5, 4, 1, 0);
    for (int y = 0; y < 4; ++y) u(0, y) = u(4, y) = 255;
    for (int x = 0; x < 5; ++x) u(x, 3) = 255;
    CHECK_EQ(connectedComponents(u, 4).count, 1);
}

TEST(analysis_contours) {
    const ImageU8 r = rectMask(12, 10, Rect(2, 3, 5, 4));
    auto cs = findContours(r);
    CHECK_EQ(cs.size(), 1u);
    CHECK_EQ(cs[0].points.size(), 14u);  // 2 * (5 + 4) - 4 boundary pixels
    CHECK_NEAR(contourAreaSigned(cs[0].points), 12.0, 1e-12);  // outer borders: counter-clockwise
    CHECK(boundingRect(cs[0].points) == Rect(2, 3, 5, 4));
    CHECK_EQ(cs[0].parent, -1);
    CHECK(!cs[0].isHole);
    CHECK(cs[0].points[0] == Point(2, 3));
    // ring + nested island: outer, hole (child of outer), island (child of hole)
    ImageU8 ring = rectMask(20, 20, Rect(2, 2, 16, 16));
    drawRectangle(ring, Rect(5, 5, 10, 10), Color(0), -1);
    drawRectangle(ring, Rect(8, 8, 4, 4), Color(255), -1);
    cs = findContours(ring);
    CHECK_EQ(cs.size(), 3u);
    CHECK(!cs[0].isHole);
    CHECK(cs[1].isHole);
    CHECK_EQ(cs[1].parent, 0);
    CHECK(!cs[2].isHole);
    CHECK_EQ(cs[2].parent, 1);
    CHECK(contourAreaSigned(cs[1].points) < 0);  // holes are traced clockwise
    // single pixel and a thin line
    ImageU8 dot(5, 5, 1, 0);
    dot(2, 2) = 1;
    cs = findContours(dot);
    CHECK_EQ(cs.size(), 1u);
    CHECK_EQ(cs[0].points.size(), 1u);
    ImageU8 line(8, 3, 1, 0);
    for (int x = 1; x < 7; ++x) line(x, 1) = 1;
    cs = findContours(line);
    CHECK_EQ(cs.size(), 1u);
    CHECK_EQ(cs[0].points.size(), 10u);  // traced out and back
    // two separate blobs at image borders
    ImageU8 two(6, 4, 1, 0);
    two(0, 0) = two(1, 0) = two(5, 3) = 255;
    CHECK_EQ(findContours(two).size(), 2u);
}

TEST(analysis_shape_descriptors) {
    std::vector<Point> pts = {{0, 0}, {10, 0}, {10, 5}, {0, 5}, {5, 2}, {3, 4}};
    const auto hull = convexHull(pts);
    CHECK_EQ(hull.size(), 4u);
    CHECK_NEAR(contourArea(hull), 50.0, 1e-12);
    CHECK_NEAR(arcLength(hull), 30.0, 1e-12);
    CHECK(pointInPolygon(hull, PointF(5, 2.5)));
    CHECK(!pointInPolygon(hull, PointF(11, 2.5)));
    const auto cs = findContours(rectMask(30, 20, Rect(3, 4, 20, 10)));
    const auto approx = approxPolyDP(cs[0].points, 1.0);
    CHECK_EQ(approx.size(), 4u);
    const Circle c = minEnclosingCircle({{0, 0}, {10, 0}, {5, 3}, {5, -3}});
    CHECK_NEAR(c.center.x, 5, 1e-9);
    CHECK_NEAR(c.radius, 5, 1e-9);
    const Circle c3 = minEnclosingCircle({{0, 0}, {6, 0}, {3, 6}});
    CHECK_NEAR(c3.center.x, 3, 1e-9);
    CHECK_NEAR(c3.center.y, 2.25, 1e-9);
    CHECK_NEAR(c3.radius, 3.75, 1e-9);
}

TEST(analysis_moments) {
    const ImageU8 r = rectMask(40, 30, Rect(5, 10, 20, 6));
    const Moments m = moments(r, true);
    CHECK_NEAR(m.m00, 120, 1e-9);
    CHECK_NEAR(m.centroid().x, 14.5, 1e-9);
    CHECK_NEAR(m.centroid().y, 12.5, 1e-9);
    CHECK_NEAR(m.mu20, 6 * (20.0 * 20 * 20 - 20) / 12.0, 1e-6);  // sum of (x - cx)^2
    CHECK_NEAR(m.mu11, 0, 1e-9);
    CHECK_NEAR(m.orientation(), 0, 1e-9);
    // Hu invariants are rotation invariant
    ImageU8 shape(40, 40, 1, 0);
    fillPolygon(shape, std::vector<Point>{{5, 5}, {30, 8}, {20, 30}, {8, 20}}, Color(255));
    const auto h1 = moments(shape, true).hu(), h2 = moments(rotate90(shape, 1), true).hu();
    for (int i = 0; i < 7; ++i) CHECK_NEAR(h1[i], h2[i], 1e-9 + 1e-9 * std::fabs(h1[i]));
    // a 45-degree diagonal bar has orientation 45 (counter-clockwise as displayed)
    ImageU8 diag(30, 30, 1, 0);
    drawLine(diag, PointF(5, 25), PointF(25, 5), Color(255), 3);
    CHECK_NEAR(moments(diag, true).orientation(), 45, 1.0);
}

TEST(analysis_distance_transform) {
    ImageU8 m(9, 7, 1, 255);
    m(4, 3) = 0;
    const ImageF l2 = distanceTransform(m, DistanceType::L2);
    CHECK_NEAR(l2(0, 0), std::sqrt(16.0 + 9.0), 1e-5);
    CHECK_NEAR(l2(4, 3), 0, 0);
    CHECK_NEAR(distanceTransform(m, DistanceType::L1)(0, 0), 7, 0);
    CHECK_NEAR(distanceTransform(m, DistanceType::Chessboard)(0, 0), 4, 0);
    // random masks vs brute force
    const ImageU8 rnd = threshold(randomImage<std::uint8_t>(23, 17, 1, 9), 200, 255, ThresholdType::BinaryInv);
    const ImageF d = distanceTransform(rnd);
    double maxErr = 0;
    for (int y = 0; y < 17; ++y)
        for (int x = 0; x < 23; ++x) {
            double best = 1e9;
            for (int yy = 0; yy < 17; ++yy)
                for (int xx = 0; xx < 23; ++xx)
                    if (!rnd(xx, yy)) best = std::min(best, std::hypot(x - xx, y - yy));
            maxErr = std::max(maxErr, std::fabs(best - d(x, y)));
        }
    CHECK(maxErr < 1e-4);
    const ImageF allFg = distanceTransform(ImageU8(4, 4, 1, 1));
    CHECK(std::isinf(allFg(1, 1)));
}

TEST(analysis_flood_fill) {
    ImageU8 img(10, 10, 3, 0);
    drawRectangle(img, Rect(2, 2, 6, 6), Color(200, 0, 0), 1);
    ImageU8 mask;
    const std::size_t n = floodFill(img, Point(4, 4), {0, 255, 0}, 0.0, 4, &mask);
    CHECK_EQ(n, 16u);  // interior of the 6x6 outline
    CHECK_EQ(img(4, 4, 1), 255);
    CHECK_EQ(img(0, 0, 1), 0);
    CHECK_EQ(countNonZero(mask), 16u);
    const std::size_t outside = floodFill(img, Point(0, 0), {9, 9, 9}, 0.0, 4);
    CHECK_EQ(outside, 100u - 36u);
    CHECK_THROWS(floodFill(img, Point(20, 0), {1, 1, 1}));
}
