#include <ipf/draw.hpp>
#include <ipf/features.hpp>

#include "test_framework.hpp"
#include "test_util.hpp"

using namespace ipf;
using namespace ipftest;

TEST(features_canny) {
    ImageU8 img(40, 40, 1, 20);
    drawRectangle(img, Rect(10, 10, 20, 20), Color(220), -1);
    const ImageU8 e = canny(img, 50, 150);
    CHECK(countNonZero(e) > 60u);
    CHECK_EQ(e(20, 20), 0);  // interior
    CHECK_EQ(e(2, 2), 0);    // background
    int nearEdge = 0;
    for (int y = 0; y < 40; ++y)
        for (int x = 0; x < 40; ++x)
            if (e(x, y)) {
                const bool close = (std::abs(x - 10) <= 1 || std::abs(x - 29) <= 1 || std::abs(y - 10) <= 1 ||
                                    std::abs(y - 29) <= 1);
                nearEdge += close ? 1 : 0;
                CHECK(close);
            }
    CHECK(nearEdge > 0);
    // edges are thin: no 2x2 block of edge pixels along the straight sides
    for (int y = 12; y < 27; ++y) CHECK(!(e(9, y) && e(10, y) && e(9, y + 1) && e(10, y + 1)));
    CHECK_EQ(countNonZero(canny(ImageU8(20, 20, 1, 100), 10, 20)), 0u);
    const ImageU8 rgb = convertChannels(img, 3);
    CHECK(canny(rgb, 50, 150) == e);
    CHECK(countNonZero(cannyAuto(img)) > 40u);
}

TEST(features_corners) {
    ImageU8 img(60, 60, 1, 0);
    drawRectangle(img, Rect(15, 20, 30, 25), Color(255), -1);
    const auto pts = goodFeaturesToTrack(img, 10, 0.1, 5.0);
    CHECK_EQ(pts.size(), 4u);
    const Point corners[4] = {{15, 20}, {44, 20}, {15, 44}, {44, 44}};
    for (const Point& c : corners) {
        bool found = false;
        for (const Point& p : pts) found |= std::abs(p.x - c.x) <= 2 && std::abs(p.y - c.y) <= 2;
        CHECK(found);
    }
    const auto harris = goodFeaturesToTrack(img, 10, 0.1, 5.0, 3, true);
    CHECK_EQ(harris.size(), 4u);
    const ImageF r = cornerHarris(img);
    CHECK(r(15, 20) > 0);
    CHECK(r(30, 20) < 0);  // edges have negative Harris response
    CHECK_NEAR(r(5, 5), 0, 1e-9);
}

TEST(features_hough) {
    ImageU8 e(80, 60, 1, 0);
    drawLine(e, PointF(0, 10), PointF(79, 10), Color(255));    // horizontal: theta 90 deg, rho 10
    drawLine(e, PointF(30, 0), PointF(30, 59), Color(255));    // vertical: theta 0, rho 30
    const auto lines = houghLines(e, 1.0, detail::kPi / 180, 40);
    CHECK(lines.size() >= 2u);
    bool horiz = false, vert = false;
    for (const auto& l : lines) {
        if (std::fabs(l.theta - detail::kPi / 2) < 0.02 && std::fabs(l.rho - 10) <= 1) horiz = true;
        if (std::fabs(l.theta) < 0.02 && std::fabs(l.rho - 30) <= 1) vert = true;
    }
    CHECK(horiz && vert);
    ImageU8 img(100, 100, 1, 30);
    drawCircle(img, PointF(48, 52), 20, Color(220), -1);
    drawCircle(img, PointF(15, 15), 8, Color(200), -1);
    const auto circles = houghCircles(img, 5, 30, 100, 15, 10);
    CHECK(circles.size() >= 2u);
    bool big = false, small = false;
    for (const auto& c : circles) {
        if (std::hypot(c.center.x - 48, c.center.y - 52) <= 2 && std::fabs(c.radius - 20) <= 2) big = true;
        if (std::hypot(c.center.x - 15, c.center.y - 15) <= 2 && std::fabs(c.radius - 8) <= 2) small = true;
    }
    CHECK(big && small);
    // straight edges, corners and text must not produce circles
    ImageU8 noCircles(120, 90, 1, 40);
    drawRectangle(noCircles, Rect(10, 10, 50, 30), Color(220), -1);
    fillPolygon(noCircles, std::vector<Point>{{70, 80}, {115, 80}, {95, 40}}, Color(180));
    drawLine(noCircles, PointF(0, 60), PointF(119, 50), Color(250), 2);
    drawText(noCircles, "abc 123", Point(10, 65), Color(255), 2);
    CHECK_EQ(houghCircles(noCircles, 5, 40, 100, 15, 10).size(), 0u);
}

TEST(features_template_matching) {
    const ImageU8 img = randomImage<std::uint8_t>(50, 40, 3, 21);
    const ImageU8 templ = img.crop(Rect(17, 11, 9, 7));
    for (MatchMethod m : {MatchMethod::SqDiff, MatchMethod::SqDiffNormed, MatchMethod::CCorrNormed,
                          MatchMethod::CCoeff, MatchMethod::CCoeffNormed}) {
        const MatchResult r = findTemplate(img, templ, m);
        CHECK(r.location == Point(17, 11));
    }
    const ImageF res = matchTemplate(img, templ, MatchMethod::CCoeffNormed);
    CHECK_EQ(res.width(), 42);
    CHECK_EQ(res.height(), 34);
    CHECK_NEAR(res(17, 11), 1.0, 1e-5);
    CHECK_NEAR(matchTemplate(img, templ, MatchMethod::SqDiff)(17, 11), 0.0, 1e-3);
    CHECK_THROWS(matchTemplate(templ, img, MatchMethod::CCorr));
}

TEST(features_lbp_and_nms) {
    const ImageU8 flat(5, 5, 1, 9);
    CHECK_EQ(localBinaryPattern(flat)(2, 2), 255);
    ImageU8 spot(3, 3, 1, 0);
    spot(1, 1) = 50;
    CHECK_EQ(localBinaryPattern(spot)(1, 1), 0);
    spot(2, 1) = 60;  // right neighbour is bit 4
    CHECK_EQ(localBinaryPattern(spot)(1, 1), 1 << 4);
    std::vector<Rect> boxes = {Rect(0, 0, 10, 10), Rect(1, 1, 10, 10), Rect(50, 50, 5, 5), Rect(2, 0, 10, 10)};
    const auto keep = nmsBoxes(boxes, {0.9, 0.8, 0.7, 0.95}, 0.5);
    CHECK_EQ(keep.size(), 2u);
    CHECK_EQ(keep[0], 3);
    CHECK_EQ(keep[1], 2);
    CHECK_NEAR(iou(Rect(0, 0, 2, 2), Rect(1, 0, 2, 2)), 1.0 / 3.0, 1e-12);
    ImageF resp(20, 20, 1, 0.f);
    resp(5, 5) = 3;
    resp(6, 5) = 2;
    resp(15, 15) = 1;
    const auto peaks = findPeaks(resp, 0.5f, 3);
    CHECK_EQ(peaks.size(), 2u);
    CHECK(peaks[0].pos == Point(5, 5));
}
