// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf draw: in-place drawing of lines, rectangles, circles, ellipses, polygons,
// markers and text (built-in 5x8 bitmap font) with clipping, thickness and optional
// anti-aliasing.
//
// Colors: ipf::Color(gray), Color(r, g, b) or Color(r, g, b, a) in the image's value range.
// A gray color on an RGB image is replicated; an RGB color on a gray image is converted to
// luma; images with alpha get an opaque alpha unless one is given.
#pragma once

#include <array>
#include <string>

#include "core.hpp"

namespace ipf {

struct Color {
    std::array<double, 4> v{{0, 0, 0, -1}};
    int n = 1;
    Color(double gray) : v{{gray, gray, gray, -1}}, n(1) {}  // NOLINT: implicit by design
    Color(double r, double g, double b) : v{{r, g, b, -1}}, n(3) {}
    Color(double r, double g, double b, double a) : v{{r, g, b, a}}, n(4) {}
};

namespace detail {

template <class T>
std::vector<double> resolveColor(const Image<T>& img, const Color& col) {
    const int cn = img.channels();
    const double opaque = PixelTraits<T>::max();
    const double gray = col.n == 1 ? col.v[0] : 0.299 * col.v[0] + 0.587 * col.v[1] + 0.114 * col.v[2];
    const double alpha = col.v[3] >= 0 ? col.v[3] : opaque;
    switch (cn) {
        case 1: return {gray};
        case 2: return {gray, alpha};
        case 3: return {col.v[0], col.v[1], col.v[2]};
        case 4: return {col.v[0], col.v[1], col.v[2], alpha};
        default: return std::vector<double>(static_cast<std::size_t>(cn), gray);
    }
}

/// Writes (coverage = 1) or blends (0 < coverage < 1) a color into a pixel, with clipping.
template <class T>
inline void plot(Image<T>& img, int x, int y, const std::vector<double>& col, double coverage = 1.0) {
    if (coverage <= 0 || !img.inside(x, y)) return;
    T* p = img.ptr(x, y);
    for (int c = 0; c < img.channels(); ++c)
        p[c] = coverage >= 1 ? saturate_cast<T>(col[c])
                             : saturate_cast<T>(static_cast<double>(p[c]) * (1 - coverage) + col[c] * coverage);
}

inline double distToSegment(double px, double py, PointF a, PointF b) {
    const double vx = b.x - a.x, vy = b.y - a.y, len2 = vx * vx + vy * vy;
    const double t = len2 > 0 ? clamp(((px - a.x) * vx + (py - a.y) * vy) / len2, 0.0, 1.0) : 0.0;
    return std::hypot(px - (a.x + t * vx), py - (a.y + t * vy));
}

/// Coverage of a pixel at signed distance d inside (d > 0) a shape edge.
inline double coverage(double inside, bool aa) { return aa ? clamp(inside + 0.5, 0.0, 1.0) : (inside >= 0 ? 1.0 : 0.0); }

// 5x8 font for ASCII 32..126 (7-row body + descender row): 5 column bytes per glyph, bit 0 = top row.
inline constexpr std::uint8_t kFont5x7[95][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x5F, 0x00, 0x00}, {0x00, 0x07, 0x00, 0x07, 0x00},
    {0x14, 0x7F, 0x14, 0x7F, 0x14}, {0x24, 0x2A, 0x7F, 0x2A, 0x12}, {0x23, 0x13, 0x08, 0x64, 0x62},
    {0x36, 0x49, 0x55, 0x22, 0x50}, {0x00, 0x05, 0x03, 0x00, 0x00}, {0x00, 0x1C, 0x22, 0x41, 0x00},
    {0x00, 0x41, 0x22, 0x1C, 0x00}, {0x14, 0x08, 0x3E, 0x08, 0x14}, {0x08, 0x08, 0x3E, 0x08, 0x08},
    {0x00, 0x50, 0x30, 0x00, 0x00}, {0x08, 0x08, 0x08, 0x08, 0x08}, {0x00, 0x60, 0x60, 0x00, 0x00},
    {0x20, 0x10, 0x08, 0x04, 0x02}, {0x3E, 0x51, 0x49, 0x45, 0x3E}, {0x00, 0x42, 0x7F, 0x40, 0x00},
    {0x42, 0x61, 0x51, 0x49, 0x46}, {0x21, 0x41, 0x45, 0x4B, 0x31}, {0x18, 0x14, 0x12, 0x7F, 0x10},
    {0x27, 0x45, 0x45, 0x45, 0x39}, {0x3C, 0x4A, 0x49, 0x49, 0x30}, {0x01, 0x71, 0x09, 0x05, 0x03},
    {0x36, 0x49, 0x49, 0x49, 0x36}, {0x06, 0x49, 0x49, 0x29, 0x1E}, {0x00, 0x36, 0x36, 0x00, 0x00},
    {0x00, 0x56, 0x36, 0x00, 0x00}, {0x08, 0x14, 0x22, 0x41, 0x00}, {0x14, 0x14, 0x14, 0x14, 0x14},
    {0x00, 0x41, 0x22, 0x14, 0x08}, {0x02, 0x01, 0x51, 0x09, 0x06}, {0x32, 0x49, 0x79, 0x41, 0x3E},
    {0x7E, 0x11, 0x11, 0x11, 0x7E}, {0x7F, 0x49, 0x49, 0x49, 0x36}, {0x3E, 0x41, 0x41, 0x41, 0x22},
    {0x7F, 0x41, 0x41, 0x22, 0x1C}, {0x7F, 0x49, 0x49, 0x49, 0x41}, {0x7F, 0x09, 0x09, 0x09, 0x01},
    {0x3E, 0x41, 0x49, 0x49, 0x7A}, {0x7F, 0x08, 0x08, 0x08, 0x7F}, {0x00, 0x41, 0x7F, 0x41, 0x00},
    {0x20, 0x40, 0x41, 0x3F, 0x01}, {0x7F, 0x08, 0x14, 0x22, 0x41}, {0x7F, 0x40, 0x40, 0x40, 0x40},
    {0x7F, 0x02, 0x0C, 0x02, 0x7F}, {0x7F, 0x04, 0x08, 0x10, 0x7F}, {0x3E, 0x41, 0x41, 0x41, 0x3E},
    {0x7F, 0x09, 0x09, 0x09, 0x06}, {0x3E, 0x41, 0x51, 0x21, 0x5E}, {0x7F, 0x09, 0x19, 0x29, 0x46},
    {0x46, 0x49, 0x49, 0x49, 0x31}, {0x01, 0x01, 0x7F, 0x01, 0x01}, {0x3F, 0x40, 0x40, 0x40, 0x3F},
    {0x1F, 0x20, 0x40, 0x20, 0x1F}, {0x3F, 0x40, 0x38, 0x40, 0x3F}, {0x63, 0x14, 0x08, 0x14, 0x63},
    {0x07, 0x08, 0x70, 0x08, 0x07}, {0x61, 0x51, 0x49, 0x45, 0x43}, {0x00, 0x7F, 0x41, 0x41, 0x00},
    {0x02, 0x04, 0x08, 0x10, 0x20}, {0x00, 0x41, 0x41, 0x7F, 0x00}, {0x04, 0x02, 0x01, 0x02, 0x04},
    {0x40, 0x40, 0x40, 0x40, 0x40}, {0x00, 0x01, 0x02, 0x04, 0x00}, {0x20, 0x54, 0x54, 0x54, 0x78},
    {0x7F, 0x48, 0x44, 0x44, 0x38}, {0x38, 0x44, 0x44, 0x44, 0x20}, {0x38, 0x44, 0x44, 0x48, 0x7F},
    {0x38, 0x54, 0x54, 0x54, 0x18}, {0x08, 0x7E, 0x09, 0x01, 0x02}, {0x18, 0xA4, 0xA4, 0xA4, 0x7C},
    {0x7F, 0x08, 0x04, 0x04, 0x78}, {0x00, 0x44, 0x7D, 0x40, 0x00}, {0x40, 0x80, 0x84, 0x7D, 0x00},
    {0x7F, 0x10, 0x28, 0x44, 0x00}, {0x00, 0x41, 0x7F, 0x40, 0x00}, {0x7C, 0x04, 0x18, 0x04, 0x78},
    {0x7C, 0x08, 0x04, 0x04, 0x78}, {0x38, 0x44, 0x44, 0x44, 0x38}, {0xFC, 0x24, 0x24, 0x24, 0x18},
    {0x18, 0x24, 0x24, 0x24, 0xFC}, {0x7C, 0x08, 0x04, 0x04, 0x08}, {0x48, 0x54, 0x54, 0x54, 0x20},
    {0x04, 0x3F, 0x44, 0x40, 0x20}, {0x3C, 0x40, 0x40, 0x20, 0x7C}, {0x1C, 0x20, 0x40, 0x20, 0x1C},
    {0x3C, 0x40, 0x30, 0x40, 0x3C}, {0x44, 0x28, 0x10, 0x28, 0x44}, {0x1C, 0xA0, 0xA0, 0xA0, 0x7C},
    {0x44, 0x64, 0x54, 0x4C, 0x44}, {0x00, 0x08, 0x36, 0x41, 0x00}, {0x00, 0x00, 0x7F, 0x00, 0x00},
    {0x00, 0x41, 0x36, 0x08, 0x00}, {0x08, 0x04, 0x08, 0x10, 0x08}};

}  // namespace detail

/// Sets one pixel (ignored when outside the image).
template <class T>
void drawPixel(Image<T>& img, Point p, const Color& color) {
    detail::plot(img, p.x, p.y, detail::resolveColor(img, color));
}

/// Draws a line segment. thickness 1 without anti-aliasing uses Bresenham's algorithm;
/// otherwise the segment is rendered as a round-capped stroke of the given width.
template <class T>
void drawLine(Image<T>& img, PointF p0, PointF p1, const Color& color, int thickness = 1, bool antialias = false) {
    const std::vector<double> col = detail::resolveColor(img, color);
    if (thickness <= 1 && !antialias) {
        int x0 = static_cast<int>(std::lround(p0.x)), y0 = static_cast<int>(std::lround(p0.y));
        const int x1 = static_cast<int>(std::lround(p1.x)), y1 = static_cast<int>(std::lround(p1.y));
        const int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1, dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;
        for (;;) {
            detail::plot(img, x0, y0, col);
            if (x0 == x1 && y0 == y1) break;
            const int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
        return;
    }
    const double half = std::max(1, thickness) / 2.0;
    const int xa = static_cast<int>(std::floor(std::min(p0.x, p1.x) - half - 1));
    const int xb = static_cast<int>(std::ceil(std::max(p0.x, p1.x) + half + 1));
    const int ya = static_cast<int>(std::floor(std::min(p0.y, p1.y) - half - 1));
    const int yb = static_cast<int>(std::ceil(std::max(p0.y, p1.y) + half + 1));
    for (int y = std::max(ya, 0); y <= std::min(yb, img.height() - 1); ++y)
        for (int x = std::max(xa, 0); x <= std::min(xb, img.width() - 1); ++x)
            detail::plot(img, x, y, col, detail::coverage(half - detail::distToSegment(x, y, p0, p1), antialias));
}

/// Draws a connected polyline through `pts` (closed = true joins the last point to the first).
template <class T>
void drawPolyline(Image<T>& img, const std::vector<Point>& pts, bool closed, const Color& color, int thickness = 1,
                  bool antialias = false) {
    for (std::size_t i = 0; i + 1 < pts.size(); ++i) drawLine(img, pts[i], pts[i + 1], color, thickness, antialias);
    if (closed && pts.size() > 2) drawLine(img, pts.back(), pts.front(), color, thickness, antialias);
}

/// Fills a polygon (even-odd rule, any shape incl. self-intersecting), boundary included.
template <class T>
void fillPolygon(Image<T>& img, const std::vector<PointF>& pts, const Color& color) {
    if (pts.size() < 3) return;
    const std::vector<double> col = detail::resolveColor(img, color);
    double ymin = pts[0].y, ymax = pts[0].y;
    for (const PointF& p : pts) {
        ymin = std::min(ymin, p.y);
        ymax = std::max(ymax, p.y);
    }
    std::vector<double> xs;
    for (int y = std::max(0, static_cast<int>(std::ceil(ymin))); y <= std::min(img.height() - 1, static_cast<int>(std::floor(ymax))); ++y) {
        xs.clear();
        for (std::size_t i = 0, j = pts.size() - 1; i < pts.size(); j = i++) {
            const PointF a = pts[j], b = pts[i];
            if ((a.y <= y && b.y > y) || (b.y <= y && a.y > y)) xs.push_back(a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y));
        }
        std::sort(xs.begin(), xs.end());
        for (std::size_t k = 0; k + 1 < xs.size(); k += 2)
            for (int x = std::max(0, static_cast<int>(std::ceil(xs[k]))); x <= std::min(img.width() - 1, static_cast<int>(std::floor(xs[k + 1]))); ++x)
                detail::plot(img, x, y, col);
    }
    // include the boundary itself (like OpenCV's fillPoly)
    for (std::size_t i = 0; i < pts.size(); ++i) drawLine(img, pts[i], pts[(i + 1) % pts.size()], color);
}
template <class T>
void fillPolygon(Image<T>& img, const std::vector<Point>& pts, const Color& color) {
    fillPolygon(img, std::vector<PointF>(pts.begin(), pts.end()), color);
}

/// Draws a rectangle outline of the given thickness (growing inwards); thickness < 0 fills it.
template <class T>
void drawRectangle(Image<T>& img, const Rect& r, const Color& color, int thickness = 1) {
    if (r.empty()) return;
    const std::vector<double> col = detail::resolveColor(img, color);
    const Rect clipped = r & img.rect();
    const int t = thickness < 0 ? std::max(r.width, r.height) : thickness;
    for (int y = clipped.y; y < clipped.bottom(); ++y)
        for (int x = clipped.x; x < clipped.right(); ++x) {
            const bool edge = x < r.x + t || x >= r.right() - t || y < r.y + t || y >= r.bottom() - t;
            if (edge) detail::plot(img, x, y, col);
        }
}

/// Draws a circle. thickness < 0 fills the disk; thickness 1 without anti-aliasing uses the
/// midpoint algorithm; otherwise a ring of the given width is rendered.
template <class T>
void drawCircle(Image<T>& img, PointF center, double radius, const Color& color, int thickness = 1, bool antialias = false) {
    const std::vector<double> col = detail::resolveColor(img, color);
    if (thickness == 1 && !antialias) {
        const int cx = static_cast<int>(std::lround(center.x)), cy = static_cast<int>(std::lround(center.y));
        int x = static_cast<int>(std::lround(radius)), y = 0, err = 1 - x;
        while (x >= y) {
            const int pts[8][2] = {{x, y}, {y, x}, {-y, x}, {-x, y}, {-x, -y}, {-y, -x}, {y, -x}, {x, -y}};
            for (const auto& p : pts) detail::plot(img, cx + p[0], cy + p[1], col);
            ++y;
            if (err < 0) {
                err += 2 * y + 1;
            } else {
                --x;
                err += 2 * (y - x) + 1;
            }
        }
        return;
    }
    const double half = thickness < 0 ? 0 : thickness / 2.0;
    const double outer = radius + half + 1;
    for (int y = std::max(0, static_cast<int>(std::floor(center.y - outer)));
         y <= std::min(img.height() - 1, static_cast<int>(std::ceil(center.y + outer))); ++y)
        for (int x = std::max(0, static_cast<int>(std::floor(center.x - outer)));
             x <= std::min(img.width() - 1, static_cast<int>(std::ceil(center.x + outer))); ++x) {
            const double d = std::hypot(x - center.x, y - center.y);
            const double inside = thickness < 0 ? radius - d : half - std::fabs(d - radius);
            detail::plot(img, x, y, col, detail::coverage(inside, antialias));
        }
}

/// Draws a (rotated) ellipse with semi-axes (axes.x, axes.y) rotated by angleDegrees
/// (counter-clockwise as displayed). thickness < 0 fills it.
template <class T>
void drawEllipse(Image<T>& img, PointF center, PointF axes, double angleDegrees, const Color& color, int thickness = 1,
                 bool antialias = false) {
    const int n = std::max(16, static_cast<int>(std::ceil((axes.x + axes.y) * 2)));
    const double a = -angleDegrees * detail::kPi / 180.0, ca = std::cos(a), sa = std::sin(a);
    std::vector<PointF> poly;
    for (int i = 0; i < n; ++i) {
        const double t = 2 * detail::kPi * i / n, ex = axes.x * std::cos(t), ey = axes.y * std::sin(t);
        poly.push_back({center.x + ex * ca - ey * sa, center.y + ex * sa + ey * ca});
    }
    if (thickness < 0) {
        fillPolygon(img, poly, color);
        return;
    }
    for (int i = 0; i < n; ++i) drawLine(img, poly[i], poly[(i + 1) % n], color, thickness, antialias);
}

enum class MarkerType { Cross, TiltedCross, Square, Diamond, Dot };

/// Draws a marker symbol of the given size centered at p.
template <class T>
void drawMarker(Image<T>& img, PointF p, const Color& color, MarkerType type = MarkerType::Cross, int size = 9,
                int thickness = 1) {
    const double h = size / 2.0;
    switch (type) {
        case MarkerType::Cross:
            drawLine(img, {p.x - h, p.y}, {p.x + h, p.y}, color, thickness);
            drawLine(img, {p.x, p.y - h}, {p.x, p.y + h}, color, thickness);
            break;
        case MarkerType::TiltedCross:
            drawLine(img, {p.x - h, p.y - h}, {p.x + h, p.y + h}, color, thickness);
            drawLine(img, {p.x - h, p.y + h}, {p.x + h, p.y - h}, color, thickness);
            break;
        case MarkerType::Square:
            drawPolyline(img, {Point(int(p.x - h), int(p.y - h)), Point(int(p.x + h), int(p.y - h)),
                               Point(int(p.x + h), int(p.y + h)), Point(int(p.x - h), int(p.y + h))}, true, color, thickness);
            break;
        case MarkerType::Diamond:
            drawPolyline(img, {Point(int(p.x), int(p.y - h)), Point(int(p.x + h), int(p.y)), Point(int(p.x), int(p.y + h)),
                               Point(int(p.x - h), int(p.y))}, true, color, thickness);
            break;
        default:
            drawCircle(img, p, h, color, -1);
            break;
    }
}

/// Size in pixels of `text` rendered by drawText at the given integer scale.
inline Size getTextSize(const std::string& text, int scale = 1) {
    int lines = 1, cur = 0, widest = 0;
    for (char ch : text) {
        if (ch == '\n') {
            ++lines;
            cur = 0;
            continue;
        }
        widest = std::max(widest, ++cur);
    }
    return Size(widest > 0 ? (widest * 6 - 1) * scale : 0, (lines * 10 - 2) * scale);
}

/// Renders ASCII text with the built-in 5x8 font; `origin` is the top-left corner.
/// Each glyph cell is 6 x 10 pixels times `scale`; '\n' starts a new line.
template <class T>
void drawText(Image<T>& img, const std::string& text, Point origin, const Color& color, int scale = 1) {
    detail::require(scale >= 1, "drawText: scale must be >= 1");
    const std::vector<double> col = detail::resolveColor(img, color);
    int cx = origin.x, cy = origin.y;
    for (char ch : text) {
        if (ch == '\n') {
            cx = origin.x;
            cy += 10 * scale;
            continue;
        }
        const int code = static_cast<unsigned char>(ch);
        const std::uint8_t* glyph = detail::kFont5x7[(code >= 32 && code <= 126) ? code - 32 : '?' - 32];
        for (int gx = 0; gx < 5; ++gx)
            for (int gy = 0; gy < 8; ++gy)
                if (glyph[gx] & (1 << gy))
                    for (int sy = 0; sy < scale; ++sy)
                        for (int sx = 0; sx < scale; ++sx) detail::plot(img, cx + gx * scale + sx, cy + gy * scale + sy, col);
        cx += 6 * scale;
    }
}

}  // namespace ipf
