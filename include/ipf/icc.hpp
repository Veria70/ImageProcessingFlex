// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf icc: ICC color management. Parses ICC v2/v4 profiles (matrix/TRC RGB, gray TRC and
// LUT-based profiles: lut8 'mft1', lut16 'mft2', lutAtoB 'mAB ', lutBtoA 'mBA ', e.g. CMYK
// printer profiles) and converts images between profiles through the D50 profile connection
// space (PCS).
//
//   ipf::ImageMetadata meta;
//   ipf::ImageU8 img = ipf::loadImage("photo.jpg", {}, &meta);
//   if (!meta.iccProfile.empty())
//       img = ipf::convertColorProfile(img, ipf::IccProfile::fromBytes(meta.iccProfile));  // -> sRGB
//
// (loadImage can do this automatically with LoadOptions::convertToSRGB.)
// Supported intents: perceptual, relative colorimetric and saturation (absolute colorimetric
// is treated as relative). No black point compensation.
#pragma once

#include <array>
#include <memory>

#include "codec/common.hpp"
#include "core.hpp"

namespace ipf {

enum class RenderingIntent { Perceptual = 0, RelativeColorimetric = 1, Saturation = 2, AbsoluteColorimetric = 3 };

namespace detail {
namespace icc {

constexpr double kD50[3] = {0.9642, 1.0, 0.8249};

inline std::uint32_t sig(const char* s) {
    return (static_cast<std::uint32_t>(static_cast<unsigned char>(s[0])) << 24) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(s[1])) << 16) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(s[2])) << 8) | static_cast<unsigned char>(s[3]);
}

/// A one-dimensional transfer curve (identity, gamma, sampled table or parametric).
struct Curve {
    enum Kind { Identity, Gamma, Table, Param } kind = Identity;
    double gamma = 1.0;
    std::vector<double> table;  // samples over [0, 1]
    int ptype = 0;
    double p[7] = {1, 1, 0, 0, 0, 0, 0};  // g, a, b, c, d, e, f

    double eval(double x) const {
        x = clamp(x, 0.0, 1.0);
        switch (kind) {
            case Gamma: return std::pow(x, gamma);
            case Table: {
                const double pos = x * static_cast<double>(table.size() - 1);
                const std::size_t i = std::min(table.size() - 2, static_cast<std::size_t>(pos));
                const double f = pos - static_cast<double>(i);
                return table[i] + (table[i + 1] - table[i]) * f;
            }
            case Param: {
                const double g = p[0], a = p[1], b = p[2], c = p[3], d = p[4], e = p[5], f = p[6];
                auto pw = [g](double v) { return v > 0 ? std::pow(v, g) : 0.0; };
                switch (ptype) {
                    case 0: return pw(x);
                    case 1: return x >= -b / a ? pw(a * x + b) : 0.0;
                    case 2: return x >= -b / a ? pw(a * x + b) + c : c;
                    case 3: return x >= d ? pw(a * x + b) : c * x;
                    default: return x >= d ? pw(a * x + b) + e : c * x + f;
                }
            }
            default: return x;
        }
    }

    double inverse(double y) const {
        y = clamp(y, 0.0, 1.0);
        switch (kind) {
            case Gamma: return std::pow(y, 1.0 / gamma);
            case Table: {  // monotonic table: binary search + linear interpolation
                const bool up = table.back() >= table.front();
                std::size_t lo = 0, hi = table.size() - 1;
                if (up ? y <= table[lo] : y >= table[lo]) return 0.0;
                if (up ? y >= table[hi] : y <= table[hi]) return 1.0;
                while (hi - lo > 1) {
                    const std::size_t mid = (lo + hi) / 2;
                    if (up ? table[mid] <= y : table[mid] >= y) lo = mid;
                    else hi = mid;
                }
                const double span = table[hi] - table[lo];
                const double f = span != 0 ? (y - table[lo]) / span : 0.0;
                return (static_cast<double>(lo) + f) / static_cast<double>(table.size() - 1);
            }
            case Param: {
                const double g = p[0], a = p[1], b = p[2], c = p[3], d = p[4], e = p[5], f = p[6];
                auto root = [g](double v) { return v > 0 ? std::pow(v, 1.0 / g) : 0.0; };
                double x;
                switch (ptype) {
                    case 0: x = root(y); break;
                    case 1: x = (root(y) - b) / a; break;
                    case 2: x = y > c ? (root(y - c) - b) / a : -b / a; break;
                    case 3: {
                        const double yd = std::pow(std::max(0.0, a * d + b), g);
                        x = y >= yd ? (root(y) - b) / a : (c != 0 ? y / c : 0.0);
                        break;
                    }
                    default: {
                        const double yd = std::pow(std::max(0.0, a * d + b), g) + e;
                        x = y >= yd ? (root(y - e) - b) / a : (c != 0 ? (y - f) / c : 0.0);
                        break;
                    }
                }
                return clamp(x, 0.0, 1.0);
            }
            default: return y;
        }
    }
};

/// A multi-dimensional lookup pipeline (lut8 / lut16 / lutAtoB / lutBtoA).
struct Lut {
    int in = 0, out = 0;
    bool isAB = false, isBA = false;  // mAB / mBA ordering; otherwise mft1 / mft2
    bool legacy16 = false;            // mft2: 16-bit legacy Lab PCS encoding
    bool trilinear = false;           // Lab-indexed 3D tables use trilinear interpolation (as littleCMS)
    std::vector<Curve> inCurves, outCurves, mCurves;  // (A/B naming handled by the evaluation order)
    bool hasMatrix = false;
    double matrix[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};  // 3x3 + offsets (mAB/mBA)
    std::vector<int> grid;     // grid points per input dimension (empty = no CLUT)
    std::vector<double> clut;  // normalized samples, outputs interleaved, first input slowest

    void applyMatrix(double* v) const {
        const double x = v[0], y = v[1], z = v[2];
        v[0] = clamp(matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[9], 0.0, 1.0);
        v[1] = clamp(matrix[3] * x + matrix[4] * y + matrix[5] * z + matrix[10], 0.0, 1.0);
        v[2] = clamp(matrix[6] * x + matrix[7] * y + matrix[8] * z + matrix[11], 0.0, 1.0);
    }

    // Interpolation of the CLUT, following littleCMS: tetrahedral for 3 inputs, recursive linear
    // interpolation over the first input for more, (bi)linear for fewer.
    void tetra(const double* v, std::size_t base, const std::size_t* stride, int dim0, double* outv) const {
        double r[3];
        std::size_t x0[3], x1[3];
        for (int k = 0; k < 3; ++k) {
            const int g = grid[static_cast<std::size_t>(dim0 + k)];
            const double pos = clamp(v[k], 0.0, 1.0) * (g - 1);
            const int i = std::min(g - 1, static_cast<int>(pos));
            r[k] = pos - i;
            x0[k] = static_cast<std::size_t>(i) * stride[dim0 + k];
            x1[k] = x0[k] + (r[k] > 0 && i < g - 1 ? stride[dim0 + k] : 0);
        }
        auto V = [&](std::size_t a, std::size_t b, std::size_t c, int o) { return clut[base + a + b + c + static_cast<std::size_t>(o)]; };
        const double rx = r[0], ry = r[1], rz = r[2];
        for (int o = 0; o < out; ++o) {
            const double c0 = V(x0[0], x0[1], x0[2], o);
            double c1 = 0, c2 = 0, c3 = 0;
            if (rx >= ry && ry >= rz) {
                c1 = V(x1[0], x0[1], x0[2], o) - c0;
                c2 = V(x1[0], x1[1], x0[2], o) - V(x1[0], x0[1], x0[2], o);
                c3 = V(x1[0], x1[1], x1[2], o) - V(x1[0], x1[1], x0[2], o);
            } else if (rx >= rz && rz >= ry) {
                c1 = V(x1[0], x0[1], x0[2], o) - c0;
                c2 = V(x1[0], x1[1], x1[2], o) - V(x1[0], x0[1], x1[2], o);
                c3 = V(x1[0], x0[1], x1[2], o) - V(x1[0], x0[1], x0[2], o);
            } else if (rz >= rx && rx >= ry) {
                c1 = V(x1[0], x0[1], x1[2], o) - V(x0[0], x0[1], x1[2], o);
                c2 = V(x1[0], x1[1], x1[2], o) - V(x1[0], x0[1], x1[2], o);
                c3 = V(x0[0], x0[1], x1[2], o) - c0;
            } else if (ry >= rx && rx >= rz) {
                c1 = V(x1[0], x1[1], x0[2], o) - V(x0[0], x1[1], x0[2], o);
                c2 = V(x0[0], x1[1], x0[2], o) - c0;
                c3 = V(x1[0], x1[1], x1[2], o) - V(x1[0], x1[1], x0[2], o);
            } else if (ry >= rz && rz >= rx) {
                c1 = V(x1[0], x1[1], x1[2], o) - V(x0[0], x1[1], x1[2], o);
                c2 = V(x0[0], x1[1], x0[2], o) - c0;
                c3 = V(x0[0], x1[1], x1[2], o) - V(x0[0], x1[1], x0[2], o);
            } else if (rz >= ry && ry >= rx) {
                c1 = V(x1[0], x1[1], x1[2], o) - V(x0[0], x1[1], x1[2], o);
                c2 = V(x0[0], x1[1], x1[2], o) - V(x0[0], x0[1], x1[2], o);
                c3 = V(x0[0], x0[1], x1[2], o) - c0;
            }
            outv[o] = c0 + c1 * rx + c2 * ry + c3 * rz;
        }
    }

    void interp(const double* v, std::size_t base, const std::size_t* stride, int dim0, int n, double* outv) const {
        if (n == 3 && !trilinear) {
            tetra(v, base, stride, dim0, outv);
            return;
        }
        const int g = grid[static_cast<std::size_t>(dim0)];
        const double pos = clamp(v[0], 0.0, 1.0) * (g - 1);
        const int i = std::min(g - 1, static_cast<int>(pos));
        const double f = pos - i;
        if (n == 1) {
            const std::size_t a = base + static_cast<std::size_t>(i) * stride[dim0];
            const std::size_t b = a + (f > 0 && i < g - 1 ? stride[dim0] : 0);
            for (int o = 0; o < out; ++o) outv[o] = clut[a + static_cast<std::size_t>(o)] * (1 - f) + clut[b + static_cast<std::size_t>(o)] * f;
            return;
        }
        double lo[16], hi[16];
        interp(v + 1, base + static_cast<std::size_t>(i) * stride[dim0], stride, dim0 + 1, n - 1, lo);
        if (f > 0 && i < g - 1) interp(v + 1, base + static_cast<std::size_t>(i + 1) * stride[dim0], stride, dim0 + 1, n - 1, hi);
        else std::copy(lo, lo + out, hi);
        for (int o = 0; o < out; ++o) outv[o] = lo[o] + (hi[o] - lo[o]) * f;
    }

    void applyClut(const double* v, double* outv) const {
        std::size_t stride[16];
        std::size_t s = static_cast<std::size_t>(out);
        for (int k = in - 1; k >= 0; --k) {
            stride[k] = s;
            s *= static_cast<std::size_t>(grid[static_cast<std::size_t>(k)]);
        }
        interp(v, 0, stride, 0, in, outv);
    }

    /// Evaluates the pipeline on normalized inputs (in channels) producing out channels.
    void eval(const double* inv, double* outv) const {
        double a[16], b[16];
        std::copy(inv, inv + in, a);
        auto curves = [](const std::vector<Curve>& cs, double* v, int n) {
            if (cs.empty()) return;
            for (int k = 0; k < n; ++k) v[k] = cs[static_cast<std::size_t>(k)].eval(v[k]);
        };
        if (isAB) {  // A curves -> CLUT -> M curves -> matrix -> B curves
            curves(inCurves, a, in);
            if (!grid.empty()) applyClut(a, b);
            else std::copy(a, a + out, b);
            curves(mCurves, b, out);
            if (hasMatrix) applyMatrix(b);
            curves(outCurves, b, out);
        } else if (isBA) {  // B curves -> matrix -> M curves -> CLUT -> A curves
            curves(inCurves, a, in);
            if (hasMatrix) applyMatrix(a);
            curves(mCurves, a, in);
            if (!grid.empty()) applyClut(a, b);
            else std::copy(a, a + out, b);
            curves(outCurves, b, out);
        } else {  // lut8 / lut16: (matrix, XYZ input only) -> input curves -> CLUT -> output curves
            if (hasMatrix) applyMatrix(a);
            curves(inCurves, a, in);
            applyClut(a, b);
            curves(outCurves, b, out);
        }
        std::copy(b, b + out, outv);
    }
};

inline void labToXyz(const double* lab, double* xyz) {
    const double fy = (lab[0] + 16.0) / 116.0, fx = fy + lab[1] / 500.0, fz = fy - lab[2] / 200.0;
    auto inv = [](double t) { return t > 6.0 / 29.0 ? t * t * t : 3 * (6.0 / 29.0) * (6.0 / 29.0) * (t - 4.0 / 29.0); };
    xyz[0] = kD50[0] * inv(fx);
    xyz[1] = kD50[1] * inv(fy);
    xyz[2] = kD50[2] * inv(fz);
}
inline void xyzToLab(const double* xyz, double* lab) {
    auto f = [](double t) { return t > 216.0 / 24389.0 ? std::cbrt(t) : (24389.0 / 27.0 * t + 16.0) / 116.0; };
    const double fx = f(xyz[0] / kD50[0]), fy = f(xyz[1] / kD50[1]), fz = f(xyz[2] / kD50[2]);
    lab[0] = 116.0 * fy - 16.0;
    lab[1] = 500.0 * (fx - fy);
    lab[2] = 200.0 * (fy - fz);
}

inline bool invert3x3(const double* m, double* r) {
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (std::fabs(det) < 1e-12) return false;
    r[0] = (m[4] * m[8] - m[5] * m[7]) / det;
    r[1] = (m[2] * m[7] - m[1] * m[8]) / det;
    r[2] = (m[1] * m[5] - m[2] * m[4]) / det;
    r[3] = (m[5] * m[6] - m[3] * m[8]) / det;
    r[4] = (m[0] * m[8] - m[2] * m[6]) / det;
    r[5] = (m[2] * m[3] - m[0] * m[5]) / det;
    r[6] = (m[3] * m[7] - m[4] * m[6]) / det;
    r[7] = (m[1] * m[6] - m[0] * m[7]) / det;
    r[8] = (m[0] * m[4] - m[1] * m[3]) / det;
    return true;
}

}  // namespace icc
}  // namespace detail

/// A parsed ICC profile that can map device colors to and from the D50 PCS.
class IccProfile {
public:
    IccProfile() = default;

    /// Parses an ICC profile (e.g. ImageMetadata::iccProfile). Throws ipf::Error if the profile
    /// is malformed or uses no supported way to reach the PCS.
    static IccProfile fromBytes(const std::vector<std::uint8_t>& bytes) {
        using namespace detail::icc;
        IccProfile p;
        if (bytes.size() < 132) detail::fail("ICC: profile too short");
        detail::ByteReader h(bytes.data(), bytes.size(), "ICC");
        h.skip(8);
        p.version_ = h.u8();
        h.seek(12);
        p.deviceClass_ = h.u32be();
        p.colorSpace_ = h.u32be();
        p.pcs_ = h.u32be();
        if (p.pcs_ != sig("XYZ ") && p.pcs_ != sig("Lab ")) detail::fail("ICC: unsupported PCS");
        p.channels_ = channelsOf(p.colorSpace_);
        if (!p.channels_) detail::fail("ICC: unsupported data color space");
        h.seek(128);
        const std::uint32_t count = h.u32be();
        if (count > 1000) detail::fail("ICC: too many tags");
        struct Tag {
            std::uint32_t sig, off, size;
        };
        std::vector<Tag> tags;
        for (std::uint32_t i = 0; i < count; ++i) {
            Tag t{h.u32be(), h.u32be(), h.u32be()};
            if (static_cast<std::size_t>(t.off) + t.size > bytes.size() || t.size < 8) detail::fail("ICC: tag outside the profile");
            tags.push_back(t);
        }
        auto find = [&](const char* name) -> const Tag* {
            for (const Tag& t : tags)
                if (t.sig == sig(name)) return &t;
            return nullptr;
        };
        auto reader = [&](const Tag& t) { return detail::ByteReader(bytes.data() + t.off, t.size, "ICC tag"); };
        if (const Tag* t = find("desc")) p.description_ = parseText(bytes, *t);
        const char* a2b[3] = {"A2B0", "A2B1", "A2B2"};
        const char* b2a[3] = {"B2A0", "B2A1", "B2A2"};
        for (int k = 0; k < 3; ++k) {
            if (const Tag* t = find(a2b[k])) {
                auto rd = reader(*t);
                p.a2b_[k] = std::make_shared<Lut>(parseLut(rd));
            }
            if (const Tag* t = find(b2a[k])) {
                auto rd = reader(*t);
                p.b2a_[k] = std::make_shared<Lut>(parseLut(rd));
                p.b2a_[k]->trilinear = p.pcs_ == sig("Lab ");  // Lab is not a cube-like space
            }
        }
        const Tag *rX = find("rXYZ"), *gX = find("gXYZ"), *bX = find("bXYZ");
        const Tag *rT = find("rTRC"), *gT = find("gTRC"), *bT = find("bTRC"), *kT = find("kTRC");
        if (p.channels_ == 3 && rX && gX && bX && rT && gT && bT) {
            const Tag* cols[3] = {rX, gX, bX};
            for (int c = 0; c < 3; ++c) {
                auto rd = reader(*cols[c]);
                if (rd.u32be() != sig("XYZ ")) detail::fail("ICC: bad XYZ tag");
                rd.skip(4);
                for (int k = 0; k < 3; ++k) p.matrix_[k * 3 + c] = s15(rd.u32be());
            }
            const Tag* trcs[3] = {rT, gT, bT};
            for (int c = 0; c < 3; ++c) {
                auto rd = reader(*trcs[c]);
                p.trc_[c] = parseCurve(rd);
            }
            p.matrixShaper_ = invert3x3(p.matrix_, p.inverse_);
        } else if (p.channels_ == 1 && kT) {
            auto rd = reader(*kT);
            p.trc_[0] = parseCurve(rd);
            p.grayTrc_ = true;
        }
        if (!p.a2b_[0] && !p.a2b_[1] && !p.a2b_[2] && !p.matrixShaper_ && !p.grayTrc_)
            detail::fail("ICC: profile has no supported device-to-PCS transform");
        p.valid_ = true;
        return p;
    }

    /// Matrix/TRC RGB profile from chromaticities (xy of red, green, blue and white) and a
    /// transfer curve given as an ICC parametric curve (type 0-4, parameters g a b c d e f).
    static IccProfile fromPrimaries(const std::array<double, 8>& xy, int curveType, const std::array<double, 7>& params,
                                    const std::string& description = "") {
        using namespace detail::icc;
        IccProfile p;
        p.colorSpace_ = sig("RGB ");
        p.pcs_ = sig("XYZ ");
        p.deviceClass_ = sig("mntr");
        p.channels_ = 3;
        p.version_ = 4;
        p.description_ = description;
        // RGB -> XYZ for the given white, then Bradford-adapt to D50
        auto toXYZ = [](double x, double y) { return std::array<double, 3>{x / y, 1.0, (1 - x - y) / y}; };
        const auto R = toXYZ(xy[0], xy[1]), G = toXYZ(xy[2], xy[3]), B = toXYZ(xy[4], xy[5]), Wt = toXYZ(xy[6], xy[7]);
        const double P[9] = {R[0], G[0], B[0], R[1], G[1], B[1], R[2], G[2], B[2]};
        double Pi[9];
        invert3x3(P, Pi);
        const double S[3] = {Pi[0] * Wt[0] + Pi[1] * Wt[1] + Pi[2] * Wt[2], Pi[3] * Wt[0] + Pi[4] * Wt[1] + Pi[5] * Wt[2],
                             Pi[6] * Wt[0] + Pi[7] * Wt[1] + Pi[8] * Wt[2]};
        double M[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) M[r * 3 + c] = P[r * 3 + c] * S[c];
        const double brad[9] = {0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296};
        double bradInv[9];
        invert3x3(brad, bradInv);
        double cs[3], cd[3];
        for (int r = 0; r < 3; ++r) {
            cs[r] = brad[r * 3] * Wt[0] + brad[r * 3 + 1] * Wt[1] + brad[r * 3 + 2] * Wt[2];
            cd[r] = brad[r * 3] * kD50[0] + brad[r * 3 + 1] * kD50[1] + brad[r * 3 + 2] * kD50[2];
        }
        double A[9], tmp[9];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) tmp[r * 3 + c] = brad[r * 3 + c] * cd[r] / cs[r];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                A[r * 3 + c] = bradInv[r * 3] * tmp[c] + bradInv[r * 3 + 1] * tmp[3 + c] + bradInv[r * 3 + 2] * tmp[6 + c];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                p.matrix_[r * 3 + c] = A[r * 3] * M[c] + A[r * 3 + 1] * M[3 + c] + A[r * 3 + 2] * M[6 + c];
        for (auto& t : p.trc_) {
            t.kind = Curve::Param;
            t.ptype = curveType;
            for (int k = 0; k < 7; ++k) t.p[k] = params[static_cast<std::size_t>(k)];
        }
        p.matrixShaper_ = invert3x3(p.matrix_, p.inverse_);
        p.valid_ = true;
        return p;
    }

    /// The standard sRGB (IEC 61966-2-1) profile.
    static IccProfile sRGB() {
        return fromPrimaries({0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290}, 4,
                             {2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045, 0, 0}, "sRGB");
    }
    /// Display P3 (DCI-P3 primaries, D65 white, sRGB transfer curve).
    static IccProfile displayP3() {
        return fromPrimaries({0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290}, 4,
                             {2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045, 0, 0}, "Display P3");
    }
    /// Adobe RGB (1998) (gamma 563/256).
    static IccProfile adobeRGB() {
        return fromPrimaries({0.64, 0.33, 0.21, 0.71, 0.15, 0.06, 0.3127, 0.3290}, 0, {563.0 / 256.0, 0, 0, 0, 0, 0, 0},
                             "Adobe RGB (1998)");
    }
    /// Linear-light sRGB primaries (gamma 1).
    static IccProfile linearSRGB() {
        return fromPrimaries({0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290}, 0, {1, 0, 0, 0, 0, 0, 0}, "Linear sRGB");
    }

    bool valid() const { return valid_; }
    int channels() const { return channels_; }
    /// Data color space signature as text, e.g. "RGB ", "GRAY", "CMYK".
    std::string colorSpace() const {
        std::string s(4, ' ');
        for (int k = 0; k < 4; ++k) s[static_cast<std::size_t>(k)] = static_cast<char>((colorSpace_ >> (24 - 8 * k)) & 0xFF);
        return s;
    }
    const std::string& description() const { return description_; }
    bool isMatrixShaper() const { return matrixShaper_; }

    /// Device values (normalized to [0, 1]) -> PCS XYZ (D50, Y of white = 1).
    void toXYZ(const double* in, double* xyz, RenderingIntent intent = RenderingIntent::Perceptual) const {
        using namespace detail::icc;
        if (const Lut* lut = pick(a2b_, intent)) {
            double v[16];
            lut->eval(in, v);
            decodePCS(*lut, v, xyz);
            return;
        }
        if (grayTrc_) {
            const double y = trc_[0].eval(in[0]);
            for (int k = 0; k < 3; ++k) xyz[k] = kD50[k] * y;
            return;
        }
        const double r = trc_[0].eval(in[0]), g = trc_[1].eval(in[1]), b = trc_[2].eval(in[2]);
        for (int k = 0; k < 3; ++k) xyz[k] = matrix_[k * 3] * r + matrix_[k * 3 + 1] * g + matrix_[k * 3 + 2] * b;
    }

    /// PCS XYZ (D50) -> device values in [0, 1] (out-of-gamut colors are clipped).
    void fromXYZ(const double* xyz, double* out, RenderingIntent intent = RenderingIntent::Perceptual) const {
        using namespace detail::icc;
        if (const Lut* lut = pick(b2a_, intent)) {
            double v[16];
            encodePCS(*lut, xyz, v);
            lut->eval(v, out);
            for (int k = 0; k < channels_; ++k) out[k] = clamp(out[k], 0.0, 1.0);
            return;
        }
        if (grayTrc_) {
            out[0] = trc_[0].inverse(xyz[1] / kD50[1]);
            return;
        }
        if (!matrixShaper_) detail::fail("ICC: profile cannot be used as a destination (no B2A or matrix/TRC)");
        for (int k = 0; k < 3; ++k) {
            const double lin = inverse_[k * 3] * xyz[0] + inverse_[k * 3 + 1] * xyz[1] + inverse_[k * 3 + 2] * xyz[2];
            out[k] = trc_[k].inverse(clamp(lin, 0.0, 1.0));
        }
    }

private:
    static int channelsOf(std::uint32_t cs) {
        using detail::icc::sig;
        if (cs == sig("GRAY")) return 1;
        if (cs == sig("RGB ") || cs == sig("Lab ") || cs == sig("XYZ ") || cs == sig("YCbr") || cs == sig("HSV ") ||
            cs == sig("CMY ") || cs == sig("Luv ") || cs == sig("Yxy ") || cs == sig("HLS "))
            return 3;
        if (cs == sig("CMYK")) return 4;
        for (int n = 2; n <= 15; ++n) {  // "nCLR" spaces
            const char c = static_cast<char>(n < 10 ? '0' + n : 'A' + n - 10);
            const char name[5] = {c, 'C', 'L', 'R', 0};
            if (cs == sig(name)) return n;
        }
        return 0;
    }

    static double s15(std::uint32_t v) { return static_cast<double>(static_cast<std::int32_t>(v)) / 65536.0; }

    template <class TagT>
    static std::string parseText(const std::vector<std::uint8_t>& bytes, const TagT& t) {
        using detail::icc::sig;
        detail::ByteReader rd(bytes.data() + t.off, t.size, "ICC text");
        const std::uint32_t type = rd.u32be();
        rd.skip(4);
        try {
            if (type == sig("desc")) {  // textDescriptionType: ASCII string
                const std::uint32_t n = rd.u32be();
                const std::uint8_t* s = rd.take(n);
                std::string out(reinterpret_cast<const char*>(s), n);
                while (!out.empty() && out.back() == 0) out.pop_back();
                return out;
            }
            if (type == sig("mluc")) {  // first record, UTF-16BE (non-ASCII replaced by '?')
                const std::uint32_t records = rd.u32be();
                rd.skip(4);
                if (!records) return "";
                rd.skip(4);
                const std::uint32_t len = rd.u32be(), off = rd.u32be();
                detail::ByteReader s(bytes.data() + t.off, t.size, "ICC text");
                s.seek(off);
                std::string out;
                for (std::uint32_t i = 0; i + 1 < len; i += 2) {
                    const unsigned ch = s.u16be();
                    out += ch < 0x80 ? static_cast<char>(ch) : '?';
                }
                return out;
            }
            if (type == sig("text")) {
                const std::uint8_t* s = rd.take(rd.remaining());
                std::string out(reinterpret_cast<const char*>(s), t.size - 8);
                while (!out.empty() && out.back() == 0) out.pop_back();
                return out;
            }
        } catch (const Error&) {
        }
        return "";
    }

    static detail::icc::Curve parseCurve(detail::ByteReader& rd) {
        using namespace detail::icc;
        Curve c;
        const std::uint32_t type = rd.u32be();
        rd.skip(4);
        if (type == sig("curv")) {
            const std::uint32_t n = rd.u32be();
            if (n == 0) {
                c.kind = Curve::Identity;
            } else if (n == 1) {
                c.kind = Curve::Gamma;
                c.gamma = rd.u16be() / 256.0;
            } else {
                if (n > (1u << 20)) detail::fail("ICC: curve too large");
                c.kind = Curve::Table;
                c.table.resize(n);
                for (auto& v : c.table) v = rd.u16be() / 65535.0;
            }
        } else if (type == sig("para")) {
            c.kind = Curve::Param;
            c.ptype = rd.u16be();
            rd.skip(2);
            static const int counts[5] = {1, 3, 4, 5, 7};
            if (c.ptype > 4) detail::fail("ICC: unknown parametric curve type");
            const int idx[5][7] = {{0}, {0, 1, 2}, {0, 1, 2, 3}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5, 6}};
            for (int k = 0; k < counts[c.ptype]; ++k) c.p[idx[c.ptype][k]] = s15(rd.u32be());
            if (c.ptype >= 1 && c.p[1] == 0) detail::fail("ICC: invalid parametric curve");
        } else {
            detail::fail("ICC: unsupported curve type");
        }
        return c;
    }

    static detail::icc::Lut parseLut(detail::ByteReader& rd) {
        using namespace detail::icc;
        Lut lut;
        const std::uint32_t type = rd.u32be();
        rd.skip(4);
        lut.in = rd.u8();
        lut.out = rd.u8();
        if (lut.in < 1 || lut.in > 15 || lut.out < 1 || lut.out > 15) detail::fail("ICC: invalid LUT channel counts");
        if (type == sig("mft1") || type == sig("mft2")) {
            const bool is16 = type == sig("mft2");
            lut.legacy16 = is16;
            const int g = rd.u8();
            rd.skip(1);
            for (int k = 0; k < 9; ++k) lut.matrix[k] = s15(rd.u32be());
            const bool identity = lut.matrix[0] == 1 && lut.matrix[4] == 1 && lut.matrix[8] == 1 && lut.matrix[1] == 0 &&
                                  lut.matrix[2] == 0 && lut.matrix[3] == 0 && lut.matrix[5] == 0 && lut.matrix[6] == 0 &&
                                  lut.matrix[7] == 0;
            lut.hasMatrix = !identity && lut.in == 3;  // only meaningful for XYZ input
            const std::uint32_t inEntries = is16 ? rd.u16be() : 256, outEntries = is16 ? rd.u16be() : 256;
            if (inEntries < 2 || outEntries < 2 || inEntries > 4096 || outEntries > 4096 || g < 2)
                detail::fail("ICC: invalid lut16 table sizes");
            auto readTable = [&](std::uint32_t n) {
                Curve c;
                c.kind = Curve::Table;
                c.table.resize(n);
                for (auto& v : c.table) v = is16 ? rd.u16be() / 65535.0 : rd.u8() / 255.0;
                return c;
            };
            for (int k = 0; k < lut.in; ++k) lut.inCurves.push_back(readTable(inEntries));
            lut.grid.assign(static_cast<std::size_t>(lut.in), g);
            std::size_t n = static_cast<std::size_t>(lut.out);
            for (int k = 0; k < lut.in; ++k) {
                n *= static_cast<std::size_t>(g);
                if (n > (1u << 26)) detail::fail("ICC: CLUT too large");
            }
            lut.clut.resize(n);
            for (auto& v : lut.clut) v = is16 ? rd.u16be() / 65535.0 : rd.u8() / 255.0;
            for (int k = 0; k < lut.out; ++k) lut.outCurves.push_back(readTable(outEntries));
            return lut;
        }
        if (type != sig("mAB ") && type != sig("mBA ")) detail::fail("ICC: unsupported LUT type");
        const bool ab = type == sig("mAB ");
        lut.isAB = ab;
        lut.isBA = !ab;
        rd.skip(2);
        const std::uint32_t offB = rd.u32be(), offMat = rd.u32be(), offM = rd.u32be(), offClut = rd.u32be(), offA = rd.u32be();
        // A curves are on the device side (input of AtoB, output of BtoA); B curves on the PCS side.
        auto readCurves = [&](std::uint32_t off, int n) {
            std::vector<Curve> cs;
            if (!off) return cs;
            detail::ByteReader c = rd;
            c.seek(off);
            for (int k = 0; k < n; ++k) {
                cs.push_back(parseCurve(c));
                const std::size_t pos = c.pos();
                c.seek((pos + 3) & ~static_cast<std::size_t>(3));  // curves are 4-byte aligned
            }
            return cs;
        };
        const int devCh = ab ? lut.in : lut.out, pcsCh = ab ? lut.out : lut.in;
        const std::vector<Curve> aCurves = readCurves(offA, devCh), bCurves = readCurves(offB, pcsCh);
        lut.mCurves = readCurves(offM, pcsCh);
        if (offMat) {
            detail::ByteReader m = rd;
            m.seek(offMat);
            for (int k = 0; k < 12; ++k) lut.matrix[k] = s15(m.u32be());
            lut.hasMatrix = true;
        }
        if (offClut) {
            detail::ByteReader c = rd;
            c.seek(offClut);
            const std::uint8_t* gp = c.take(16);
            std::size_t n = static_cast<std::size_t>(lut.out);
            for (int k = 0; k < lut.in; ++k) {
                if (gp[k] < 2) detail::fail("ICC: invalid CLUT grid");
                lut.grid.push_back(gp[k]);
                n *= gp[k];
                if (n > (1u << 26)) detail::fail("ICC: CLUT too large");
            }
            const int prec = c.u8();
            c.skip(3);
            if (prec != 1 && prec != 2) detail::fail("ICC: invalid CLUT precision");
            lut.clut.resize(n);
            for (auto& v : lut.clut) v = prec == 2 ? c.u16be() / 65535.0 : c.u8() / 255.0;
        } else if (lut.in != lut.out) {
            detail::fail("ICC: LUT without CLUT must have equal channel counts");
        }
        lut.inCurves = ab ? aCurves : bCurves;
        lut.outCurves = ab ? bCurves : aCurves;
        return lut;
    }

    const detail::icc::Lut* pick(const std::shared_ptr<detail::icc::Lut>* luts, RenderingIntent intent) const {
        int i = static_cast<int>(intent);
        if (i > 2) i = 1;  // absolute -> relative colorimetric
        if (luts[i]) return luts[i].get();
        return luts[0] ? luts[0].get() : nullptr;
    }

    /// LUT output -> PCS XYZ, honoring the PCS encoding of the LUT type.
    void decodePCS(const detail::icc::Lut& lut, const double* v, double* xyz) const {
        using namespace detail::icc;
        if (pcs_ == sig("XYZ ")) {
            for (int k = 0; k < 3; ++k) xyz[k] = v[k] * (65535.0 / 32768.0);
            return;
        }
        const double s = lut.legacy16 ? 65535.0 / 65280.0 : 1.0;  // lut16 uses the v2 Lab encoding
        const double lab[3] = {v[0] * s * 100.0, v[1] * s * 255.0 - 128.0, v[2] * s * 255.0 - 128.0};
        labToXyz(lab, xyz);
    }
    void encodePCS(const detail::icc::Lut& lut, const double* xyz, double* v) const {
        using namespace detail::icc;
        if (pcs_ == sig("XYZ ")) {
            for (int k = 0; k < 3; ++k) v[k] = clamp(xyz[k] * (32768.0 / 65535.0), 0.0, 1.0);
            return;
        }
        double lab[3];
        xyzToLab(xyz, lab);
        const double s = lut.legacy16 ? 65280.0 / 65535.0 : 1.0;
        v[0] = clamp(lab[0] / 100.0 * s, 0.0, 1.0);
        v[1] = clamp((lab[1] + 128.0) / 255.0 * s, 0.0, 1.0);
        v[2] = clamp((lab[2] + 128.0) / 255.0 * s, 0.0, 1.0);
    }

    bool valid_ = false, matrixShaper_ = false, grayTrc_ = false;
    int version_ = 2, channels_ = 0;
    std::uint32_t deviceClass_ = 0, colorSpace_ = 0, pcs_ = 0;
    std::string description_;
    double matrix_[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, inverse_[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    detail::icc::Curve trc_[3];
    std::shared_ptr<detail::icc::Lut> a2b_[3], b2a_[3];
};

/// Converts an image from the `src` profile to the `dst` profile (default: sRGB). Channel
/// counts follow the profiles (e.g. 4-channel CMYK in, 3-channel RGB out); an extra alpha
/// channel is passed through unchanged. Integer images use their full range, floats [0, 1].
template <class T>
Image<T> convertColorProfile(const Image<T>& img, const IccProfile& src, const IccProfile& dst = IccProfile::sRGB(),
                             RenderingIntent intent = RenderingIntent::Perceptual) {
    detail::require(src.valid() && dst.valid(), "convertColorProfile: invalid profile");
    const int sc = src.channels(), dc = dst.channels(), cn = img.channels();
    detail::require(cn == sc || cn == sc + 1, "convertColorProfile: image channels do not match the source profile");
    const bool alpha = cn == sc + 1;
    Image<T> out(img.width(), img.height(), dc + (alpha ? 1 : 0));
    const double mx = PixelTraits<T>::max();
    detail::forEachPixel(img, out, [&](const T* s, T* d) {
        double in[16], xyz[3], o[16];
        for (int k = 0; k < sc; ++k) in[k] = static_cast<double>(s[k]) / mx;
        src.toXYZ(in, xyz, intent);
        dst.fromXYZ(xyz, o, intent);
        for (int k = 0; k < dc; ++k) d[k] = saturate_cast<T>(o[k] * mx);
        if (alpha) d[dc] = s[sc];
    }, 200);
    return out;
}

}  // namespace ipf
