// Minimal .npy reader/writer used by the cross-validation tool (test-only code).
#pragma once

#include <ipf/codec/common.hpp>

#include <sstream>
#include <string>

namespace npy {

template <class T>
const char* descr() {
    if (std::is_same<T, std::uint8_t>::value) return "|u1";
    if (std::is_same<T, std::uint16_t>::value) return "<u2";
    if (std::is_same<T, std::int16_t>::value) return "<i2";
    if (std::is_same<T, std::int32_t>::value) return "<i4";
    if (std::is_same<T, float>::value) return "<f4";
    if (std::is_same<T, double>::value) return "<f8";
    return "??";
}

/// Writes an image as a (h, w, c) array (c axis dropped for single-channel images).
template <class T>
void save(const ipf::Image<T>& img, const std::string& path) {
    std::ostringstream h;
    h << "{'descr': '" << descr<T>() << "', 'fortran_order': False, 'shape': (" << img.height() << ", " << img.width();
    if (img.channels() > 1) h << ", " << img.channels();
    h << "), }";
    std::string header = h.str();
    while ((10 + header.size() + 1) % 64) header += ' ';
    header += '\n';
    std::vector<std::uint8_t> out = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    ipf::detail::put16le(out, static_cast<unsigned>(header.size()));
    out.insert(out.end(), header.begin(), header.end());
    const auto* p = reinterpret_cast<const std::uint8_t*>(img.data());
    out.insert(out.end(), p, p + img.sampleCount() * sizeof(T));
    ipf::writeFile(path, out);
}

struct Array {
    std::string descr;
    std::vector<long long> shape;
    std::vector<std::uint8_t> data;
};

inline Array loadRaw(const std::string& path) {
    const std::vector<std::uint8_t> b = ipf::readFile(path);
    if (b.size() < 10 || b[0] != 0x93 || b[1] != 'N') ipf::detail::fail("npy: not a .npy file: " + path);
    std::size_t hlen, off;
    if (b[6] == 1) {
        hlen = b[8] | (b[9] << 8);
        off = 10;
    } else {
        hlen = b[8] | (b[9] << 8) | (b[10] << 16) | (static_cast<std::size_t>(b[11]) << 24);
        off = 12;
    }
    const std::string header(b.begin() + static_cast<std::ptrdiff_t>(off),
                             b.begin() + static_cast<std::ptrdiff_t>(off + hlen));
    Array a;
    const std::size_t d = header.find("'descr'");
    const std::size_t q1 = header.find('\'', header.find(':', d) + 1);
    a.descr = header.substr(q1 + 1, header.find('\'', q1 + 1) - q1 - 1);
    if (header.find("'fortran_order': True") != std::string::npos) ipf::detail::fail("npy: fortran order unsupported");
    const std::size_t s0 = header.find('(', header.find("'shape'")), s1 = header.find(')', s0);
    std::stringstream ss(header.substr(s0 + 1, s1 - s0 - 1));
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (tok.find_first_not_of(' ') != std::string::npos) a.shape.push_back(std::stoll(tok));
    a.data.assign(b.begin() + static_cast<std::ptrdiff_t>(off + hlen), b.end());
    return a;
}

/// Loads a (h, w) or (h, w, c) array, converting the element type to T.
template <class T>
ipf::Image<T> load(const std::string& path) {
    const Array a = loadRaw(path);
    if (a.shape.empty() || a.shape.size() > 3) ipf::detail::fail("npy: expected a 1D, 2D or 3D array");
    const bool oneD = a.shape.size() == 1;  // 1D arrays load as a single row
    const int h = oneD ? 1 : static_cast<int>(a.shape[0]), w = static_cast<int>(a.shape[oneD ? 0 : 1]);
    const int c = a.shape.size() == 3 ? static_cast<int>(a.shape[2]) : 1;
    ipf::Image<T> img(w, h, c);
    const std::size_t n = img.sampleCount();
    auto conv = [&](auto tag) {
        using S = decltype(tag);
        if (a.data.size() < n * sizeof(S)) ipf::detail::fail("npy: truncated data");
        for (std::size_t i = 0; i < n; ++i) {
            S v;
            std::memcpy(&v, a.data.data() + i * sizeof(S), sizeof(S));
            img.data()[i] = static_cast<T>(v);
        }
    };
    if (a.descr == "|u1") conv(std::uint8_t{});
    else if (a.descr == "<u2") conv(std::uint16_t{});
    else if (a.descr == "<i2") conv(std::int16_t{});
    else if (a.descr == "<i4") conv(std::int32_t{});
    else if (a.descr == "<f4") conv(float{});
    else if (a.descr == "<f8") conv(double{});
    else if (a.descr == "|b1") conv(std::uint8_t{});
    else ipf::detail::fail("npy: unsupported dtype " + a.descr);
    return img;
}

}  // namespace npy
