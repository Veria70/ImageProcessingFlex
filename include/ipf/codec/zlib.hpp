// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf codec: self-contained zlib/DEFLATE (RFC 1950/1951) decompressor and compressor.
// Used by the PNG codec; also usable on its own via ipf::zlibCompress / ipf::zlibDecompress.
#pragma once

#include <queue>

#include "common.hpp"

namespace ipf {

namespace detail {
namespace zl {

inline constexpr std::uint16_t kLenBase[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                               31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
inline constexpr std::uint8_t kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                               2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
inline constexpr std::uint16_t kDistBase[30] = {1,   2,   3,   4,   5,   7,    9,    13,   17,   25,
                                                33,  49,  65,  97,  129, 193,  257,  385,  513,  769,
                                                1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
inline constexpr std::uint8_t kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                                                6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
inline constexpr std::uint8_t kClOrder[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

inline std::uint32_t reverseBits(std::uint32_t v, int n) {
    std::uint32_t r = 0;
    for (int i = 0; i < n; ++i, v >>= 1) r = (r << 1) | (v & 1u);
    return r;
}

/// Canonical Huffman codes from code lengths (RFC 1951 section 3.2.2).
inline std::vector<std::uint32_t> canonicalCodes(const std::uint8_t* lens, int n) {
    int count[16] = {0};
    for (int i = 0; i < n; ++i) ++count[lens[i]];
    count[0] = 0;
    std::uint32_t next[16] = {0};
    std::uint32_t code = 0;
    for (int b = 1; b < 16; ++b) {
        code = (code + static_cast<std::uint32_t>(count[b - 1])) << 1;
        next[b] = code;
    }
    std::vector<std::uint32_t> codes(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i)
        if (lens[i]) codes[i] = next[lens[i]]++;
    return codes;
}

/// Single-level lookup table decoder indexed by the next maxLen (bit-reversed) bits.
struct HuffDecoder {
    std::vector<std::uint32_t> table;  // (symbol << 4) | length, 0 = invalid code
    int maxLen = 0;

    void build(const std::uint8_t* lens, int n) {
        int count[16] = {0};
        maxLen = 0;
        for (int i = 0; i < n; ++i) {
            if (lens[i] > 15) fail("zlib: invalid code length");
            ++count[lens[i]];
            if (lens[i] > maxLen) maxLen = lens[i];
        }
        count[0] = 0;
        int left = 1;
        for (int b = 1; b < 16; ++b) {
            left <<= 1;
            left -= count[b];
            if (left < 0) fail("zlib: over-subscribed Huffman code");
        }
        table.assign(maxLen ? (std::size_t(1) << maxLen) : 0, 0);
        const std::vector<std::uint32_t> codes = canonicalCodes(lens, n);
        for (int s = 0; s < n; ++s) {
            const int len = lens[s];
            if (!len) continue;
            const std::uint32_t rev = reverseBits(codes[s], len);
            for (std::uint32_t k = rev; k < table.size(); k += (1u << len))
                table[k] = (static_cast<std::uint32_t>(s) << 4) | static_cast<std::uint32_t>(len);
        }
    }
};

class BitIn {
public:
    BitIn(const std::uint8_t* p, std::size_t n) : p_(p), n_(n) {}
    std::uint32_t peek(int k) {
        if (cnt_ < k) refill();
        return static_cast<std::uint32_t>(buf_ & ((1ULL << k) - 1));
    }
    void consume(int k) {
        buf_ >>= k;
        cnt_ -= k;
        if (pos_ * 8 - static_cast<std::size_t>(cnt_) > n_ * 8) fail("zlib: unexpected end of compressed data");
    }
    std::uint32_t bits(int k) {
        if (k == 0) return 0;
        const std::uint32_t v = peek(k);
        consume(k);
        return v;
    }
    void alignByte() { consume(cnt_ & 7); }
    int decode(const HuffDecoder& h) {
        if (h.maxLen == 0) fail("zlib: use of an empty Huffman table");
        const std::uint32_t e = h.table[peek(h.maxLen)];
        if (!e) fail("zlib: invalid Huffman code");
        consume(static_cast<int>(e & 15u));
        return static_cast<int>(e >> 4);
    }
    /// Copies `len` raw bytes (stream must be byte aligned).
    void copyBytes(std::vector<std::uint8_t>& out, std::size_t len) {
        while (len && cnt_ >= 8) {
            out.push_back(static_cast<std::uint8_t>(bits(8)));
            --len;
        }
        if (!len) return;
        if (pos_ > n_ || len > n_ - pos_) fail("zlib: stored block exceeds input");
        out.insert(out.end(), p_ + pos_, p_ + pos_ + len);
        pos_ += len;
    }
    /// Number of whole bytes consumed so far (after alignByte()).
    std::size_t bytesConsumed() const { return pos_ - static_cast<std::size_t>(cnt_) / 8; }

private:
    void refill() {
        while (cnt_ <= 56) {
            const std::uint64_t b = pos_ < n_ ? p_[pos_] : 0;
            ++pos_;
            buf_ |= b << cnt_;
            cnt_ += 8;
        }
    }
    const std::uint8_t* p_;
    std::size_t n_;
    std::size_t pos_ = 0;
    std::uint64_t buf_ = 0;
    int cnt_ = 0;
};

class BitOut {
public:
    explicit BitOut(std::vector<std::uint8_t>& o) : o_(o) {}
    void put(std::uint32_t v, int n) {
        buf_ |= static_cast<std::uint64_t>(v) << cnt_;
        cnt_ += n;
        while (cnt_ >= 8) {
            o_.push_back(static_cast<std::uint8_t>(buf_));
            buf_ >>= 8;
            cnt_ -= 8;
        }
    }
    void putCode(std::uint32_t code, int len) { put(reverseBits(code, len), len); }
    void flushByte() {
        if (cnt_ > 0) put(0, 8 - cnt_);
    }

private:
    std::vector<std::uint8_t>& o_;
    std::uint64_t buf_ = 0;
    int cnt_ = 0;
};

/// Builds length-limited Huffman code lengths from symbol frequencies.
inline std::vector<std::uint8_t> buildLengths(std::vector<std::uint32_t> freq, int maxBits) {
    const int n = static_cast<int>(freq.size());
    std::vector<std::uint8_t> lens(static_cast<std::size_t>(n), 0);
    std::vector<int> used;
    for (int i = 0; i < n; ++i)
        if (freq[i]) used.push_back(i);
    if (used.empty()) return lens;
    if (used.size() == 1) {  // a single symbol still needs a complete 1-bit code
        lens[used[0]] = 1;
        lens[used[0] == 0 ? 1 : 0] = 1;
        return lens;
    }
    for (;;) {
        struct Node {
            std::uint64_t w;
            int left, right;
        };
        std::vector<Node> nodes;
        using Item = std::pair<std::uint64_t, int>;
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
        for (int s : used) {
            nodes.push_back({freq[s], -1, s});
            pq.push({freq[s], static_cast<int>(nodes.size()) - 1});
        }
        while (pq.size() > 1) {
            const Item a = pq.top(); pq.pop();
            const Item b = pq.top(); pq.pop();
            nodes.push_back({a.first + b.first, a.second, b.second});
            pq.push({a.first + b.first, static_cast<int>(nodes.size()) - 1});
        }
        // Depth-first walk to assign depths to leaves.
        int maxDepth = 0;
        std::vector<std::pair<int, int>> stack{{pq.top().second, 0}};
        while (!stack.empty()) {
            const auto [id, depth] = stack.back();
            stack.pop_back();
            const Node& nd = nodes[id];
            if (nd.left < 0) {
                lens[nd.right] = static_cast<std::uint8_t>(std::min(depth, 255));
                maxDepth = std::max(maxDepth, depth);
            } else {
                stack.push_back({nd.left, depth + 1});
                stack.push_back({nd.right, depth + 1});
            }
        }
        if (maxDepth <= maxBits) return lens;
        for (int s : used) freq[s] = (freq[s] >> 1) | 1u;  // flatten and retry
    }
}

inline int lengthSymbol(int len) {  // len in [3, 258]
    static const auto table = [] {
        std::vector<std::uint8_t> t(259, 0);
        for (int l = 3, s = 0; l <= 258; ++l) {
            while (s < 28 && kLenBase[s + 1] <= l) ++s;
            t[l] = static_cast<std::uint8_t>(s);
        }
        return t;
    }();
    return table[static_cast<std::size_t>(len)];
}
inline int distSymbol(int dist) {  // dist in [1, 32768]
    static const auto table = [] {
        std::vector<std::uint8_t> t(32769, 0);
        for (int d = 1, s = 0; d <= 32768; ++d) {
            while (s < 29 && kDistBase[s + 1] <= d) ++s;
            t[d] = static_cast<std::uint8_t>(s);
        }
        return t;
    }();
    return table[static_cast<std::size_t>(dist)];
}

struct Token {
    std::uint16_t lenOrLit;  // literal byte, or match length (3..258)
    std::uint16_t dist;      // 0 for literals
};

}  // namespace zl
}  // namespace detail

/// CRC-32 (IEEE 802.3, as used by PNG and gzip).
inline std::uint32_t crc32(const std::uint8_t* data, std::size_t n, std::uint32_t crc = 0) {
    static const auto table = [] {
        std::vector<std::uint32_t> t(256);
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (std::size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}

/// Adler-32 checksum (zlib).
inline std::uint32_t adler32(const std::uint8_t* data, std::size_t n, std::uint32_t adler = 1) {
    std::uint32_t a = adler & 0xFFFF, b = adler >> 16;
    while (n) {
        const std::size_t chunk = std::min<std::size_t>(n, 5552);
        for (std::size_t i = 0; i < chunk; ++i) {
            a += data[i];
            b += a;
        }
        a %= 65521;
        b %= 65521;
        data += chunk;
        n -= chunk;
    }
    return (b << 16) | a;
}

/// Decompresses a raw DEFLATE stream. `consumed` (optional) receives the number of input bytes used.
inline std::vector<std::uint8_t> inflate(const std::uint8_t* data, std::size_t n, std::size_t sizeHint = 0,
                                         std::size_t* consumed = nullptr) {
    using namespace detail::zl;
    std::vector<std::uint8_t> out;
    out.reserve(sizeHint ? sizeHint : n * 4);
    BitIn in(data, n);
    HuffDecoder lit, dist;
    bool last = false;
    while (!last) {
        last = in.bits(1) != 0;
        const std::uint32_t type = in.bits(2);
        if (type == 0) {
            in.alignByte();
            const std::uint32_t len = in.bits(16), nlen = in.bits(16);
            if ((len ^ 0xFFFFu) != nlen) detail::fail("zlib: corrupt stored block length");
            in.copyBytes(out, len);
            continue;
        }
        if (type == 1) {
            std::uint8_t l[288], d[30];
            for (int i = 0; i < 288; ++i) l[i] = static_cast<std::uint8_t>(i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8);
            for (int i = 0; i < 30; ++i) d[i] = 5;
            lit.build(l, 288);
            dist.build(d, 30);
        } else if (type == 2) {
            const int hlit = static_cast<int>(in.bits(5)) + 257;
            const int hdist = static_cast<int>(in.bits(5)) + 1;
            const int hclen = static_cast<int>(in.bits(4)) + 4;
            if (hlit > 286 || hdist > 30) detail::fail("zlib: invalid dynamic block header");
            std::uint8_t cl[19] = {0};
            for (int i = 0; i < hclen; ++i) cl[kClOrder[i]] = static_cast<std::uint8_t>(in.bits(3));
            HuffDecoder clDec;
            clDec.build(cl, 19);
            std::uint8_t lens[286 + 30] = {0};
            int i = 0;
            while (i < hlit + hdist) {
                const int sym = in.decode(clDec);
                if (sym < 16) {
                    lens[i++] = static_cast<std::uint8_t>(sym);
                    continue;
                }
                int rep = 0;
                std::uint8_t val = 0;
                if (sym == 16) {
                    if (i == 0) detail::fail("zlib: repeat with no previous length");
                    val = lens[i - 1];
                    rep = 3 + static_cast<int>(in.bits(2));
                } else if (sym == 17) {
                    rep = 3 + static_cast<int>(in.bits(3));
                } else {
                    rep = 11 + static_cast<int>(in.bits(7));
                }
                if (i + rep > hlit + hdist) detail::fail("zlib: code lengths overflow");
                while (rep--) lens[i++] = val;
            }
            if (lens[256] == 0) detail::fail("zlib: missing end-of-block code");
            lit.build(lens, hlit);
            dist.build(lens + hlit, hdist);
        } else {
            detail::fail("zlib: invalid block type");
        }
        for (;;) {
            int sym = in.decode(lit);
            if (sym < 256) {
                out.push_back(static_cast<std::uint8_t>(sym));
            } else if (sym == 256) {
                break;
            } else {
                sym -= 257;
                if (sym >= 29) detail::fail("zlib: invalid length symbol");
                const std::size_t len = kLenBase[sym] + in.bits(kLenExtra[sym]);
                const int ds = in.decode(dist);
                if (ds >= 30) detail::fail("zlib: invalid distance symbol");
                const std::size_t d = kDistBase[ds] + in.bits(kDistExtra[ds]);
                if (d > out.size()) detail::fail("zlib: distance too far back");
                std::size_t from = out.size() - d;
                for (std::size_t k = 0; k < len; ++k) out.push_back(out[from + k]);
            }
        }
    }
    in.alignByte();
    if (consumed) *consumed = in.bytesConsumed();
    return out;
}

/// Compresses data into a raw DEFLATE stream. level: 0 = store only, 1 = fastest .. 9 = best.
inline std::vector<std::uint8_t> deflate(const std::uint8_t* data, std::size_t n, int level = 6) {
    using namespace detail::zl;
    level = clamp(level, 0, 9);
    std::vector<std::uint8_t> out;
    BitOut bw(out);
    auto writeStored = [&](std::size_t start, std::size_t len, bool final) {
        do {
            const std::size_t chunk = std::min<std::size_t>(len, 65535);
            len -= chunk;
            bw.put(final && len == 0 ? 1u : 0u, 1);
            bw.put(0, 2);
            bw.flushByte();
            bw.put(static_cast<std::uint32_t>(chunk), 16);
            bw.put(static_cast<std::uint32_t>(chunk) ^ 0xFFFFu, 16);
            out.insert(out.end(), data + start, data + start + chunk);
            start += chunk;
        } while (len > 0);
    };
    if (level == 0) {
        writeStored(0, n, true);
        return out;
    }

    // zlib's configuration table: good_length, max_lazy, nice_length, max_chain per level.
    static const int kGood[10] = {0, 4, 4, 4, 4, 8, 8, 8, 32, 32};
    static const int kLazy[10] = {0, 4, 5, 6, 4, 16, 16, 32, 128, 258};
    static const int kNice[10] = {0, 8, 16, 32, 16, 32, 128, 128, 258, 258};
    static const int kChain[10] = {0, 4, 8, 32, 16, 32, 128, 256, 1024, 4096};
    const int maxChain = kChain[level], niceLen = kNice[level], goodLen = kGood[level], maxLazy = kLazy[level];
    const bool lazy = level >= 4;
    const std::size_t kWindow = 32768, kMask = kWindow - 1;
    const int kHashBits = 15;
    detail::require(n < 0x7FFFFFFFu, "deflate: input too large (max 2 GiB)");
    std::vector<std::int32_t> head(std::size_t(1) << kHashBits, -1);
    std::vector<std::int32_t> prev(kWindow, -1);
    auto hashAt = [&](std::size_t i) -> std::size_t {
        return ((static_cast<std::size_t>(data[i]) << 10) ^ (static_cast<std::size_t>(data[i + 1]) << 5) ^ data[i + 2]) &
               ((std::size_t(1) << kHashBits) - 1);
    };
    auto insert = [&](std::size_t i) {
        if (i + 2 >= n) return;
        const std::size_t h = hashAt(i);
        prev[i & kMask] = head[h];
        head[h] = static_cast<std::int32_t>(i);
    };
    auto findMatch = [&](std::size_t i, int& bestLen, int& bestDist, int chain) {
        bestLen = 0;
        bestDist = 0;
        if (i + 2 >= n) return;
        const std::size_t maxLen = std::min<std::size_t>(258, n - i);
        std::int32_t cand = head[hashAt(i)];
        while (cand >= 0 && chain-- > 0) {
            const std::size_t c = static_cast<std::size_t>(cand);
            const std::size_t d = i - c;
            if (d > kWindow - 1 || d == 0) break;
            if (data[c + static_cast<std::size_t>(bestLen)] == data[i + static_cast<std::size_t>(bestLen)]) {
                std::size_t l = 0;
                while (l < maxLen && data[c + l] == data[i + l]) ++l;
                if (static_cast<int>(l) > bestLen) {
                    bestLen = static_cast<int>(l);
                    bestDist = static_cast<int>(d);
                    if (l == maxLen || bestLen >= niceLen) break;  // good enough: stop searching
                }
            }
            const std::int32_t nxt = prev[c & kMask];
            if (nxt >= cand) break;  // stale chain entry
            cand = nxt;
        }
        if (bestLen < 3) bestLen = 0;
    };

    const std::size_t kBlockTokens = 1 << 15;
    std::vector<Token> tokens;
    tokens.reserve(kBlockTokens);
    std::size_t blockStart = 0;

    auto flushBlock = [&](std::size_t blockEnd, bool final) {
        std::vector<std::uint32_t> lf(286, 0), df(30, 0);
        for (const Token& t : tokens) {
            if (t.dist == 0) {
                ++lf[t.lenOrLit];
            } else {
                ++lf[257 + lengthSymbol(t.lenOrLit)];
                ++df[distSymbol(t.dist)];
            }
        }
        lf[256] = 1;
        std::vector<std::uint8_t> ll = buildLengths(lf, 15), dl = buildLengths(df, 15);
        if (std::all_of(dl.begin(), dl.end(), [](std::uint8_t v) { return v == 0; })) dl[0] = dl[1] = 1;
        int hlit = 286, hdist = 30;
        while (hlit > 257 && ll[hlit - 1] == 0) --hlit;
        while (hdist > 1 && dl[hdist - 1] == 0) --hdist;
        // Run-length encode the code lengths.
        std::vector<std::uint8_t> all(ll.begin(), ll.begin() + hlit);
        all.insert(all.end(), dl.begin(), dl.begin() + hdist);
        struct ClSym {
            int sym, extra, nbits;
        };
        std::vector<ClSym> rle;
        for (std::size_t i = 0; i < all.size();) {
            std::size_t run = 1;
            while (i + run < all.size() && all[i + run] == all[i]) ++run;
            if (all[i] == 0 && run >= 3) {
                const std::size_t r = std::min<std::size_t>(run, 138);
                if (r >= 11) rle.push_back({18, static_cast<int>(r - 11), 7});
                else rle.push_back({17, static_cast<int>(r - 3), 3});
                i += r;
            } else if (all[i] != 0 && run >= 4) {
                rle.push_back({all[i], 0, 0});
                std::size_t r = std::min<std::size_t>(run - 1, 6);
                rle.push_back({16, static_cast<int>(r - 3), 2});
                i += r + 1;
            } else {
                rle.push_back({all[i], 0, 0});
                ++i;
            }
        }
        std::vector<std::uint32_t> cf(19, 0);
        for (const ClSym& s : rle) ++cf[s.sym];
        const std::vector<std::uint8_t> cl = buildLengths(cf, 7);
        int hclen = 19;
        while (hclen > 4 && cl[kClOrder[hclen - 1]] == 0) --hclen;

        // Estimate the dynamic block size and fall back to a stored block if larger.
        std::uint64_t bitsDyn = 3 + 14 + 3ULL * hclen;
        for (const ClSym& s : rle) bitsDyn += cl[s.sym] + s.nbits;
        for (int s = 0; s < 286; ++s) bitsDyn += static_cast<std::uint64_t>(lf[s]) * ll[s];
        for (int s = 0; s < 29; ++s) bitsDyn += static_cast<std::uint64_t>(lf[257 + s]) * kLenExtra[s];
        for (int s = 0; s < 30; ++s) bitsDyn += static_cast<std::uint64_t>(df[s]) * (dl[s] + kDistExtra[s]);
        // Fixed-Huffman block cost (RFC 1951 section 3.2.6).
        std::vector<std::uint8_t> fl(288), fd(30, 5);
        for (int s = 0; s < 288; ++s) fl[s] = static_cast<std::uint8_t>(s < 144 ? 8 : s < 256 ? 9 : s < 280 ? 7 : 8);
        std::uint64_t bitsFixed = 3;
        for (int s = 0; s < 286; ++s) bitsFixed += static_cast<std::uint64_t>(lf[s]) * fl[s];
        for (int s = 0; s < 29; ++s) bitsFixed += static_cast<std::uint64_t>(lf[257 + s]) * kLenExtra[s];
        for (int s = 0; s < 30; ++s) bitsFixed += static_cast<std::uint64_t>(df[s]) * (5 + kDistExtra[s]);
        const std::size_t rawLen = blockEnd - blockStart;
        const std::uint64_t bitsStored = (rawLen + 5 * (rawLen / 65535 + 1)) * 8ULL + 8;
        if (bitsStored < bitsDyn && bitsStored < bitsFixed) {
            writeStored(blockStart, rawLen, final);
        } else {
            const bool fixed = bitsFixed <= bitsDyn;
            if (fixed) {
                ll = fl;
                dl = fd;
            }
            std::vector<std::uint32_t> lc = canonicalCodes(ll.data(), static_cast<int>(ll.size())),
                                       dc = canonicalCodes(dl.data(), 30), cc = canonicalCodes(cl.data(), 19);
            // Codes are sent MSB-first into an LSB-first stream: reverse them once per block.
            for (std::size_t k = 0; k < lc.size(); ++k) lc[k] = reverseBits(lc[k], ll[k]);
            for (std::size_t k = 0; k < dc.size(); ++k) dc[k] = reverseBits(dc[k], dl[k]);
            for (std::size_t k = 0; k < cc.size(); ++k) cc[k] = reverseBits(cc[k], cl[k]);
            bw.put(final ? 1u : 0u, 1);
            if (fixed) {
                bw.put(1, 2);
            } else {
                bw.put(2, 2);
                bw.put(static_cast<std::uint32_t>(hlit - 257), 5);
                bw.put(static_cast<std::uint32_t>(hdist - 1), 5);
                bw.put(static_cast<std::uint32_t>(hclen - 4), 4);
                for (int i = 0; i < hclen; ++i) bw.put(cl[kClOrder[i]], 3);
                for (const ClSym& s : rle) {
                    bw.put(cc[s.sym], cl[s.sym]);
                    if (s.nbits) bw.put(static_cast<std::uint32_t>(s.extra), s.nbits);
                }
            }
            for (const Token& t : tokens) {
                if (t.dist == 0) {
                    bw.put(lc[t.lenOrLit], ll[t.lenOrLit]);
                } else {
                    const int ls = lengthSymbol(t.lenOrLit);
                    bw.put(lc[257 + ls], ll[257 + ls]);
                    bw.put(static_cast<std::uint32_t>(t.lenOrLit - kLenBase[ls]), kLenExtra[ls]);
                    const int ds = distSymbol(t.dist);
                    bw.put(dc[ds], dl[ds]);
                    bw.put(static_cast<std::uint32_t>(t.dist - kDistBase[ds]), kDistExtra[ds]);
                }
            }
            bw.put(lc[256], ll[256]);
        }
        tokens.clear();
        blockStart = blockEnd;
    };

    std::size_t i = 0;
    while (i < n) {
        int len, dst;
        findMatch(i, len, dst, maxChain);
        if (len && lazy && len < maxLazy && i + 1 < n) {
            insert(i);
            int len2, dst2;
            findMatch(i + 1, len2, dst2, len >= goodLen ? maxChain / 4 : maxChain);
            if (len2 > len) {
                tokens.push_back({data[i], 0});
                ++i;
                if (tokens.size() >= kBlockTokens) flushBlock(i, false);
                continue;  // i was already inserted
            }
            tokens.push_back({static_cast<std::uint16_t>(len), static_cast<std::uint16_t>(dst)});
            for (std::size_t k = 1; k < static_cast<std::size_t>(len); ++k) insert(i + k);
            i += static_cast<std::size_t>(len);
        } else if (len) {
            tokens.push_back({static_cast<std::uint16_t>(len), static_cast<std::uint16_t>(dst)});
            for (std::size_t k = 0; k < static_cast<std::size_t>(len); ++k) insert(i + k);
            i += static_cast<std::size_t>(len);
        } else {
            tokens.push_back({data[i], 0});
            insert(i);
            ++i;
        }
        if (tokens.size() >= kBlockTokens) flushBlock(i, false);
    }
    flushBlock(n, true);
    bw.flushByte();
    return out;
}

/// Compresses data into a zlib stream (2-byte header + DEFLATE + Adler-32).
inline std::vector<std::uint8_t> zlibCompress(const std::uint8_t* data, std::size_t n, int level = 6) {
    std::vector<std::uint8_t> out{0x78, static_cast<std::uint8_t>(level == 0 ? 0x01 : 0x9C)};
    const std::vector<std::uint8_t> body = deflate(data, n, level);
    out.insert(out.end(), body.begin(), body.end());
    detail::put32be(out, adler32(data, n));
    return out;
}

/// Decompresses a zlib stream, verifying the header and Adler-32 checksum.
inline std::vector<std::uint8_t> zlibDecompress(const std::uint8_t* data, std::size_t n, std::size_t sizeHint = 0) {
    if (n < 2) detail::fail("zlib: stream too short");
    const unsigned cmf = data[0], flg = data[1];
    if ((cmf & 0x0F) != 8 || (cmf >> 4) > 7) detail::fail("zlib: unsupported compression method");
    if ((cmf * 256 + flg) % 31 != 0) detail::fail("zlib: corrupt header");
    if (flg & 0x20) detail::fail("zlib: preset dictionaries are not supported");
    std::size_t used = 0;
    std::vector<std::uint8_t> out = inflate(data + 2, n - 2, sizeHint, &used);
    const std::size_t trailer = 2 + used;
    if (trailer + 4 <= n) {
        const std::uint32_t expect = (static_cast<std::uint32_t>(data[trailer]) << 24) |
                                     (static_cast<std::uint32_t>(data[trailer + 1]) << 16) |
                                     (static_cast<std::uint32_t>(data[trailer + 2]) << 8) | data[trailer + 3];
        if (expect != adler32(out.data(), out.size())) detail::fail("zlib: Adler-32 checksum mismatch");
    }
    return out;
}

}  // namespace ipf
