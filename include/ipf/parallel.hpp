// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniel. Part of ImageProcessingFlex (ipf); see LICENSE.
// ipf parallel: a tiny, dependency-free parallel-for used by the heavy algorithms.
//
//   ipf::setNumThreads(4);   // 0 (default) = all hardware threads, 1 = run serially
//
// Work is split into contiguous index ranges (typically image rows); every range is
// processed exactly as in the serial code, so results do not depend on the thread count.
// Nested parallel regions run serially. Define IPF_NO_THREADS to compile without threads.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <vector>

#ifndef IPF_NO_THREADS
#include <system_error>
#include <thread>
#endif

namespace ipf {

namespace detail {
inline std::atomic<int>& threadSetting() {
    static std::atomic<int> n{0};
    return n;
}
inline bool& inParallelRegion() {
    thread_local bool inside = false;
    return inside;
}
}  // namespace detail

/// Sets the number of worker threads used by ipf (0 = hardware concurrency, 1 = serial).
inline void setNumThreads(int n) { detail::threadSetting() = n < 0 ? 0 : n; }

/// Number of threads ipf will use for large operations.
inline int getNumThreads() {
#ifdef IPF_NO_THREADS
    return 1;
#else
    const int n = detail::threadSetting();
    if (n > 0) return n;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw ? static_cast<int>(hw) : 1;
#endif
}

/// Calls fn(i0, i1) on contiguous sub-ranges covering [begin, end), in parallel when the
/// estimated amount of work (count * costPerItem, in "pixel operations") is large enough
/// to pay for the threads. Exceptions thrown by fn are rethrown in the calling thread.
template <class Fn>
void parallelFor(int begin, int end, Fn&& fn, double costPerItem = 1.0) {
    const int n = end - begin;
    if (n <= 0) return;
    int threads = getNumThreads();
    const double kMinWorkPerThread = 32768.0;
    threads = std::min({threads, n, std::max(1, static_cast<int>(n * costPerItem / kMinWorkPerThread))});
    if (threads <= 1 || detail::inParallelRegion()) {
        fn(begin, end);
        return;
    }
#ifdef IPF_NO_THREADS
    fn(begin, end);
#else
    std::exception_ptr error;
    std::mutex errorMutex;
    auto run = [&](int a, int b) {
        detail::inParallelRegion() = true;
        try {
            fn(a, b);
        } catch (...) {
            std::lock_guard<std::mutex> lock(errorMutex);
            if (!error) error = std::current_exception();
        }
        detail::inParallelRegion() = false;
    };
    auto bound = [&](int k) { return begin + static_cast<int>(static_cast<long long>(n) * k / threads); };
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads - 1));
    for (int k = 1; k < threads; ++k) {
        try {
            workers.emplace_back(run, bound(k), bound(k + 1));
        } catch (const std::system_error&) {  // could not start a thread: do the chunk here
            run(bound(k), bound(k + 1));
        }
    }
    run(bound(0), bound(1));
    for (std::thread& t : workers) t.join();
    if (error) std::rethrow_exception(error);
#endif
}

/// parallelFor over a flat index range [0, n) of any size (e.g. all pixels of an image):
/// calls fn(i0, i1) with std::size_t bounds; costPerItem as for parallelFor.
template <class Fn>
void parallelForN(std::size_t n, Fn&& fn, double costPerItem = 1.0) {
    const std::size_t kChunk = 4096;
    const std::size_t chunks = (n + kChunk - 1) / kChunk;
    if (chunks == 0) return;
    parallelFor(0, static_cast<int>(chunks), [&](int c0, int c1) {
        fn(static_cast<std::size_t>(c0) * kChunk, std::min(n, static_cast<std::size_t>(c1) * kChunk));
    }, costPerItem * static_cast<double>(kChunk));
}

}  // namespace ipf
