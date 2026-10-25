// Test runner: runs every registered test, or only those whose name contains argv[1].
#include <cstring>

#include "test_framework.hpp"

int main(int argc, char** argv) {
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int run = 0, failedTests = 0;
    for (const auto& t : ipftest::registry()) {
        if (filter && !std::strstr(t.name, filter)) continue;
        const int before = ipftest::state().failures;
        try {
            t.fn();
        } catch (const std::exception& e) {
            ipftest::reportFailure(t.name, 0, std::string("unexpected exception: ") + e.what());
        }
        ++run;
        const bool ok = ipftest::state().failures == before;
        if (!ok) ++failedTests;
        std::printf("[%s] %s\n", ok ? " OK " : "FAIL", t.name);
    }
    std::printf("\n%d tests, %d checks, %d failed tests, %d failed checks\n", run, ipftest::state().checks,
                failedTests, ipftest::state().failures);
    return failedTests == 0 && run > 0 ? 0 : 1;
}
