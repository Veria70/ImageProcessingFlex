// Minimal self-contained unit test framework for the ipf test-suite.
#pragma once

#include <cmath>
#include <cstdio>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace ipftest {

struct TestCase {
    const char* name;
    void (*fn)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

struct State {
    int checks = 0;
    int failures = 0;
};
inline State& state() {
    static State s;
    return s;
}

inline void reportFailure(const char* file, int line, const std::string& what) {
    ++state().failures;
    std::printf("    FAILED %s:%d: %s\n", file, line, what.c_str());
}

template <class A, class B>
std::string describe(const A& a, const B& b) {
    std::ostringstream os;
    os << "[" << +a << "] vs [" << +b << "]";
    return os.str();
}

}  // namespace ipftest

#define IPF_TEST_CONCAT2(a, b) a##b
#define IPF_TEST_CONCAT(a, b) IPF_TEST_CONCAT2(a, b)

#define TEST(name)                                                                   \
    static void name();                                                              \
    static ::ipftest::Registrar IPF_TEST_CONCAT(ipf_registrar_, name)(#name, &name); \
    static void name()

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        ++::ipftest::state().checks;                                                      \
        if (!(cond)) ::ipftest::reportFailure(__FILE__, __LINE__, "CHECK(" #cond ")");    \
    } while (0)

#define CHECK_EQ(a, b)                                                                                   \
    do {                                                                                                 \
        ++::ipftest::state().checks;                                                                     \
        const auto ipf_a_ = (a);                                                                         \
        const auto ipf_b_ = (b);                                                                         \
        if (!(ipf_a_ == ipf_b_))                                                                         \
            ::ipftest::reportFailure(__FILE__, __LINE__,                                                 \
                                     "CHECK_EQ(" #a ", " #b ") " + ::ipftest::describe(ipf_a_, ipf_b_)); \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                        \
    do {                                                                                             \
        ++::ipftest::state().checks;                                                                 \
        const double ipf_a_ = static_cast<double>(a);                                                \
        const double ipf_b_ = static_cast<double>(b);                                                \
        if (!(std::fabs(ipf_a_ - ipf_b_) <= (tol)))                                                  \
            ::ipftest::reportFailure(__FILE__, __LINE__, "CHECK_NEAR(" #a ", " #b ", " #tol ") " +   \
                                                             ::ipftest::describe(ipf_a_, ipf_b_));   \
    } while (0)

#define CHECK_THROWS(expr)                                                                      \
    do {                                                                                        \
        ++::ipftest::state().checks;                                                            \
        bool ipf_threw_ = false;                                                                \
        try {                                                                                   \
            (void)(expr);                                                                       \
        } catch (const std::exception&) {                                                       \
            ipf_threw_ = true;                                                                  \
        }                                                                                       \
        if (!ipf_threw_) ::ipftest::reportFailure(__FILE__, __LINE__, "CHECK_THROWS(" #expr ")"); \
    } while (0)
