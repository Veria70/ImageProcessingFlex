#!/usr/bin/env bash
# Build with MSVC (via build.bat) and run the unit tests (Windows / Git Bash).
# Usage: tools/build.sh [Debug|Release] [17|20] [asan|""] [test-name-filter]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-Release}"
STD="${2:-17}"
EXTRA="${3:-}"
SUFFIX=""
[ "$EXTRA" = "asan" ] && SUFFIX="-asan"
cmd //c "$(cygpath -w "$ROOT/tools/build.bat")" "$CONFIG" "$STD" "$EXTRA"
cd "$ROOT/build/$CONFIG-cpp$STD$SUFFIX/tests" && PATH="$PATH:/c/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64" ./ipf_tests.exe "${@:4}"
