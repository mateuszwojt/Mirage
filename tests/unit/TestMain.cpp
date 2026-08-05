// Single translation unit that implements doctest's main() for
// mirage_unit_tests. Every other unit/*.cpp file just #includes
// <doctest/doctest.h> and defines TEST_CASE()s.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
