#include <doctest/doctest.h>

#include "mirage/utils/Log.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace Mirage;

TEST_SUITE("Logging")
{
    TEST_CASE("ParseLevel accepts every documented spelling, case-insensitively")
    {
        Logging::Level level;

        CHECK(Logging::ParseLevel("trace", &level));
        CHECK(level == Logging::Level::eTrace);

        CHECK(Logging::ParseLevel("DEBUG", &level));
        CHECK(level == Logging::Level::eDebug);

        CHECK(Logging::ParseLevel("Info", &level));
        CHECK(level == Logging::Level::eInfo);

        CHECK(Logging::ParseLevel("warn", &level));
        CHECK(level == Logging::Level::eWarn);
        CHECK(Logging::ParseLevel("warning", &level));
        CHECK(level == Logging::Level::eWarn);

        CHECK(Logging::ParseLevel("error", &level));
        CHECK(level == Logging::Level::eError);
        CHECK(Logging::ParseLevel("err", &level));
        CHECK(level == Logging::Level::eError);

        CHECK(Logging::ParseLevel("critical", &level));
        CHECK(level == Logging::Level::eCritical);
        CHECK(Logging::ParseLevel("fatal", &level));
        CHECK(level == Logging::Level::eCritical);
    }

    TEST_CASE("ParseLevel rejects unrecognized strings and leaves *out untouched")
    {
        Logging::Level level = Logging::Level::eWarn;
        CHECK_FALSE(Logging::ParseLevel("bogus", &level));
        // Untouched on failure - callers rely on this to fall back to
        // whatever default they already had.
        CHECK(level == Logging::Level::eWarn);

        CHECK_FALSE(Logging::ParseLevel(nullptr, &level));
        CHECK_FALSE(Logging::ParseLevel("info", nullptr));
    }

    // Exercises the JSON-lines sink end to end (mirage/utils/Log.cpp's
    // JsonLinesFormatter) - nothing else in the codebase writes to it yet,
    // so without this test a formatter bug (malformed JSON, wrong field
    // names, an escaping error) would only surface once a farm dispatcher
    // tried to parse real output.
    TEST_CASE("Init with jsonLinesPath writes valid-looking JSON-lines records")
    {
        auto path = std::filesystem::temp_directory_path() / "mirage_log_test.jsonl";
        std::filesystem::remove(path);

        Logging::Init(Logging::Level::eInfo, path.string().c_str());
        MIRAGE_LOG_INFO("hello \"world\" with a backslash \\ and a newline\nin it");
        spdlog::default_logger()->flush();

        std::ifstream in(path);
        REQUIRE(in.good());
        std::string line;
        REQUIRE(std::getline(in, line));

        // Not a full JSON parse (no JSON library in this repo, and adding
        // one just for this test isn't worth it) - just the structural
        // properties a farm dispatcher's line-based JSON parser would
        // actually rely on.
        CHECK(line.front() == '{');
        CHECK(line.back() == '}');
        CHECK(line.find("\"timestamp\":\"") != std::string::npos);
        CHECK(line.find("\"level\":\"info\"") != std::string::npos);
        CHECK(line.find("\"logger\":\"mirage\"") != std::string::npos);
        // The embedded quote/backslash/newline must have been escaped, not
        // left raw (which would break any line-based JSON parser downstream).
        CHECK(line.find("\\\"world\\\"") != std::string::npos);
        CHECK(line.find("\\\\") != std::string::npos);
        CHECK(line.find("\\n") != std::string::npos);
        CHECK(line.find('\n') == std::string::npos); // no raw newline mid-record

        std::filesystem::remove(path);

        // Restore a plain default logger so later tests in this binary
        // aren't left pointed at a temp file removed above.
        Logging::Init();
    }
}
