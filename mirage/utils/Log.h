#pragma once

// Structured logging for Mirage, wrapping spdlog (see mirage/CMakeLists.txt's
// FetchContent pin) behind thin MIRAGE_LOG_* macros so call sites stay simple
// and the backend stays swappable without touching every call site again.
// Resolves docs/PRODUCTION_READINESS.md's Tier 3 "no logging framework, no
// crash reporting" finding (raw fprintf(stderr)/printf/std::cout scattered
// across VulkanRenderer.cpp, TextureLoader.cpp, and both CLI tools, with no
// severity levels or structured output a farm dispatcher could parse).
//
// Mirage::Logging::Init() should be called once, early in each executable's
// main(), before InstallCrashHandler() (mirage/utils/CrashHandler.h) and
// before any MIRAGE_LOG_* call whose ordering relative to Init() matters -
// spdlog's built-in default logger works fine even if Init() is never
// called, just without Mirage's preferred pattern/level/JSON-lines sink.

#include <spdlog/spdlog.h>

namespace Mirage::Logging
{
    enum class Level
    {
        eTrace = spdlog::level::trace,
        eDebug = spdlog::level::debug,
        eInfo = spdlog::level::info,
        eWarn = spdlog::level::warn,
        eError = spdlog::level::err,
        eCritical = spdlog::level::critical,
    };

    // Configures Mirage's process-wide default logger: a color stderr sink at
    // `level`, plus - if jsonLinesPath is non-null - a second sink writing
    // structured JSON-lines records to that path (one JSON object per line:
    // timestamp/level/logger/message), the format a farm dispatcher needs to
    // parse render-job logs reliably. jsonLinesPath's parent directory must
    // already exist; this does not create it.
    //
    // Safe to call more than once (e.g. a CLI tool re-parsing --log-level
    // after an initial default-level Init() at startup) - each call replaces
    // the previously configured default logger.
    void Init(Level level = Level::eInfo, const char *jsonLinesPath = nullptr);

    // Parses a --log-level style string ("trace"/"debug"/"info"/"warn"/
    // "error"/"critical", case-insensitive). Returns false (and leaves *out
    // untouched) on an unrecognized string, so callers can report a usage
    // error instead of silently substituting a default.
    bool ParseLevel(const char *text, Level *out);
} // namespace Mirage::Logging

// SPDLOG_* (rather than calling spdlog::info(...) etc. directly) preserves
// source file/line info in the log record and is compiled out entirely below
// SPDLOG_ACTIVE_LEVEL. Every target that uses these macros defines
// SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE (see mirage/CMakeLists.txt and both
// CLI tools' CMakeLists.txt) specifically so that compile-time gating never
// shadows Init()/ParseLevel()'s *runtime* level control below - without that
// definition, spdlog's default active level (info) would silently compile
// MIRAGE_LOG_TRACE/DEBUG into no-ops regardless of what --log-level or
// Init() request at runtime, which defeats the point of exposing a runtime
// level at all.
#define MIRAGE_LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define MIRAGE_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define MIRAGE_LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define MIRAGE_LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define MIRAGE_LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define MIRAGE_LOG_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)
