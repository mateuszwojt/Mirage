#include "mirage/utils/Log.h"

#include <spdlog/fmt/fmt.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <cctype>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace Mirage::Logging
{
    namespace
    {
        void AppendRaw(spdlog::memory_buf_t &dest, spdlog::string_view_t text)
        {
            for (char c : text)
                dest.push_back(c);
        }

        void AppendJsonEscaped(spdlog::memory_buf_t &dest, spdlog::string_view_t text)
        {
            for (char c : text)
            {
                switch (c)
                {
                case '"':
                    AppendRaw(dest, "\\\"");
                    break;
                case '\\':
                    AppendRaw(dest, "\\\\");
                    break;
                case '\n':
                    AppendRaw(dest, "\\n");
                    break;
                case '\r':
                    AppendRaw(dest, "\\r");
                    break;
                case '\t':
                    AppendRaw(dest, "\\t");
                    break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20)
                    {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                        AppendRaw(dest, buf);
                    }
                    else
                    {
                        dest.push_back(c);
                    }
                    break;
                }
            }
        }

        // Minimal, dependency-free JSON-lines formatter: one JSON object per
        // log record (timestamp/level/logger/message), no pretty-printing, no
        // extra structured fields beyond what spdlog's log_msg already
        // carries. This is intentionally small - see Log.h's comment on why
        // this exists at all (a farm dispatcher needs machine-parseable
        // output, which the plain-text console sink's pattern can't
        // guarantee once messages contain arbitrary characters).
        class JsonLinesFormatter : public spdlog::formatter
        {
        public:
            void format(const spdlog::details::log_msg &msg, spdlog::memory_buf_t &dest) override
            {
                using namespace std::chrono;
                auto micros = duration_cast<microseconds>(msg.time.time_since_epoch()) % 1000000;
                std::time_t t = system_clock::to_time_t(msg.time);
                std::tm tmBuf{};
#if defined(_WIN32)
                gmtime_s(&tmBuf, &t);
#else
                gmtime_r(&t, &tmBuf);
#endif
                char timeBuf[32];
                std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%dT%H:%M:%S", &tmBuf);

                AppendRaw(dest, R"({"timestamp":")");
                AppendRaw(dest, timeBuf);
                char usBuf[16];
                std::snprintf(usBuf, sizeof(usBuf), ".%06lldZ", static_cast<long long>(micros.count()));
                AppendRaw(dest, usBuf);
                AppendRaw(dest, R"(","level":")");
                AppendRaw(dest, spdlog::level::to_string_view(msg.level));
                AppendRaw(dest, R"(","logger":")");
                AppendJsonEscaped(dest, msg.logger_name);
                AppendRaw(dest, R"(","message":")");
                AppendJsonEscaped(dest, spdlog::string_view_t(msg.payload.data(), msg.payload.size()));
                AppendRaw(dest, "\"}\n");
            }

            std::unique_ptr<spdlog::formatter> clone() const override
            {
                return std::make_unique<JsonLinesFormatter>();
            }
        };

        spdlog::level::level_enum ToSpdlogLevel(Level level)
        {
            return static_cast<spdlog::level::level_enum>(level);
        }
    } // namespace

    void Init(Level level, const char *jsonLinesPath)
    {
        std::vector<spdlog::sink_ptr> sinks;

        auto consoleSink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        consoleSink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
        sinks.push_back(consoleSink);

        if (jsonLinesPath != nullptr)
        {
            auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(jsonLinesPath, /*truncate=*/false);
            fileSink->set_formatter(std::make_unique<JsonLinesFormatter>());
            sinks.push_back(fileSink);
        }

        auto logger = std::make_shared<spdlog::logger>("mirage", sinks.begin(), sinks.end());
        logger->set_level(ToSpdlogLevel(level));
        // Warnings and above are flushed immediately so a crash (see
        // mirage/utils/CrashHandler.cpp) never loses the diagnostic that
        // explains it to a buffered-but-unflushed sink.
        logger->flush_on(spdlog::level::warn);

        spdlog::set_default_logger(logger);
        spdlog::set_level(ToSpdlogLevel(level));
    }

    bool ParseLevel(const char *text, Level *out)
    {
        if (text == nullptr || out == nullptr)
            return false;

        std::string lower(text);
        for (char &c : lower)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        if (lower == "trace")
            *out = Level::eTrace;
        else if (lower == "debug")
            *out = Level::eDebug;
        else if (lower == "info")
            *out = Level::eInfo;
        else if (lower == "warn" || lower == "warning")
            *out = Level::eWarn;
        else if (lower == "error" || lower == "err")
            *out = Level::eError;
        else if (lower == "critical" || lower == "fatal")
            *out = Level::eCritical;
        else
            return false;

        return true;
    }
} // namespace Mirage::Logging
