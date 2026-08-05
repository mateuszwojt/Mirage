#include "mirage/utils/CrashHandler.h"
#include "mirage/utils/Log.h"

#include <spdlog/spdlog.h>

#include <csignal>
#include <cstdlib>
#include <exception>

#if defined(__APPLE__) || defined(__linux__)
#define MIRAGE_HAVE_POSIX_BACKTRACE 1
#include <execinfo.h>
#endif

namespace Mirage::Logging
{
    namespace
    {
        // backtrace_symbols() allocates via malloc and the logging calls
        // below are not async-signal-safe in the strict POSIX sense - a
        // stricter handler would pre-allocate everything and use only
        // write(2). This is a deliberate, pragmatic tradeoff: the priority
        // here is a best-effort diagnostic on the way out, and in practice a
        // process that's already fatally signaling has little left to lose
        // from a second-order signal-safety violation in its own crash path.
        void LogBacktrace()
        {
#if defined(MIRAGE_HAVE_POSIX_BACKTRACE)
            constexpr int kMaxFrames = 64;
            void *frames[kMaxFrames];
            int count = backtrace(frames, kMaxFrames);
            MIRAGE_LOG_CRITICAL("Crash backtrace ({} frames):", count);

            char **symbols = backtrace_symbols(frames, count);
            if (symbols != nullptr)
            {
                for (int i = 0; i < count; ++i)
                    MIRAGE_LOG_CRITICAL("  #{}: {}", i, symbols[i]);
                std::free(symbols);
            }
#else
            MIRAGE_LOG_CRITICAL(
                "Crash backtrace: not available on this platform yet "
                "(see mirage/utils/CrashHandler.h's Windows note)");
#endif
        }

        void FlushLogger()
        {
            if (auto logger = spdlog::default_logger())
                logger->flush();
        }

        [[noreturn]] void OnTerminate()
        {
            MIRAGE_LOG_CRITICAL("std::terminate called (unhandled exception or noexcept violation)");
            LogBacktrace();
            FlushLogger();
            std::abort();
        }

#if defined(MIRAGE_HAVE_POSIX_BACKTRACE)
        extern "C" void OnFatalSignal(int signum)
        {
            const char *name = "unknown signal";
            switch (signum)
            {
            case SIGSEGV:
                name = "SIGSEGV";
                break;
            case SIGABRT:
                name = "SIGABRT";
                break;
#ifdef SIGBUS
            case SIGBUS:
                name = "SIGBUS";
                break;
#endif
            case SIGFPE:
                name = "SIGFPE";
                break;
            case SIGILL:
                name = "SIGILL";
                break;
            }

            MIRAGE_LOG_CRITICAL("Fatal signal received: {} ({})", name, signum);
            LogBacktrace();
            FlushLogger();

            // Restore the default disposition and re-raise, rather than
            // calling _exit()/abort() directly - this preserves the OS's
            // normal behavior (core dump, correct exit status for a farm
            // dispatcher or debugger attached to the process) instead of
            // masking it behind our own handler.
            std::signal(signum, SIG_DFL);
            std::raise(signum);
        }
#endif
    } // namespace

    void InstallCrashHandler()
    {
        std::set_terminate(OnTerminate);

#if defined(MIRAGE_HAVE_POSIX_BACKTRACE)
        std::signal(SIGSEGV, OnFatalSignal);
        std::signal(SIGABRT, OnFatalSignal);
#ifdef SIGBUS
        std::signal(SIGBUS, OnFatalSignal);
#endif
        std::signal(SIGFPE, OnFatalSignal);
        std::signal(SIGILL, OnFatalSignal);
#else
        MIRAGE_LOG_WARN(
            "InstallCrashHandler: signal-based crash reporting is not "
            "implemented on this platform yet (see CrashHandler.h)");
#endif
    }
} // namespace Mirage::Logging
