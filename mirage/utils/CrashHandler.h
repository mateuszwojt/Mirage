#pragma once

// Installs process-wide handlers so a crash produces a logged, flushed
// backtrace instead of silently vanishing - resolves the "no crash
// reporting" half of docs/PRODUCTION_READINESS.md's Tier 3 logging finding.
//
// Call once, early in main(), after Mirage::Logging::Init() (mirage/utils/Log.h)
// so the handlers have a configured logger/sink to write to.
//
// POSIX (macOS/Linux): std::set_terminate + SIGSEGV/SIGABRT/SIGBUS/SIGFPE/
// SIGILL handlers that log a backtrace via backtrace()/backtrace_symbols(),
// flush the logger, then re-raise the signal with its default disposition so
// the process still exits/cores the way the OS and any outer supervisor
// (farm dispatcher, debugger) expects.
//
// Windows: not yet implemented. A Windows leg needs CaptureStackBackTrace/
// DbgHelp (SymFromAddr) instead of execinfo.h's backtrace(), and typically a
// vectored exception handler (AddVectoredExceptionHandler) rather than
// signal() for SEH-level faults - tracked alongside
// docs/PRODUCTION_READINESS.md's Tier 3 Windows-platform-support item, since
// it needs a real Windows build to develop and test against. Calling
// InstallCrashHandler() on Windows today is safe (it installs the
// std::terminate hook, which is portable) but logs a one-time warning that
// signal-level crash reporting isn't wired up yet.
namespace Mirage::Logging
{
    void InstallCrashHandler();
}
