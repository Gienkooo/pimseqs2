#pragma once
#include "Debug.h"

// =============================================================================
// DPU Logging System — Runtime-Configurable via MMseqs2 Verbosity
//
// Levels:
//   LOG_BENCH   — Always enabled (essential for performance measurements)
//   LOG_TRACE   — High-level milestones (Debug::INFO, verbosity >= 3)
//   LOG_MRAM    — Memory layout diagnostics (Debug::INFO, verbosity >= 4)
//   LOG_INDEX   — Index data quality (Debug::INFO, verbosity >= 4)
//   LOG_INDEX_EXTENDED — Extended index diagnostics (Debug::INFO, verbosity >= 4)
//   LOG_RESULTS — Hit statistics (Debug::INFO, verbosity >= 4)
//
// Control via: mmseqs2 -v <level> (0=NOTHING, 1=ERROR, 2=WARNING, 3=INFO)
// =============================================================================

// --- Benchmark logging: ALWAYS enabled (timings must always be visible) ---
#define LOG_BENCH(msg) Debug(Debug::INFO) << "[BENCH] " << msg << "\n"

// --- Trace logging: visible at default verbosity (INFO) ---
#define LOG_TRACE(msg) Debug(Debug::INFO) << "[TRACE] " << msg << "\n"

// --- Detailed diagnostics: only visible at high verbosity ---
// MMseqs2 Debug supports: NOTHING=0, ERROR=1, WARNING=2, INFO=3
// We gate detailed logs behind INFO so they appear with -v 3 (default)
// but can be suppressed with -v 2 or lower.
#define LOG_MRAM(msg) \
    do { if (Debug::debugLevel >= Debug::INFO) { Debug(Debug::INFO) << "[MRAM]  " << msg << "\n"; } } while (0)

#define LOG_INDEX(msg) \
    do { if (Debug::debugLevel >= Debug::INFO) { Debug(Debug::INFO) << "[INDEX] " << msg << "\n"; } } while (0)

#define LOG_INDEX_EXTENDED(msg) \
    do { if (Debug::debugLevel >= Debug::INFO) { Debug(Debug::INFO) << "[INDEX EXTENDED] " << msg << "\n"; } } while (0)

#define LOG_RESULTS(msg) \
    do { if (Debug::debugLevel >= Debug::INFO) { Debug(Debug::INFO) << "[RESULTS] " << msg << "\n"; } } while (0)
