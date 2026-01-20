#pragma once
#include "Debug.h"

// --- Toggles ---
//#define DPU_LOG_TRACE   // High-level milestones
#define DPU_LOG_BENCH   // Timing/Throughput
#define DPU_LOG_MRAM    // Memory layout
#define DPU_LOG_INDEX   // Data quality
#define DPU_LOG_INDEX_EXTENDED   // Extended data quality
#define DPU_LOG_RESULTS // Hit statistics

// --- Macro Helpers ---

#ifdef DPU_LOG_TRACE
  #define LOG_TRACE(msg) Debug(Debug::INFO) << "[TRACE] " << msg << "\n"
#else
  #define LOG_TRACE(msg)
#endif

#ifdef DPU_LOG_BENCH
  #define LOG_BENCH(msg) Debug(Debug::INFO) << "[BENCH] " << msg << "\n"
#else
  #define LOG_BENCH(msg)
#endif

#ifdef DPU_LOG_MRAM
  #define LOG_MRAM(msg) Debug(Debug::INFO) << "[MRAM]  " << msg << "\n"
#else
  #define LOG_MRAM(msg)
#endif

#ifdef DPU_LOG_INDEX
  #define LOG_INDEX(msg) Debug(Debug::INFO) << "[INDEX] " << msg << "\n"
#else
  #define LOG_INDEX(msg)
#endif

#ifdef  DPU_LOG_INDEX_EXTENDED
  #define LOG_INDEX_EXTENDED(msg) Debug(Debug::INFO) << "[INDEX EXTENDED] " << msg << "\n"
#else
  #define LOG_INDEX_EXTENDED(msg)
#endif

#ifdef DPU_LOG_RESULTS
  #define LOG_RESULTS(msg) Debug(Debug::INFO) << "[RESULTS] " << msg << "\n"
#else
  #define LOG_RESULTS(msg)
#endif
