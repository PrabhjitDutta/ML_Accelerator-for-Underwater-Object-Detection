// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2026 Advanced Micro Devices, Inc. All Rights Reserved.

// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

#ifndef X_HLS_STREAM_COMMON_SIM_H
#define X_HLS_STREAM_COMMON_SIM_H

/*
 * This file contains a C++ model of hls::stream.
 * It defines C simulation model.
 */
#ifndef __cplusplus

#error C++ is required to include this header file

#else

//////////////////////////////////////////////
// C level simulation models for hls::stream
//////////////////////////////////////////////
#include <queue>
#include <iostream>
#include <typeinfo>
#include <string>
#include <sstream>
#include <unordered_map>
#include <cstring>
#include <array>
#include <limits>
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>
#include <condition_variable>
#ifdef __VFS__
#include <exception>
#include <set>
#endif

#ifndef _MSC_VER
#include <cxxabi.h>
#include <stdlib.h>
#endif

namespace hls {
#if !defined(__HLS_COSIM__) && defined(__VITIS_HLS__)
// We are in bcsim mode, where reads must be non-blocking
#define ALLOW_EMPTY_HLS_STREAM_READS
#ifdef X_HLS_TASK_H
#error "bcsim is not supported with hls::tasks"
#endif
#endif

#ifdef __VFS__
struct termination_exception: public std::exception
{
};
#endif

#ifdef __VFS__
class stream_terminable;
#endif

#ifdef __VFS__
namespace stream_globals {
#else
class stream_globals {
friend class autorestart;
friend class task;
public:
#endif

#ifdef __VFS__
std::set<stream_terminable *>& get_stream_terminables()
#if !defined(DEFINE_HLS_STREAM_GLOBALS)
;
#else
{
  static std::set<stream_terminable *> stream_terminables;
  return stream_terminables;
}
#endif
#endif

#if !defined(__VFS__)
  static
#endif
  std::atomic<int> &get_max_size() 
#if defined(__VFS__) && !defined(DEFINE_HLS_STREAM_GLOBALS)
;
#else
  {
    static std::atomic<int> max_size(0);

    return max_size;
  }
#endif

#if !defined(__VFS__)
  static
#endif
  void print_max_size() 
#if defined(__VFS__) && !defined(DEFINE_HLS_STREAM_GLOBALS)
;
#else
  {
#if !defined(VFS_SUPPRESS_HLS_SIM_MESSAGES)
    std::cout << "INFO [HLS SIM]: The maximum depth reached by any hls::stream() instance in the design is " << get_max_size() << std::endl;
#endif
  }
#endif

#if !defined(__VFS__)
  static
#endif
  std::atomic<int> &get_task_counter() 
#if defined(__VFS__) && !defined(DEFINE_HLS_STREAM_GLOBALS)
;
#else
  {
      static std::atomic<int> task_counter(0);

      return task_counter;
  }
#endif

#if !defined(__VFS__)
  static
#endif
  std::atomic<int> &get_blocked_counter() 
#if defined(__VFS__) && !defined(DEFINE_HLS_STREAM_GLOBALS)
;
#else
  {
      static std::atomic<int> blocked_counter(0);

      return blocked_counter;
  }
#endif

#if !defined(__VFS__)
  static
#endif
  void incr_blocked_counter() 
#if defined(__VFS__) && !defined(DEFINE_HLS_STREAM_GLOBALS)
;
#else
  {
    get_blocked_counter()++;
  }
#endif

#if !defined(__VFS__)
  static
#endif
  void decr_blocked_counter() 
#if defined(__VFS__) && !defined(DEFINE_HLS_STREAM_GLOBALS)
;
#else
  {
    get_blocked_counter()--;
  }
#endif

#if !defined(__VFS__)
  static
#endif
  void incr_task_counter() 
#if defined(__VFS__) && !defined(DEFINE_HLS_STREAM_GLOBALS)
;
#else
  {
    get_task_counter()++;
  }
#endif

#if !defined(__VFS__)
  static
#endif
 void decr_task_counter() 
#if defined(__VFS__) && !defined(DEFINE_HLS_STREAM_GLOBALS)
;
#else
  {
    get_task_counter()--;
  }
#endif

#if !defined(__VFS__)
  static
#endif
  void wait_for_all_blocked() 
#if defined(__VFS__) && !defined(DEFINE_HLS_STREAM_GLOBALS)
;
#else
  {
    while (get_blocked_counter() < get_task_counter()) {
    }
  }
#endif

#if !defined(__VFS__)
  static
#endif
  bool all_threads_blocked() 
#if defined(__VFS__) && !defined(DEFINE_HLS_STREAM_GLOBALS)
;
#else
  {
    return get_blocked_counter() >= get_task_counter();
  }
#endif

#if !defined(__VFS__)
  static
#endif
  bool check_deadlock() 
#if defined(__VFS__) && !defined(DEFINE_HLS_STREAM_GLOBALS)
;
#else
  {
    return get_blocked_counter() > get_task_counter();
  }
#endif

#ifndef HLS_STREAM_THREAD_UNSAFE
#if !defined(__VFS__)
  static
#endif
  std::mutex &get_mutex() 
#if defined(__VFS__) && !defined(DEFINE_HLS_STREAM_GLOBALS)
;
#else
  {
      static std::mutex mutex;

      return mutex;
  }
#endif
#endif

  static void deadlock_thread() 
#if defined(__VFS__) && !defined(DEFINE_HLS_STREAM_GLOBALS)
;
#else
  {
    while (1) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      if (check_deadlock()) {
          if (get_task_counter()) {
            std::cerr << "ERROR [HLS SIM]: deadlock detected when simulating hls::tasks." 
                    << std::endl;
            std::cerr << "Execute C simulation in debug mode in the GUI and examine the"
                    << " source code location of all the blocked hls::stream::read()"
                    << " calls to debug." << std::endl;
          } else {
            std::cerr << "ERROR [HLS SIM]: an hls::stream or hls::ap_{none,vld,ack,hs} is read while empty,"
                    << " which may result in RTL simulation hanging." << std::endl;
            std::cerr << "If this is not expected, execute C simulation in debug mode in"
                    << " the GUI and examine the source code location of the blocked"
                    << " hls::stream::read() call to debug." << std::endl;
            std::cerr << "If this is as stream access and the empty read is expected, add -DALLOW_EMPTY_HLS_STREAM_READS"
                    << " to -cflags to turn this error into a warning and allow empty"
                    << " hls::stream reads to return the default value for the data type."
                    << std::endl;
          }
          abort();
      }
    }
  }
#endif


#if !defined(__VFS__)
  static
#endif
  void start_threads() 
#if defined(__VFS__) && !defined(DEFINE_HLS_STREAM_GLOBALS)
;
#else
  {
    // These initializations must be in ONE static function that is called elsewhere
#if defined(__HLS_COSIM__) 
    static std::thread t(deadlock_thread);
#endif
    static std::atomic_flag init_done = ATOMIC_FLAG_INIT;
    if (!init_done.test_and_set()) {
      // Perform global initialization actions once
      // Register function executed at exit
      std::atexit(print_max_size);
#if defined(__HLS_COSIM__) 
      // Detach the thread to avoid error at end with unwaited thread
      t.detach();
#endif
    }
  }
#endif
};

} // namespace hls

#endif // __cplusplus
#endif  // X_HLS_STREAM_COMMON_SIM_H


