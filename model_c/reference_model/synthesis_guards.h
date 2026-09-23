// Host vs synthesis switches. HLS defines __SYNTHESIS__; g++ does not.
// Under g++: OpenMP and runtime asserts are active. Under HLS: both compile away.
#pragma once

#ifdef __SYNTHESIS__
  #define CSIM_OMP_STATIC
  #define CSIM_OMP_DYNAMIC
  #define CSIM_HOST_ASSERT(cond, msg) ((void)0)
#else
  #include <stdexcept>
  #include <string>
  #define CSIM_OMP_STATIC  _Pragma("omp parallel for schedule(static)")
  #define CSIM_OMP_DYNAMIC _Pragma("omp parallel for schedule(dynamic)")
  #define CSIM_HOST_ASSERT(cond, msg) do { if (!(cond)) throw std::runtime_error(msg); } while (0)
#endif
