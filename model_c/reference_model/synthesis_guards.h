// synthesis_guards.h -- host/synthesis switches shared by the trunk compute datapath.
//
// Vitis HLS predefines __SYNTHESIS__ during synthesis; plain g++ (the C-simulation, the validation
// reference) does not. Guarding host-only constructs here keeps ONE source serving both:
//   * under g++  -> OpenMP pragmas active, runtime assertions active  (bit-exact C-sim reference)
//   * under HLS  -> they vanish, leaving clean loops for real #pragma HLS to be added at synthesis
//
// #pragma omp is a host-thread directive and is inert under HLS regardless (see the "#pragma omp vs
// #pragma HLS" note in notes/yolo26s-c-sim.md); guarding it is about intent and keeping the synthesized
// loop bodies clean, not correctness. The _Pragma() operator form expands identically to the #pragma
// line under g++ -fopenmp, so the C-sim's measured ~11.5x threading is unchanged.
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
  // Datapath sanity checks are host-only debug aids; under synthesis the shapes are compile-time
  // invariants, so the check compiles out.
  #define CSIM_HOST_ASSERT(cond, msg) do { if (!(cond)) throw std::runtime_error(msg); } while (0)
#endif
