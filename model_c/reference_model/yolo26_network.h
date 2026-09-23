// yolo26_network.h - entry point for the FP32 C-simulation of the pruned50 YOLO26s trunk.
#pragma once
#include <string>
#include "tensor_types.h"
#include "weights_loader.h"

// Runs layers 0..23 (backbone+neck+Detect pred-convs) on `input` and writes every top-level layer
// output (<i>_csim.bin), the 6 raw head maps (o2m_{0..2}/o2o_{0..2}_csim.bin) and a few layer-10
// attention sub-outputs into dumpdir, byte-matching the PyTorch dumper's layout.
// Returns the 3 one2one head maps. dumpdir "" = no files (the frame loop), and none of the dump-only work.
std::vector<Tensor> run_trunk(const Weights& W, const Tensor& input, const std::string& dumpdir);
