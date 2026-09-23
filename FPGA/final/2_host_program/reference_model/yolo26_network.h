#pragma once
#include <string>
#include "tensor_types.h"
#include "weights_loader.h"

// Runs layers 0..23 and returns the 3 one2one head maps.
// Non-empty dumpdir: also writes every layer output (<i>_csim.bin) and the 6 head maps.
std::vector<Tensor> run_trunk(const Weights& W, const Tensor& input, const std::string& dumpdir);
