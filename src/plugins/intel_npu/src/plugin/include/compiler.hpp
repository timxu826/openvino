// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Factory

#pragma once

#include "backends.hpp"
#include "intel_npu/icompiler.hpp"
#include "intel_npu/npu_private_properties.hpp"  // AL
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace intel_npu {

ov::SoPtr<ICompiler> createCompiler(std::shared_ptr<intel_npu::NPUBackends> npuBackends,
                                    ov::intel_npu::CompilerType compilerType);

}  // namespace intel_npu
