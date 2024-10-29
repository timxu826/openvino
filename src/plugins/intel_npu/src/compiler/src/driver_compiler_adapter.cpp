// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver_compiler_adapter.hpp"

#include "graph_transformations.hpp"
#include "intel_npu/config/common.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "ze_intel_npu_uuid.h"
#include "zero_backend.hpp"
#include "zero_compiler_in_driver.hpp"
#include "zero_init.hpp"

namespace intel_npu {
namespace driverCompilerAdapter {

LevelZeroCompilerAdapter::LevelZeroCompilerAdapter(std::shared_ptr<IEngineBackend> iEngineBackend)
    : _logger("LevelZeroCompilerAdapter", Logger::global().level()) {
    _logger.debug("initialize LevelZeroCompilerAdapter start");

    std::shared_ptr<ZeroEngineBackend> zeroBackend = nullptr;
    zeroBackend = std::dynamic_pointer_cast<ZeroEngineBackend>(iEngineBackend);
    if (!zeroBackend) {
        OPENVINO_THROW("LevelZeroCompilerAdapter init failed to cast zeroBackend, zeroBackend is a nullptr");
    }

    ze_context_handle_t zeContext = static_cast<ze_context_handle_t>(zeroBackend->getContext());
    ze_driver_handle_t driverHandle = static_cast<ze_driver_handle_t>(zeroBackend->getDriverHandle());
    ze_device_handle_t deviceHandle = static_cast<ze_device_handle_t>(zeroBackend->getDeviceHandle());
    ze_graph_dditable_ext_curr_t& graph_ddi_table_ext = zeroBackend->getGraphDdiTable();

    uint32_t graphExtVersion = graph_ddi_table_ext.version();

    if (driverHandle == nullptr) {
        OPENVINO_THROW("LevelZeroCompilerAdapter failed to get properties about zeDriver");
    }

    _logger.info("LevelZeroCompilerAdapter creating adapter using graphExtVersion");

    switch (graphExtVersion) {
    case ZE_GRAPH_EXT_VERSION_1_3:
        apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ZE_GRAPH_EXT_VERSION_1_3>>(driverHandle,
                                                                                           deviceHandle,
                                                                                           zeContext,
                                                                                           graph_ddi_table_ext);
        break;
    case ZE_GRAPH_EXT_VERSION_1_4:
        apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ZE_GRAPH_EXT_VERSION_1_4>>(driverHandle,
                                                                                           deviceHandle,
                                                                                           zeContext,
                                                                                           graph_ddi_table_ext);
        break;
    case ZE_GRAPH_EXT_VERSION_1_5:
        apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ZE_GRAPH_EXT_VERSION_1_5>>(driverHandle,
                                                                                           deviceHandle,
                                                                                           zeContext,
                                                                                           graph_ddi_table_ext);
        break;
    case ZE_GRAPH_EXT_VERSION_1_6:
        apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ZE_GRAPH_EXT_VERSION_1_6>>(driverHandle,
                                                                                           deviceHandle,
                                                                                           zeContext,
                                                                                           graph_ddi_table_ext);
        break;
    case ZE_GRAPH_EXT_VERSION_1_7:
        apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ZE_GRAPH_EXT_VERSION_1_7>>(driverHandle,
                                                                                           deviceHandle,
                                                                                           zeContext,
                                                                                           graph_ddi_table_ext);
        break;
    case ZE_GRAPH_EXT_VERSION_1_8:
        apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ZE_GRAPH_EXT_VERSION_1_8>>(driverHandle,
                                                                                           deviceHandle,
                                                                                           zeContext,
                                                                                           graph_ddi_table_ext);
        break;
    default:
        apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ZE_GRAPH_EXT_VERSION_1_2>>(driverHandle,
                                                                                           deviceHandle,
                                                                                           zeContext,
                                                                                           graph_ddi_table_ext);
        break;
    }

    _logger.info("initialize LevelZeroCompilerAdapter complete, using graphExtVersion: %d.%d",
                 ZE_MAJOR_VERSION(graphExtVersion),
                 ZE_MINOR_VERSION(graphExtVersion));
}

uint32_t LevelZeroCompilerAdapter::getSupportedOpsetVersion() const {
    return apiAdapter->getSupportedOpsetVersion();
}

NetworkDescription LevelZeroCompilerAdapter::compile(const std::shared_ptr<const ov::Model>& model,
                                                     const Config& config) const {
    _logger.debug("compile start");
    return apiAdapter->compile(model, config);
}

ov::SupportedOpsMap LevelZeroCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                    const Config& config) const {
    _logger.debug("query start");
    return apiAdapter->query(model, config);
}

NetworkMetadata LevelZeroCompilerAdapter::parse(const std::vector<uint8_t>& network, const Config& config) const {
    _logger.debug("parse start");
    return apiAdapter->parse(network, config);
}

std::vector<ov::ProfilingInfo> LevelZeroCompilerAdapter::process_profiling_output(const std::vector<uint8_t>&,
                                                                                  const std::vector<uint8_t>&,
                                                                                  const Config&) const {
    OPENVINO_THROW("Profiling post-processing is not implemented.");
}

void LevelZeroCompilerAdapter::release(std::shared_ptr<const NetworkDescription> networkDescription) {
    _logger.info("release - using adapter to release networkDescription");
    apiAdapter->release(std::move(networkDescription));
}

CompiledNetwork LevelZeroCompilerAdapter::getCompiledNetwork(const NetworkDescription& networkDescription) {
    _logger.info("getCompiledNetwork - using adapter to perform getCompiledNetwork(networkDescription)");
    return apiAdapter->getCompiledNetwork(networkDescription);
}

}  // namespace driverCompilerAdapter
}  // namespace intel_npu
