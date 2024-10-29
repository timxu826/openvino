// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_executor.hpp"

#include <ze_api.h>

#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/common.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/runtime/properties.hpp"
#include "ze_command_queue_npu_ext.h"
#include "zero_device.hpp"

using namespace intel_npu;

ZeroExecutor::ZeroExecutor(const std::shared_ptr<const ZeroInitStructsHolder>& initStructs,
                           const std::shared_ptr<const NetworkDescription>& networkDescription,
                           const Config& config,
                           const uint32_t& group_ordinal)
    : _config(config),
      _logger("Graph", _config.get<LOG_LEVEL>()),
      _initStructs(initStructs),
      _networkDesc(networkDescription),
      _graph_ddi_table_ext(_initStructs->getGraphDdiTable()),
      _group_ordinal(group_ordinal),
      _command_queues{std::make_shared<CommandQueue>(_initStructs->getDevice(),
                                                     _initStructs->getContext(),
                                                     zeroUtils::toZeQueuePriority(_config.get<MODEL_PRIORITY>()),
                                                     _initStructs->getCommandQueueDdiTable(),
                                                     _config,
                                                     group_ordinal)} {
    _logger.debug("ZeroExecutor::ZeroExecutor - create graph");
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_GRAPH, itt::domains::LevelZeroBackend, "Executor::ZeroExecutor", "graphCreate");

    // _graph is a nullptr for CIP path, a new handle will be obtained from the driver based on the given
    // compiledNetwork _graph gets (reuses) graphHandle from the compiler for CID path
    if (_networkDesc->metadata.graphHandle == nullptr) {
        _logger.debug("create graph handle on executor");
        ze_graph_desc_t desc{ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                             nullptr,
                             ZE_GRAPH_FORMAT_NATIVE,
                             _networkDesc->compiledNetwork.size(),
                             _networkDesc->compiledNetwork.data(),
                             nullptr};
        ze_result_t result =
            _graph_ddi_table_ext.pfnCreate(_initStructs->getContext(), _initStructs->getDevice(), &desc, &_graph);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnCreate", result, _graph_ddi_table_ext);

    } else {
        _logger.debug("reuse graph handle created from compiler");
        _graph = static_cast<ze_graph_handle_t>(_networkDesc->metadata.graphHandle);
    }

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "pfnGetProperties");
    _logger.debug("performing pfnGetProperties");
    ze_graph_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;

    ze_result_t result = _graph_ddi_table_ext.pfnGetProperties(_graph, &props);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetProperties", result, _graph_ddi_table_ext);

    auto targetDriverExtVersion = _graph_ddi_table_ext.version();
    if (targetDriverExtVersion <= ZE_GRAPH_EXT_VERSION_1_1) {
        OPENVINO_THROW("Incompatibility between the NPU plugin and driver! The driver version is too old, please "
                       "update the driver version");
    }

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "pfnGetArgumentProperties3");
    _logger.debug("performing pfnGetArgumentProperties3");
    for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
        ze_graph_argument_properties_3_t arg3{};
        arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
        ze_result_t result = _graph_ddi_table_ext.pfnGetArgumentProperties3(_graph, index, &arg3);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetArgumentProperties3", result, _graph_ddi_table_ext);

        if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            _input_descriptors.push_back(ArgumentDescriptor{arg3, index});
        } else {
            _output_descriptors.push_back(ArgumentDescriptor{arg3, index});
        }
    }

    if (_graph_ddi_table_ext.version() < ZE_GRAPH_EXT_VERSION_1_8) {
        initialize_graph_through_command_list();
    } else {
        ze_graph_properties_2_t properties = {};
        properties.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
        _graph_ddi_table_ext.pfnGetProperties2(_graph, &properties);

        if (properties.initStageRequired & ZE_GRAPH_STAGE_INITIALIZE) {
            OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "pfnGraphInitialize");
            _graph_ddi_table_ext.pfnGraphInitialize(_graph);
        }

        if (properties.initStageRequired & ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE) {
            initialize_graph_through_command_list();
        }
    }

    if (config.has<WORKLOAD_TYPE>()) {
        setWorkloadType(config.get<WORKLOAD_TYPE>());
    }
}

void ZeroExecutor::initialize_graph_through_command_list() const {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_GRAPH,
                      itt::domains::LevelZeroBackend,
                      "Executor::ZeroExecutor",
                      "initialize_graph_through_command_list");

    _logger.debug("ZeroExecutor::ZeroExecutor init start - create graph_command_list");
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Executor::ZeroExecutor");
    CommandList graph_command_list(_initStructs->getDevice(),
                                   _initStructs->getContext(),
                                   _graph_ddi_table_ext,
                                   _config,
                                   _group_ordinal);
    _logger.debug("ZeroExecutor::ZeroExecutor - create graph_command_queue");
    CommandQueue graph_command_queue(_initStructs->getDevice(),
                                     _initStructs->getContext(),
                                     ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
                                     _initStructs->getCommandQueueDdiTable(),
                                     _config,
                                     _group_ordinal);
    _logger.debug("ZeroExecutor::ZeroExecutor - create fence");
    Fence fence(graph_command_queue, _config);

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "appendGraphInitialize");
    _logger.debug("ZeroExecutor::ZeroExecutor - performing appendGraphInitialize");
    graph_command_list.appendGraphInitialize(_graph);
    _logger.debug("ZeroExecutor::ZeroExecutor - closing graph command list");
    graph_command_list.close();

    OV_ITT_TASK_NEXT(ZERO_EXECUTOR_GRAPH, "queue_execute");
    _logger.debug("ZeroExecutor::ZeroExecutor - performing executeCommandList");
    graph_command_queue.executeCommandList(graph_command_list, fence);
    _logger.debug("ZeroExecutor::ZeroExecutor - performing hostSynchronize");
    fence.hostSynchronize();
    _logger.debug("ZeroExecutor::ZeroExecutor - hostSynchronize completed");
}

void ZeroExecutor::setWorkloadType(const ov::WorkloadType workloadType) const {
    ze_command_queue_workload_type_t zeWorkloadType;
    switch (workloadType) {
    case ov::WorkloadType::DEFAULT:
        zeWorkloadType = ze_command_queue_workload_type_t::ZE_WORKLOAD_TYPE_DEFAULT;
        break;
    case ov::WorkloadType::EFFICIENT:
        zeWorkloadType = ze_command_queue_workload_type_t::ZE_WORKLOAD_TYPE_BACKGROUND;
        break;
    default:
        OPENVINO_THROW("Unknown value for WorkloadType!");
    }

    _command_queues->setWorkloadType(zeWorkloadType);
}

void ZeroExecutor::setArgumentValue(uint32_t argi_, const void* argv_) const {
    ze_result_t result = _graph_ddi_table_ext.pfnSetArgumentValue(_graph, argi_, argv_);
    if (ZE_RESULT_SUCCESS != result) {
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("zeGraphSetArgumentValue", result, _graph_ddi_table_ext);
    }
}

void ZeroExecutor::mutexLock() const {
    _mutex.lock();
}

void ZeroExecutor::mutexUnlock() const {
    _mutex.unlock();
}

ZeroExecutor::~ZeroExecutor() {
    _logger.debug("~ZeroExecutor() - pfnDestroy _graph ");
    auto result = _graph_ddi_table_ext.pfnDestroy(_graph);
    if (ZE_RESULT_SUCCESS != result) {
        _logger.error("_graph_ddi_table_ext.pfnDestroy failed %#X", uint64_t(result));
    }
}
