// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <type_traits>
#include <utility>

#include "intel_npu/icompiler.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "zero_executor.hpp"

namespace intel_npu {
namespace driverCompilerAdapter {

using SerializedIR = std::pair<size_t, std::shared_ptr<uint8_t>>;

#define NotSupportQuery(T) (T == ZE_GRAPH_EXT_VERSION_1_2)

// ext version == 1.3 && 1.4, support API (pfnQueryNetworkCreate, pfnQueryNetworkDestroy,
// pfnQueryNetworkGetSupportedLayers)
#define SupportAPIGraphQueryNetworkV1(T) (T == ZE_GRAPH_EXT_VERSION_1_3 || T == ZE_GRAPH_EXT_VERSION_1_4)

// ext version >= 1.5, support API (pfnCreate2, pfnQueryNetworkCreate2, pfnQueryContextMemory)
#define SupportAPIGraphQueryNetworkV2(T) ((!NotSupportQuery(T) && !SupportAPIGraphQueryNetworkV1(T)))

// For ext version >= 1.5, pfnCreate2 api is avaible
#define NotSupportGraph2(T) \
    (T == ZE_GRAPH_EXT_VERSION_1_2 || T == ZE_GRAPH_EXT_VERSION_1_3 || T == ZE_GRAPH_EXT_VERSION_1_4)

// A bug inside the driver makes the "pfnGraphGetArgumentMetadata" call not safe for use prior to
// "ze_graph_dditable_ext_1_6_t".
// See: E#117498
#define NotSupportArgumentMetadata(T)                                                                   \
    (T == ZE_GRAPH_EXT_VERSION_1_2 || T == ZE_GRAPH_EXT_VERSION_1_3 || T == ZE_GRAPH_EXT_VERSION_1_4 || \
     T == ZE_GRAPH_EXT_VERSION_1_5)

#define UseCopyForNativeBinary(T)                                                                       \
    (T == ZE_GRAPH_EXT_VERSION_1_2 || T == ZE_GRAPH_EXT_VERSION_1_3 || T == ZE_GRAPH_EXT_VERSION_1_4 || \
     T == ZE_GRAPH_EXT_VERSION_1_5 || T == ZE_GRAPH_EXT_VERSION_1_6)

/**
 * Adapter to use CiD through ZeroAPI
 */
template <ze_graph_ext_version_t TableExtension>
class LevelZeroCompilerInDriver final : public ICompiler {
public:
    LevelZeroCompilerInDriver(ze_driver_handle_t driverHandle,
                              ze_device_handle_t deviceHandle,
                              ze_context_handle_t zeContext,
                              ze_graph_dditable_ext_curr_t& graph_ddi_table_ext);
    LevelZeroCompilerInDriver(const LevelZeroCompilerInDriver&) = delete;
    LevelZeroCompilerInDriver& operator=(const LevelZeroCompilerInDriver&) = delete;
    ~LevelZeroCompilerInDriver() override;

    uint32_t getSupportedOpsetVersion() const override final;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const override;

    NetworkDescription compile(const std::shared_ptr<const ov::Model>& model,
                               const Config& config) const override final;

    ze_result_t seriazlideIRModelAndCreateGraph(const std::shared_ptr<const ov::Model>& model,
                                                const Config& config,
                                                ze_device_graph_properties_t deviceGraphProperties,
                                                ze_graph_handle_t& graphHandle) const;

    NetworkMetadata parse(const std::vector<uint8_t>& network, const Config& config) const override final;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const std::vector<uint8_t>& network,
                                                            const Config& config) const override final {
        OPENVINO_THROW("Profiling post-processing is not implemented.");
    }

    template <ze_graph_ext_version_t T = TableExtension, std::enable_if_t<!NotSupportQuery(T), bool> = true>
    std::unordered_set<std::string> getQueryResultFromSupportedLayers(
        ze_result_t result,
        ze_graph_query_network_handle_t& hGraphQueryNetwork) const;

    /**
     * @brief Serialize input / output information to string format.
     * @details Format:
     * --inputs_precisions="0:<input1Precision> [1:<input2Precision>]"
     * --inputs_layouts="0:<input1Layout> [1:<input2Layout>]"
     * --outputs_precisions="0:<output1Precision>"
     * --outputs_layouts="0:<output1Layout>"
     *
     * For older compiler versions, the name of the inputs/outputs may be used instead of their indices.
     *
     * Since the layout information is no longer an important part of the metadata values when using the 2.0 OV
     * API, the layout fields shall be filled with default values in order to assure the backward compatibility
     * with the driver.
     */
    static std::string serializeIOInfo(const std::shared_ptr<const ov::Model>& model, const bool useIndices);

    void release(std::shared_ptr<const NetworkDescription> networkDescription) override;

    CompiledNetwork getCompiledNetwork(const NetworkDescription& networkDescription) override;

private:
    NetworkMetadata getNetworkMeta(ze_graph_handle_t graphHandle) const;

    SerializedIR serializeIR(const std::shared_ptr<const ov::Model>& model,
                             ze_graph_compiler_version_info_t compilerVersion) const;
    std::string serializeConfig(const Config& config, ze_graph_compiler_version_info_t& compilerVersion) const;

    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<NotSupportArgumentMetadata(T), bool> = true>
    void getMetadata(ze_graph_dditable_ext_curr_t& graphDdiTableExt,
                     ze_graph_handle_t graphHandle,
                     uint32_t index,
                     std::vector<IODescriptor>& inputs,
                     std::vector<IODescriptor>& outputs) const;

    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<!NotSupportArgumentMetadata(T), bool> = true>
    void getMetadata(ze_graph_dditable_ext_curr_t& graphDdiTableExt,
                     ze_graph_handle_t graphHandle,
                     uint32_t index,
                     std::vector<IODescriptor>& inputs,
                     std::vector<IODescriptor>& outputs) const;

    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<UseCopyForNativeBinary(T), bool> = true>
    void getNativeBinary(ze_graph_dditable_ext_curr_t& graphDdiTableExt,
                         ze_graph_handle_t graphHandle,
                         std::vector<uint8_t>& blob,
                         const uint8_t*& blobPtr,
                         size_t& blobSize) const;

    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<!UseCopyForNativeBinary(T), bool> = true>
    void getNativeBinary(ze_graph_dditable_ext_curr_t& graphDdiTableExt,
                         ze_graph_handle_t graphHandle,
                         std::vector<uint8_t>& /* unusedBlob */,
                         const uint8_t*& blobPtr,
                         size_t& blobSize) const;

    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<SupportAPIGraphQueryNetworkV2(T), bool> = true>
    ze_result_t seriazlideIRModelAndQueryNetworkCreateV2(const std::shared_ptr<const ov::Model>& model,
                                                         const Config& config,
                                                         ze_device_graph_properties_t deviceGraphProperties,
                                                         const ze_device_handle_t& _deviceHandle,
                                                         ze_graph_query_network_handle_t& hGraphQueryNetwork) const;

    // ext version >= 1.5, support API (pfnCreate2, pfnQueryNetworkCreate2, pfnQueryContextMemory)
    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<SupportAPIGraphQueryNetworkV2(T), bool> = true>
    std::unordered_set<std::string> queryImpl(const std::shared_ptr<const ov::Model>& model,
                                              const Config& config) const;

    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<SupportAPIGraphQueryNetworkV1(T), bool> = true>
    ze_result_t seriazlideIRModelAndQueryNetworkCreateV1(const std::shared_ptr<const ov::Model>& model,
                                                         const Config& config,
                                                         ze_device_graph_properties_t deviceGraphProperties,
                                                         const ze_device_handle_t& _deviceHandle,
                                                         ze_graph_query_network_handle_t& hGraphQueryNetwork) const;

    // ext version == 1.3 && 1.4, support API (pfnQueryNetworkCreate, pfnQueryNetworkDestroy,
    // pfnQueryNetworkGetSupportedLayers)
    template <ze_graph_ext_version_t T = TableExtension,
              typename std::enable_if_t<SupportAPIGraphQueryNetworkV1(T), bool> = true>
    std::unordered_set<std::string> queryImpl(const std::shared_ptr<const ov::Model>& model,
                                              const Config& config) const;

    // For ext version < 1.3
    template <ze_graph_ext_version_t T = TableExtension, typename std::enable_if_t<NotSupportQuery(T), bool> = true>
    std::unordered_set<std::string> queryImpl(const std::shared_ptr<const ov::Model>& model,
                                              const Config& config) const;

    template <ze_graph_ext_version_t T = TableExtension, typename std::enable_if_t<NotSupportGraph2(T), bool> = true>
    ze_result_t createGraph(const ze_graph_format_t& format,
                            const SerializedIR& serializedIR,
                            const std::string& buildFlags,
                            const uint32_t& flags,
                            ze_graph_handle_t* graph) const;

    template <ze_graph_ext_version_t T = TableExtension, typename std::enable_if_t<!NotSupportGraph2(T), bool> = true>
    ze_result_t createGraph(const ze_graph_format_t& format,
                            const SerializedIR& serializedIR,
                            const std::string& buildFlags,
                            const uint32_t& flags,
                            ze_graph_handle_t* graph) const;

private:
    ze_driver_handle_t _driverHandle = nullptr;
    ze_device_handle_t _deviceHandle = nullptr;
    ze_context_handle_t _context = nullptr;

    ze_graph_dditable_ext_curr_t& _graphDdiTableExt;
    Logger _logger;
};

}  // namespace driverCompilerAdapter
}  // namespace intel_npu
