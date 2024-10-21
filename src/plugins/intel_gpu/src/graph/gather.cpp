// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

#include "gather_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(gather)

template <class T>
void dump2(memory::ptr mem, stream& stream, bool dump_raw, int output_num) {
    auto&& size = mem->get_layout().get_tensor();

    GPU_DEBUG_GET_INSTANCE(debug_config);
    auto batch_size = std::max(std::min(debug_config->dump_layers_limit_batch, size.batch[0]), 1);
    tensor tmp_size(size);
    tmp_size.batch[0] = batch_size;
    if (tmp_size == size) {
        std::cout << "shape: " << size.to_string() << " ";
        std::cout << "(count: " << size.count()
                    << (dump_raw ? " raw data" : "") << std::endl;
    } else {
        std::cout << "shape: " << tmp_size.to_string() << " ";
        std::cout << "(count: " << tmp_size.count()
                    << ", original shape: " << size.to_string() << ")"
                    << (dump_raw ? " raw data" : "") << std::endl;
    }

    if (size.count() == 0) {
        std::cout << "Empty buffer" << std::endl;
        return;
    }

    mem_lock<T, mem_lock_type::read> lock(mem, stream);
    auto mem_ptr = lock.data();

    auto layout = mem->get_layout();
    size_t x_pitch;

    try {
        auto tensor_x0 = tensor(batch(0), feature(0), spatial(0, 0, 0, 0));
        auto tensor_x1 = tensor(batch(0), feature(0), spatial(1, 0, 0, 0));
        auto x0 = layout.get_linear_offset(tensor_x0);
        auto x1 = layout.get_linear_offset(tensor_x1);
        x_pitch =  (x1 - x0);
    } catch (...) {
        // When spatial size of x=0, x_pitch is meaningless
        x_pitch = 0;
    }
    std::stringstream buffer;

    int cnt = 0;
    if (!dump_raw) {
        for (cldnn::tensor::value_type g = 0; g < size.group[0]; ++g) {
            for (cldnn::tensor::value_type b = 0; b < batch_size; ++b) {
                for (cldnn::tensor::value_type f = 0; f < size.feature[0]; ++f) {
                    for (cldnn::tensor::value_type w = 0; w < size.spatial[3]; ++w) {
                        for (cldnn::tensor::value_type z = 0; z < size.spatial[2]; ++z) {
                            for (cldnn::tensor::value_type y = 0; y < size.spatial[1]; ++y) {
                                cldnn::tensor t(cldnn::group(g), cldnn::batch(b), cldnn::feature(f), cldnn::spatial(0, y, z, w));
                                size_t input_it = mem->get_layout().get_linear_offset(t);

                                for (cldnn::tensor::value_type x = 0; x < size.spatial[0]; ++x, input_it += x_pitch) {
                                    if (cnt < output_num) {
                                        buffer << std::fixed << std::setprecision(6) << static_cast<float>(mem_ptr[input_it]) << std::endl;
                                    }
                                    cnt++;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (size_t i = 0; i < lock.size(); ++i) {
            buffer << std::fixed << std::setprecision(6) << static_cast<float>(mem_ptr[i]) << std::endl;
        }
    }
    std::cout << buffer.str();
}

layout gather_inst::calc_output_layout(gather_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<gather>();

    auto input_layout = impl_param.get_input_layout();
    std::vector<tensor::value_type> dims_converted;
    for (auto dim : desc->output_shape) {
        dims_converted.push_back(static_cast<tensor::value_type>(dim));
    }
    // extend shape to 4d
    for (size_t i = dims_converted.size(); i < 4; i++)
        dims_converted.push_back(1);

    format output_format = input_layout.format;
    if (dims_converted.size() == 5) {
        switch (input_layout.format) {
        case format::bfyx:
            output_format = format::get_default_format(dims_converted.size());
            break;
        case format::b_fs_yx_fsv16:
            output_format = format::b_fs_zyx_fsv16;
            break;
        case format::b_fs_yx_fsv32:
            output_format = format::b_fs_zyx_fsv32;
            break;
        case format::bs_fs_yx_bsv16_fsv16:
            output_format = format::bs_fs_zyx_bsv16_fsv16;
            break;
        default:
            break;
        }
    } else if (dims_converted.size() == 6) {
        switch (input_layout.format) {
        case format::bfyx:
        case format::bfzyx:
        case format::b_fs_zyx_fsv16:
        case format::b_fs_zyx_fsv32:
            output_format = format::get_default_format(dims_converted.size());
            break;
        default:
            break;
        }
    }
    auto output_type = input_layout.data_type;
    if (impl_param.typed_desc<gather>()->compressed_weights) {
        output_type = impl_param.typed_desc<gather>()->decompressed_type;
    }
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    return layout{output_type,
                  output_format,
                  tensor(format::get_default_format(dims_converted.size()), dims_converted)};
}

template<typename ShapeType>
std::vector<layout> gather_inst::calc_output_layouts(gather_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<gather>();

    auto input0_layout = impl_param.get_input_layout(0);
    auto input1_layout = impl_param.get_input_layout(1);

    auto output_type = input0_layout.data_type;
    if (impl_param.typed_desc<gather>()->compressed_weights) {
        output_type = impl_param.typed_desc<gather>()->decompressed_type;
    }
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    ov::op::v8::Gather op;
    op.set_batch_dims(desc->batch_dim);
    std::vector<ShapeType> output_shapes = {ShapeType()};
    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        input1_layout.get<ShapeType>(),
        ShapeType{1} // axis input is removed on gather primitive creation, so we can't use get_dependency(2)
    };

    int64_t axis = desc->axis;

    auto axis_tensor = ov::Tensor(ov::element::i64, ov::Shape{1}, static_cast<void*>(&axis));
    std::unordered_map<size_t, ov::Tensor> const_data = {{2, axis_tensor}};
    output_shapes = ov::op::shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));

    format output_format = format::adjust_to_rank(input0_layout.format, output_shapes[0].size());

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> gather_inst::calc_output_layouts<ov::PartialShape>(gather_node const& node, const kernel_impl_params& impl_param);

std::string gather_inst::to_string(gather_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite gather_info;
    gather_info.add("input id", input.id());
    gather_info.add("axis", desc->axis);
    gather_info.add("batch_dim", desc->batch_dim);
    gather_info.add("output shape", cldnn::to_string(desc->output_shape));
    gather_info.add("compressed weights", desc->compressed_weights ? "true" : "false");
    if (desc->compressed_weights) {
        gather_info.add("decompression scale id", desc->decompression_scale.pid);
        gather_info.add("decompression zp id", desc->decompression_zero_point.pid);
        if (desc->decompression_zero_point_scalar.has_value()) {
            gather_info.add("decompression zp value", desc->decompression_zero_point_scalar.value());
        }
    }

    node_info->add("gather info", gather_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

void gather_inst::on_execute() {
    update_output_memory();
}

void gather_inst::update_output_memory() {
    if (!can_be_optimized())
        return;
    if (static_cast<bool>(_outputs[0]) && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    if (_node != nullptr)
        build_deps();

    std::string tmp = id();
    std::string match("SecondStagePostprocessor/BatchMultiClassNonMaxSuppression/map/while/MultiClassNonMaxSuppression/ClipToWindow/Gather/GatherV2_6");
    auto found = tmp.find(match);
    if (found != std::string::npos) {
        std::cout << tmp << " in gather.cpp to " << match << std::endl;
        // mem_lock<ov::float16, mem_lock_type::read> mem_lock(input_memory_ptr(), get_network().get_stream());
        // auto data = mem_lock.data();
        dump2<ov::float16>(input_memory_ptr(), get_network().get_stream(), false, 10);
        dump2<ov::float16>(dependencies().at(0).first->output_memory_ptr(), get_network().get_stream(), false, 10);
    }

    GPU_DEBUG_TRACE_DETAIL << id() << " : update_output_memory with mem of input " << get_node().get_dependency(0).id()
                           << " : " << input_memory_ptr()->buffer_ptr() << std::endl;
    _outputs[0] = input_memory_ptr();
    _mem_allocated = false;
}

gather_inst::typed_primitive_inst(network& network, gather_node const& node) : parent(network, node) {}

}  // namespace cldnn
