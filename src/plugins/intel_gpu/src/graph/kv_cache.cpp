// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/kv_cache_compressed.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"
#include "kv_cache_inst.h"
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>
#include "utils.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(kv_cache)
GPU_DEFINE_PRIMITIVE_TYPE_ID(stateless_kv)

kv_cache_inst::typed_primitive_inst(network& network, const kv_cache_node& node) :
    parent{network, node, false},
    memory_state::releasable_variable{node.get_primitive()->variable_info.variable_id} {
    thread_local size_t kv_cache_counter = 0;
    kv_cache_id = kv_cache_counter++;
}

int64_t kv_cache_inst::compute_trim_length(const kernel_impl_params& impl_param, const kv_cache& desc) {
    if (!desc.trim)
        return 0;

    const size_t past_seq_len_idx = desc.indirect ? 3 : 2;
    const auto mem_dep_it = impl_param.memory_deps.find(past_seq_len_idx);
    if (mem_dep_it == impl_param.memory_deps.end())
        return 0;

    const auto& past_seq_len_mem = mem_dep_it->second;
    const auto past_seq_len_layout = past_seq_len_mem->get_layout();
    if (past_seq_len_layout.count() == 0)
        return 0;

    OPENVINO_ASSERT(past_seq_len_layout.count() == 1);
    cldnn::mem_lock<uint8_t, mem_lock_type::read> past_seq_len_mem_lock(past_seq_len_mem, impl_param.get_stream());
    auto past_seq_len_tensor = make_tensor(past_seq_len_layout, past_seq_len_mem_lock.data());
    const auto past_dim_updated = ov::get_tensor_data_as<int64_t>(past_seq_len_tensor);

    const auto& past_layout = impl_param.get_input_layout(0);
    const auto past_shape = past_layout.get_partial_shape();
    const auto sequence_axis = kv_cache_inst::get_sequence_axis(desc.concat_axis, past_shape.size());
    OPENVINO_ASSERT(sequence_axis >= 0);
    const auto sequence_axis_idx = static_cast<size_t>(sequence_axis);
    OPENVINO_ASSERT(past_shape[sequence_axis_idx].is_static());

    const auto trim_length = past_shape[sequence_axis_idx].get_length() - past_dim_updated[0];
    OPENVINO_ASSERT(trim_length >= 0, "[GPU] past_seq_len shouldn't exceed stored sequence length");

    return trim_length;
}

layout kv_cache_inst::calc_output_layout(const kv_cache_node& node, kernel_impl_params const& impl_param) {
    return impl_param.input_layouts[0];
}

template<typename ShapeType>
std::vector<layout> kv_cache_inst::calc_output_layouts(kv_cache_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<kv_cache>();

    std::vector<ShapeType> input_shapes = {impl_param.get_input_layout(0).get<ShapeType>(),
                                           impl_param.get_input_layout(1).get<ShapeType>()};
    size_t input_idx = 2;
    if (desc->indirect) {
        input_shapes.push_back(impl_param.get_input_layout(input_idx++).get<ShapeType>());
    }

    if (desc->compressed) {
        input_shapes.push_back(impl_param.get_input_layout(3).get<ShapeType>());

        if (desc->get_compression_zp_inputs_num() > 0) {
            input_shapes.push_back(impl_param.get_input_layout(4).get<ShapeType>());
        }
    }

    const auto kv_cache_trim_length = kv_cache_inst::compute_trim_length(impl_param, *desc);

    std::vector<ShapeType> output_shapes;
    if (desc->compressed) {
        OPENVINO_ASSERT(kv_cache_trim_length == 0);  // compressed kv should not do any trim
        ov::intel_gpu::op::KVCacheCompressed op;
        op.set_output_size(desc->num_outputs);
        op.set_concat_axis(desc->concat_axis);
        op.set_gather_axis(desc->gather_axis);
        op.set_quantization_attrs(desc->quantization_attributes);
        op.set_trim(desc->trim);

        output_shapes = shape_infer(&op, input_shapes);
    } else {
        ov::intel_gpu::op::KVCache op;
        op.set_output_size(desc->num_outputs);
        op.set_concat_axis(desc->concat_axis);
        op.set_gather_axis(desc->gather_axis);
        op.set_trim(desc->trim);
        op.set_trim_length(kv_cache_trim_length);

        output_shapes = shape_infer(&op, input_shapes);
    }

    static const std::map<size_t, size_t> ports_map = {{0, 0}, {1, 2}, {2, 3}, {3, 4}};

    // For INT4 KV-cache, pack two values per byte (halve inner dim)
    if (desc->compressed) {
        const auto kv_cache_dt = impl_param.get_program().get_config().get_kv_cache_precision();
        if (ov::element::Type(kv_cache_dt).bitwidth() == 4 && output_shapes[0].size() > 0) {
            auto inner_dim_idx = output_shapes[0].size() - 1;
            output_shapes[0][inner_dim_idx] = output_shapes[0][inner_dim_idx] / 2;
        }
    }

    std::vector<layout> out_layouts;
    for (size_t i = 0; i < desc->num_outputs; i++) {
        auto out_type = desc->output_data_types[i].value_or(impl_param.get_input_layout(ports_map.at(i)).data_type);
        out_layouts.emplace_back(output_shapes[i], out_type, impl_param.get_output_layout(i).format);
    }

    return out_layouts;
}

template std::vector<layout> kv_cache_inst::calc_output_layouts<ov::PartialShape>(kv_cache_node const& node, const kernel_impl_params& impl_param);

std::string kv_cache_inst::to_string(const kv_cache_node& node) {
    auto node_info = node.desc_to_json();
    json_composite kv_cache_info;
    kv_cache_info.add("input id", node.input().id());
    kv_cache_info.add("variable id", node.get_primitive()->variable_info.variable_id);
    kv_cache_info.add("variable shape", node.get_primitive()->variable_info.data_shape);
    kv_cache_info.add("variable type", node.get_primitive()->variable_info.data_type);
    kv_cache_info.add("concat axis", node.get_primitive()->concat_axis);
    kv_cache_info.add("gather axis", node.get_primitive()->gather_axis);
    kv_cache_info.add("indirect", node.get_primitive()->indirect);
    kv_cache_info.add("compressed", node.get_primitive()->compressed);
    kv_cache_info.add("output_storage_type", static_cast<int>(node.get_primitive()->quantization_attributes.output_storage_type));
    kv_cache_info.add("scales_zp_output_order", node.get_primitive()->quantization_attributes.scales_zp_output_order);
    node_info->add("kv_cache info", kv_cache_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

int32_t kv_cache_inst::get_prealloc_iter_num() {
    // - When a kv_cache_inst runs out of the pre-allocated memory and requires additional memory,
    //   it allocate a new memory. And then it copies data in the original memory to the new memory.
    //   Since the original memory is still assigned to the ReadValue, even after the copying is finished,
    //   we will have 2x memories for the kv cache. And the original memory will be released when the ReadValue is
    //   called, i.e., at the next iteration.
    // - If this alloc/copy happens at the same time for all the kv cache memory, there will be a memory peak at that
    //   iteration.
    // - Therfore, to avoid this situation where the allocation and copying occurs simutaneously for all the kv_cache_insts,
    //   we assigned different prealloc-size for each kv cache so that we could prevent a memory peak
    return 128 + kv_cache_id % 64;
}

void kv_cache_inst::update_shape_info_tensor(const kernel_impl_params& params) {
    if (!_shape_info_memory) {
        allocate_shape_info_memory();
    }
    mem_lock<int32_t> lock(_shape_info_memory, _network.get_stream());
    auto shape_info_ptr = lock.data();
    size_t offset = 0;

    size_t i = 0;
    // [kv_state, kv_new_token, [beam_idx, bt_past]]
    for (i = 0; i < get_node().get_dependencies().size(); i++) {
        const auto& node_in_lay = get_node().get_input_layout(i);
        const auto& runtime_in_lay = params.input_layouts[i];

        GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for input[" << i << "]" << std::endl;
        fill_shape_info_data(runtime_in_lay, node_in_lay, shape_info_ptr, offset);
    }

    if (params.typed_desc<kv_cache>()->indirect) {
        auto& var = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCache&>(get_network().get_variable(variable_id()));
        const auto& bt_state = var.get_beam_table_state();
        auto bt_layout = bt_state->get_layout();
        if (bt_layout.is_dynamic()) {
            auto bt_shape = bt_layout.get_partial_shape();
            for (auto& d : bt_shape) {
                if (d.is_dynamic())
                    d = 0;
            }
            bt_layout.set_partial_shape(bt_shape);
        }

        GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for input[" << i << "]" << std::endl;
        fill_shape_info_data(bt_layout, bt_state->get_initial_layout(), shape_info_ptr, offset);
    }

    for (size_t i = 0; i < get_node().get_output_layouts().size(); i++) {
        GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for output[" << i << "]" << std::endl;
        const auto& node_out_lay = get_node().get_output_layout(i);
        const auto& runtime_out_lay = params.output_layouts[i];
        fill_shape_info_data(runtime_out_lay, node_out_lay, shape_info_ptr, offset);
    }
}

void kv_cache_inst::release_variable() {
    // if there's variable state, it should hold a reference of tensor same as outputs
    if (!get_network().has_variable(variable_id()))
        return;
    for (size_t i = 0; i < _outputs.size(); ++i) {
        auto& output = _outputs[i];
        output.reset();
    }
}

stateless_kv_inst::typed_primitive_inst(network& network, const stateless_kv_node& node) : parent{network, node, false} {
    update_output_memory();
}

int64_t stateless_kv_inst::compute_update_offset(const kernel_impl_params& impl_param, const stateless_kv& desc) {
    const auto mem_dep_it = impl_param.memory_deps.find(2);
    if (mem_dep_it == impl_param.memory_deps.end())
        return 0;

    const auto& present_seq_len_mem = mem_dep_it->second;
    const auto present_seq_len_layout = present_seq_len_mem->get_layout();
    if (present_seq_len_layout.count() == 0)
        return 0;

    OPENVINO_ASSERT(present_seq_len_layout.count() == 1);
    cldnn::mem_lock<uint8_t, mem_lock_type::read> present_seq_len_mem_lock(present_seq_len_mem, impl_param.get_stream());
    auto present_seq_len_tensor = make_tensor(present_seq_len_layout, present_seq_len_mem_lock.data());
    const auto present_dim_updated = ov::get_tensor_data_as<int64_t>(present_seq_len_tensor);

    const auto& past_layout = impl_param.get_input_layout(0);
    const auto past_shape = past_layout.get_partial_shape();
    const auto past_sequence_axis = kv_cache_inst::get_sequence_axis(desc.concat_axis, past_shape.size());
    OPENVINO_ASSERT(past_sequence_axis >= 0);
    const auto& past_dim = past_shape[static_cast<size_t>(past_sequence_axis)];
    OPENVINO_ASSERT(past_dim.is_static());
    GPU_DEBUG_TRACE_DETAIL << desc.id << " : present_len[" << present_dim_updated[0] << "] past_len[" << past_dim.get_length() << "] "
                           << (present_dim_updated[0] <= past_dim.get_length() ? "update" : "concat") << std::endl;
    // OPENVINO_ASSERT(present_dim_updated[0] <= past_dim.get_length(), "[GPU] present_seq_length shouldn't exceed max_seq_length");

    const auto& current_layout = impl_param.get_input_layout(1);
    const auto current_shape = current_layout.get_partial_shape();
    const auto current_sequence_axis = kv_cache_inst::get_sequence_axis(desc.concat_axis, current_shape.size());
    OPENVINO_ASSERT(current_sequence_axis >= 0);
    const auto& current_dim = current_shape[static_cast<size_t>(current_sequence_axis)];
    OPENVINO_ASSERT(current_dim.is_static());

    const auto update_offset = present_dim_updated[0] - current_dim.get_length();
    OPENVINO_ASSERT(update_offset >= 0, "[GPU] new_token_data shouldn't exceed present_seq_length");

    return update_offset;
}

layout stateless_kv_inst::calc_output_layout(const stateless_kv_node& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param).front();
}

template<typename ShapeType>
std::vector<layout> stateless_kv_inst::calc_output_layouts(stateless_kv_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<stateless_kv>();

    std::vector<ShapeType> input_shapes = {impl_param.get_input_layout(0).get<ShapeType>(),
                                           impl_param.get_input_layout(1).get<ShapeType>()};
    const auto concat_axis = ov::util::normalize(desc->concat_axis, input_shapes[0].size());
    OPENVINO_ASSERT(concat_axis >= 0 && static_cast<size_t>(concat_axis) < input_shapes[0].size(), "[GPU] concat_axis exceed range");
    GPU_DEBUG_TRACE_DETAIL << desc->id << " : input[" << input_shapes[0] << "][" << input_shapes[1] << "]" << std::endl;

    ov::intel_gpu::op::StatelessKV op;
    op.set_output_size(2);
    op.set_concat_axis(concat_axis);
    op.set_update_offset(stateless_kv_inst::compute_update_offset(impl_param, *desc));

    auto output_shapes = shape_infer(&op, input_shapes);
    int64_t padding = 0;
    if (output_shapes[0][concat_axis].is_static() && output_shapes[1][concat_axis].is_static()) {
        padding = output_shapes[0][concat_axis].get_length() - output_shapes[1][concat_axis].get_length();
        OPENVINO_ASSERT(padding >= 0);
    }
    GPU_DEBUG_TRACE_DETAIL << desc->id << " : output[" << output_shapes[0] << "][" << output_shapes[1] << "] padding: " << padding << std::endl;

    std::vector<layout> out_layouts;
    out_layouts.emplace_back(output_shapes[0], impl_param.get_input_layout(0).data_type, impl_param.get_output_layout(0).format);
    out_layouts.emplace_back(output_shapes[1], impl_param.get_input_layout(0).data_type, impl_param.get_output_layout(0).format);
    padding::DynamicDimsMask seq_padding_info;
    seq_padding_info[concat_axis] = 1;
    out_layouts[1].data_padding._dynamic_dims_mask = seq_padding_info;
    out_layouts[1].data_padding._upper_size[concat_axis] = padding;

    return out_layouts;
}

template std::vector<layout> stateless_kv_inst::calc_output_layouts<ov::PartialShape>(stateless_kv_node const& node, const kernel_impl_params& impl_param);

std::string stateless_kv_inst::to_string(const stateless_kv_node& node) {
    auto node_info = node.desc_to_json();
    json_composite stateless_kv_info;
    stateless_kv_info.add("input id", node.input().id());
    stateless_kv_info.add("concat axis", node.get_primitive()->concat_axis);
    node_info->add("stateless_kv info", stateless_kv_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

void stateless_kv_inst::update_output_memory() {
    if (_node != nullptr)
        build_deps();

    if (input_memory_ptr() == nullptr || _outputs.empty())
        return;

    OPENVINO_ASSERT(_outputs.size() == 2);
    if (!_outputs[0])
        return;

    auto& engine = _network.get_engine();
    OPENVINO_ASSERT(_outputs[1], "[GPU] output1 should be available when output0 is present");
    OPENVINO_ASSERT(engine.is_the_same_buffer(output_memory(0), output_memory(1)), "[GPU] output1 should be same tensor with output0");
    m_is_inplace = engine.is_the_same_buffer(output_memory(), input_memory());
    GPU_DEBUG_TRACE_DETAIL << id() << ": update_output_memory in[" << input_memory().get_layout().to_short_string() << "] out["
                           << output_memory(0).get_layout().to_short_string() << "][" << output_memory(1).get_layout().to_short_string() << "] inplace["
                           << (m_is_inplace ? 'Y' : 'N') << "]" << std::endl;
    _mem_allocated = false;
}

void stateless_kv_inst::on_execute() {
    update_output_memory();
    set_arguments();
}

} // namespace cldnn
