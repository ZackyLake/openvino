// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_fusion.hpp"
#include <memory>
#include <optional>

#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/opsets/opset8_decl.hpp"
#include "openvino/core/graph_util.hpp"

namespace ov::intel_gpu {

KVCacheFusionMatcher::KVCacheFusionMatcher() {
    using namespace ov::pass::pattern;

    auto past = wrap_type<ov::op::v6::ReadValue>();
    auto convert_past = wrap_type<ov::op::v0::Convert>({past});
    auto gather_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past, convert_past});
    auto beam_idx = wrap_type<ov::op::v0::Parameter>();
    auto gather_past = wrap_type<ov::op::v8::Gather>({gather_input, beam_idx, wrap_type<ov::op::v0::Constant>()});
    auto gather_convert = wrap_type<ov::op::v0::Convert>({gather_past});
    auto dst_idx = wrap_type<ov::op::v0::Parameter>();
    auto gather_update = wrap_type<ov::op::v8::Gather>(); 
    auto update_kv = wrap_type<ov::op::v3::ScatterElementsUpdate>({gather_input, dst_idx, gather_update, wrap_type<ov::op::v0::Constant>()});
    auto start = wrap_type<ov::op::v0::Constant>();
    auto past_seq_len = any_input();
    auto stride = wrap_type<ov::op::v0::Constant>();
    auto step = wrap_type<ov::op::v0::Constant>();
    auto slice_axes = wrap_type<ov::op::v0::Constant>();
    auto trim_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{gather_input, gather_past, gather_convert, update_kv});
    auto trim_past = wrap_type<ov::op::v8::Slice>({trim_input, start, past_seq_len, step, slice_axes});
    auto trim_past2 = wrap_type<ov::op::v1::StridedSlice>({trim_input, start, past_seq_len, stride});
    auto concat_past_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{trim_input, trim_past, trim_past2});
    auto concat = wrap_type<ov::op::v0::Concat>({concat_past_input, any_input()});
    auto convert_present = wrap_type<ov::op::v0::Convert>({concat});
    auto present_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{concat, convert_present});
    auto present = wrap_type<ov::op::v6::Assign>({present_input});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto concat_node = ov::as_type_ptr<ov::op::v0::Concat>(pattern_map.at(concat).get_node_shared_ptr());

        auto past_node = ov::as_type_ptr<ov::op::v6::ReadValue>(pattern_map.at(past).get_node_shared_ptr());
        auto present_node = ov::as_type_ptr<ov::op::v6::Assign>(pattern_map.at(present).get_node_shared_ptr());

        if (past_node->get_variable_id() != present_node->get_variable_id())
            return false;

        // TODO: Support conversion internally
        if (ov::is_type<ov::opset8::Gather>(concat_past_input)) {
            if (!concat_node || concat_node->get_output_element_type(0) != past_node->get_output_element_type(0))
                return false;
        }

        auto variable = past_node->get_variable();
        auto concat_axis = concat_node->get_axis();

        std::shared_ptr<ov::Node> variable_initializer = nullptr;
        std::shared_ptr<ov::Node> kv_cache_node = nullptr;
        if (past_node->get_input_size() == 1) {
            variable_initializer = past_node->get_input_node_shared_ptr(0);
        }

        // Replace common ReadValue op with a custom one as common one expects paired Assign operation which is removed by this transform
        auto new_read_value_node = variable_initializer ? std::make_shared<ov::intel_gpu::op::ReadValue>(variable_initializer->output(0), variable)
                                                        : std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        new_read_value_node->set_friendly_name(past_node->get_friendly_name());
        ov::copy_runtime_info(past_node, new_read_value_node);
        ov::replace_node(past_node, new_read_value_node);

        const bool has_beam_idx = pattern_map.count(gather_past) > 0;
        const bool has_update_kv = pattern_map.count(update_kv) > 0;
        const bool has_slice = pattern_map.count(trim_past) > 0;
        const bool has_strided_slice = pattern_map.count(trim_past2) > 0;
        const bool has_trim = has_slice || has_strided_slice;

        const auto adjust_axis_to_positive = [&new_read_value_node](auto axis) ->std::optional<uint64_t> {
            if (axis >= 0) {
                return static_cast<uint64_t>(axis);
            }
            const auto input_rank = new_read_value_node->get_output_partial_shape(0).rank();
            if (input_rank.is_static()) {
                const auto adjusted_axis = input_rank.get_interval().get_min_val() + axis;
                if (adjusted_axis >= 0) {
                    return static_cast<uint64_t>(adjusted_axis);
                }
            }
            return std::nullopt;
        };
        std::optional<uint64_t> target_concat_axis = adjust_axis_to_positive(concat_axis);
        OPENVINO_ASSERT(target_concat_axis.has_value(), "concat_axis should be valid, get: ", concat_axis);

        std::shared_ptr<ov::Node> past_seq_len_node;
        if (has_trim) {
            past_seq_len_node = pattern_map.at(past_seq_len).get_node_shared_ptr();
            // StridedSlice uses multi-dim for end tensor, extract only the slice dim
            if (has_strided_slice) {
                const auto strided_slice = ov::as_type_ptr<ov::op::v1::StridedSlice>(concat_node->input_value(0).get_node_shared_ptr());
                if (!strided_slice)
                    return false;
                const auto begin_mask = strided_slice->get_begin_mask();
                const auto end_mask = strided_slice->get_end_mask();
                // begin/end mask should be the same and only last element is 0 (being sliced)
                if (begin_mask != end_mask || begin_mask.empty()) {
                    return false;
                }
                if (static_cast<size_t>(std::count(begin_mask.begin(), begin_mask.end(), 1)) != (begin_mask.size() - 1) || begin_mask.back() != 0) {
                    return false;
                }
                // slice start should be all 0 and stride should be all 1
                const auto slice_start = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(start).get_node_shared_ptr());
                if (const auto start_data = slice_start->cast_vector<int64_t>(); std::any_of(start_data.begin(), start_data.end(), [](const auto val) {
                        return val != 0;
                    })) {
                    return false;
                }
                const auto slice_stride = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(stride).get_node_shared_ptr());
                if (const auto stride_data = slice_stride->cast_vector<int64_t>(); std::any_of(stride_data.begin(), stride_data.end(), [](const auto val) {
                        return val != 1;
                    })) {
                    return false;
                }
                // sliced axis should be the same with concat_axis
                if (begin_mask.size() != *target_concat_axis + 1) {
                    return false;
                }
                const auto slice_axis = ov::op::v0::Constant::create(element::i64, Shape{1}, {concat_axis});
                const auto gather_axis = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
                past_seq_len_node = std::make_shared<ov::op::v8::Gather>(past_seq_len_node, slice_axis, gather_axis);
            } else {
                // slice start should be 0 and step should be 1
                const auto slice_start = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(start).get_node_shared_ptr());
                if (const auto start_data = slice_start->cast_vector<int64_t>(); start_data.size() != 1 || start_data[0] != 0) {
                    return false;
                }
                const auto slice_step = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(step).get_node_shared_ptr());
                if (const auto step_data = slice_step->cast_vector<int64_t>(); step_data.size() != 1 || step_data[0] != 1) {
                    return false;
                }
                // slice axis should be the same as concat_axis
                const auto slice_axis = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(slice_axes).get_node_shared_ptr());
                if (const auto axis_data = slice_axis->cast_vector<int64_t>();
                    axis_data.size() != 1 || adjust_axis_to_positive(axis_data[0]) != *target_concat_axis) {
                    return false;
                }
            }
        }

        static const auto env_updkv = std::getenv("updatekv");
        static const auto noupdkv = env_updkv && std::string_view("false") == env_updkv;
        printf("@@##kv fusion: [%s]: beam[%c] trim[%c] update[%c]\n",
               past_node->get_variable_id().c_str(),
               has_beam_idx ? 'Y' : 'N',
               has_trim ? 'T' : 'N',
               has_update_kv ? (noupdkv ? 'O' : 'Y') : 'N');
        if (has_update_kv) {
            const auto slice_node = pattern_map.at(update_kv).get_node_shared_ptr();
            printf("----here updatekv: [%s](%s)\n", slice_node->get_friendly_name().c_str(), 
                pattern_map.at(past_seq_len).get_node_shared_ptr()->get_friendly_name().c_str());
        }
        
        const auto input0 = has_beam_idx ? pattern_map.at(gather_past).get_node_shared_ptr() : new_read_value_node;
        if (has_update_kv && !noupdkv) {
            OPENVINO_ASSERT(has_trim);
            kv_cache_node = std::make_shared<op::KVCache>(input0,
                                                          concat_node->input(1).get_source_output(),
                                                          past_seq_len_node,
                                                          pattern_map.at(dst_idx).get_node_shared_ptr(),
                                                          pattern_map.at(gather_update).get_node_shared_ptr(),
                                                          variable,
                                                          concat_axis,
                                                          new_read_value_node->get_output_element_type(0));
        } else if (has_trim) {
            kv_cache_node = std::make_shared<op::KVCache>(input0,
                                                          concat_node->input(1).get_source_output(),
                                                          past_seq_len_node,
                                                          variable,
                                                          concat_axis,
                                                          new_read_value_node->get_output_element_type(0));
        } else {
            kv_cache_node = std::make_shared<op::KVCache>(input0,
                                                          concat_node->input(1).get_source_output(),
                                                          variable,
                                                          concat_axis,
                                                          new_read_value_node->get_output_element_type(0));
        }
        kv_cache_node->set_friendly_name(concat_node->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), kv_cache_node);
        ov::replace_node(concat_node, kv_cache_node);

        present_node->set_argument(0, kv_cache_node->output(0));

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(present, "KVCacheFusionMatcher");
    this->register_matcher(m, callback);
}

bool KVCacheFusion::run_on_model(const std::shared_ptr<ov::Model>& m) {
    bool res = pass::GraphRewrite::run_on_model(m);
    if (res) {
        ov::SinkVector sinks = m->get_sinks();
        for (auto& sink : sinks) {
            if (sink && sink->get_input_node_ptr(0)->get_type_info() == op::KVCache::get_type_info_static()) {
                m->remove_sink(sink);
            }
        }
    }

    return res;
}

KVCacheFusion::KVCacheFusion() {
    add_matcher<ov::intel_gpu::KVCacheFusionMatcher>();
}

StatelessKVFusionMatcher::StatelessKVFusionMatcher() {
    using namespace ov::pass::pattern;
    using namespace ov::op;

    auto past = wrap_type<ov::op::v0::Parameter>();
    auto new_token_data = any_input();

    auto total_seqlen = wrap_type<ov::op::v0::Parameter>(shape_matches("[1]"));
    auto total_seqlen_cvt = wrap_type<ov::op::v0::Convert>({total_seqlen});
    auto total_seqlen_actual = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{total_seqlen, total_seqlen_cvt});
    auto seqlens_k = wrap_type<ov::op::v0::Parameter>(shape_matches("[1,1]"));
    auto seqlens_k_cvt = wrap_type<ov::op::v0::Convert>({seqlens_k});
    auto seqlens_k_actual = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{seqlens_k, seqlens_k_cvt});
    auto real_seqlens = wrap_type<ov::op::v1::Add>({seqlens_k_actual, 1});
    auto seqlens_1d = wrap_type<ov::op::v1::Reshape>({real_seqlens, 1});
    auto concat_kv_len = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{total_seqlen, total_seqlen_cvt, seqlens_1d});

    auto cur_seqlen_shapeof = wrap_type<ov::op::v3::ShapeOf>({any_input()});
    auto seqlen_dim = wrap_type<ov::op::v0::Constant>(shape_matches("[1]"));
    auto cur_seqlen = wrap_type<ov::op::v8::Gather>({cur_seqlen_shapeof, seqlen_dim, 0});
    auto cur_seqlen_neg = wrap_type<ov::op::v1::Multiply>({cur_seqlen, -1});
    auto cur_seqlen_neg_const = wrap_type<ov::op::v0::Constant>(shape_matches("[?]"));
    auto past_seqlen_add =
        wrap_type<ov::op::v1::Add>({concat_kv_len, std::make_shared<ov::pass::pattern::op::Or>(OutputVector{cur_seqlen_neg, cur_seqlen_neg_const})});
    auto past_seqlen_sub = wrap_type<ov::op::v1::Subtract>({concat_kv_len, cur_seqlen});

    auto range_cur = wrap_type<ov::op::v4::Range>({0, any_input(), 1});
    auto const_range_cur = wrap_type<ov::op::v0::Constant>(shape_matches("[?]"));
    auto pos_idx_base = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{range_cur, const_range_cur});
    auto past_seqlen_from_param = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past_seqlen_add, past_seqlen_sub, any_input()});
    auto shifted_pos_idx = wrap_type<ov::op::v1::Add>({pos_idx_base, past_seqlen_from_param});
    auto pos_idx = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{shifted_pos_idx, any_input()});
    auto scatter_axis = wrap_type<ov::op::v0::Constant>(shape_matches("[1]"));
    auto scatter_update = wrap_type<ov::op::v3::ScatterUpdate>({past, pos_idx, new_token_data, scatter_axis});

    auto slice_axis = wrap_type<ov::op::v0::Constant>();
    auto past_seqlen_actual = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past_seqlen_add, past_seqlen_sub, any_input()});
    auto slice = wrap_type<ov::op::v8::Slice>({past, 0, past_seqlen_actual, 1, slice_axis});
    auto concat = wrap_type<ov::op::v0::Concat>({slice, new_token_data});

    auto kv_actual = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{scatter_update, concat});
    auto result = wrap_type<ov::op::v0::Result>({kv_actual});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        static const auto env_slkv = std::getenv("statelesskv");
        static const auto noslkv = env_slkv && std::string_view("false") == env_slkv;
        if (noslkv) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto result_node = ov::as_type_ptr<ov::op::v0::Result>(pattern_map.at(result).get_node_shared_ptr());
        const auto result_input = result_node->input(0);
        const auto kv_present_output = result_input.get_source_output();
        const auto past_output = pattern_map.at(past);
        const auto new_token_output = pattern_map.at(new_token_data);
        const auto new_token_shape = new_token_output.get_partial_shape();
        std::optional<ov::Input<ov::Node>> kv_to_sdpa_input;
        std::optional<ov::Input<ov::Node>> kv_sdpa_input;
        ov::Output<ov::Node> pos_idx_output;
        ov::Output<ov::Node> seqlen_output;
        ov::NodeVector node_infos;
        int64_t target_axis = 0;
        bool is_present_len = true;

        const bool is_slice_concat = pattern_map.count(concat) > 0;
        bool is_update_split = false;

        const auto kv_present_consumers = kv_present_output.get_target_inputs();
        if (kv_present_consumers.size() != 2) {
            return false;
        }
        for (const auto& input : kv_present_consumers) {
            if (input != result_input) {
                kv_to_sdpa_input.emplace(input);
            }
        }
        OPENVINO_ASSERT(kv_to_sdpa_input.has_value());

        std::optional<int64_t> shapeof_axis;
        if (pattern_map.count(seqlen_dim) > 0) {
            auto seqlen_dim_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(seqlen_dim).get_node_shared_ptr());
            shapeof_axis = seqlen_dim_node->cast_vector<int64_t>()[0];
        }
        std::optional<int64_t> neg_cur_seqlen;
        if (pattern_map.count(cur_seqlen_neg_const) > 0) {
            auto cur_neg_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(cur_seqlen_neg_const).get_node_shared_ptr());
            neg_cur_seqlen = cur_neg_node->cast_vector<int64_t>()[0];
        }
        static const auto gqareuse = []() {
            const auto txt = std::getenv("gqareuse");
            return !(txt && txt == std::string_view("false"));
        }();
        ov::Output<ov::Node> total_seqlen_output;
        if (pattern_map.count(total_seqlen) > 0) {
            total_seqlen_output = pattern_map.at(total_seqlen);
        }
        ov::Output<ov::Node> seqlens_k_output;
        if (pattern_map.count(seqlens_k) > 0) {
            seqlens_k_output = pattern_map.at(seqlens_k);
        }
        ov::Output<ov::Node> concat_kvlen_output;
        if (pattern_map.count(concat_kv_len) > 0) {
            concat_kvlen_output = pattern_map.at(concat_kv_len);
        }
        std::shared_ptr<CachedNodes> cache;
        if (total_seqlen_output.get_node()) {
            if (const auto it = m_cache.find(total_seqlen_output.get_node_shared_ptr()); it != m_cache.end()) {
                cache = it->second;
            }
        }
        if (!cache && seqlens_k_output.get_node()) {
            if (const auto it = m_cache.find(seqlens_k_output.get_node_shared_ptr()); it != m_cache.end()) {
                cache = it->second;
            }
        }
        if (!cache && concat_kvlen_output.get_node()) {
            if (const auto it = m_cache.find(concat_kvlen_output.get_node_shared_ptr()); it != m_cache.end()) {
                cache = it->second;
            }
        }
        if (!cache) {
            cache = std::make_shared<CachedNodes>();
        }
        if (total_seqlen_output.get_node() && !cache->total_seqlen) {
            cache->total_seqlen = total_seqlen_output.get_node_shared_ptr();
            if (!cache->present_kv_len.get_node()) {
                cache->present_kv_len = total_seqlen_output;
            }
            m_cache.try_emplace(total_seqlen_output.get_node_shared_ptr(), cache);
        }
        if (seqlens_k_output.get_node() && !cache->seqlens_k) {
            cache->seqlens_k = seqlens_k_output.get_node_shared_ptr();
            if (!cache->present_kv_len.get_node()) {
                cache->present_kv_len = pattern_map.at(real_seqlens); //  skip reshape to pass correct shape-inder dependency
            }
            m_cache.try_emplace(seqlens_k_output.get_node_shared_ptr(), cache);
        }
        if (concat_kvlen_output.get_node()) {
            if (!cache->present_kv_len.get_node()) {
                cache->present_kv_len = concat_kvlen_output;
            }
            m_cache.try_emplace(concat_kvlen_output.get_node_shared_ptr(), cache);
        }

        if (is_slice_concat) {  // original dynamic pattern
            auto slice_node = ov::as_type_ptr<ov::op::v8::Slice>(pattern_map.at(slice).get_node_shared_ptr());
            auto concat_node = ov::as_type_ptr<ov::op::v0::Concat>(pattern_map.at(concat).get_node_shared_ptr());
            auto slice_stop_node = pattern_map.at(past_seqlen_actual).get_node_shared_ptr();
            auto slice_axis_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(slice_axis).get_node_shared_ptr());

            if (slice_axis_node->cast_vector<int64_t>()[0] != concat_node->get_axis()) {
                return false;
            }
            target_axis = concat_node->get_axis();
            
            if (cache->present_kv_len.get_node()) {
                seqlen_output = cache->present_kv_len;
            } else {
                seqlen_output = slice_stop_node;
                is_present_len = false;
            }
            node_infos = {slice_node, concat_node};
        } else {
            auto scatter_axis_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(scatter_axis).get_node_shared_ptr());
            target_axis = scatter_axis_node->cast_vector<int64_t>()[0];
            auto update_node = ov::as_type_ptr<ov::op::v3::ScatterUpdate>(pattern_map.at(scatter_update).get_node_shared_ptr());
            pos_idx_output = pattern_map.at(pos_idx);
            node_infos.push_back(update_node);
            
            const auto split_node = ov::as_type_ptr<ov::op::v1::VariadicSplit>(kv_to_sdpa_input->get_node()->shared_from_this());
            if (split_node) {  // static pattern with variadic split
                is_update_split = true;
                if (kv_to_sdpa_input->get_index() != 0) {
                    return false;
                }

                auto split_axis_node = ov::as_type_ptr<ov::op::v0::Constant>(split_node->get_input_node_shared_ptr(1));
                auto split_lengths_node = ov::as_type_ptr<ov::op::v0::Concat>(split_node->get_input_node_shared_ptr(2));
                if (!split_axis_node || !split_lengths_node || split_lengths_node->get_input_size() != 2) {
                    return false;
                }

                if (target_axis != split_axis_node->cast_vector<int64_t>()[0]) {
                    return false;
                }

                auto split_tail_node =
                    ov::as_type_ptr<ov::op::v0::Constant>(split_lengths_node->get_input_node_shared_ptr(1));
                if (!split_tail_node) {
                    return false;
                }
                const auto split_tail_values = split_tail_node->cast_vector<int64_t>();
                if (split_tail_values.size() != 1 || split_tail_values[0] != -1) {
                    return false;
                }

                auto split_present_output = split_lengths_node->input_value(0);
                if (const auto split_present_reshape =
                        ov::as_type_ptr<ov::op::v1::Reshape>(split_present_output.get_node_shared_ptr())) {
                    split_present_output = split_present_reshape->input_value(0);
                }
                if (const auto plen_shape = split_present_output.get_partial_shape(); plen_shape.is_dynamic() || shape_size(plen_shape.get_shape()) != 1) {
                    return false;
                }
                seqlen_output = split_present_output;

                if (split_node->get_output_size() != 2 || !split_node->output(1).get_target_inputs().empty()) {
                    return false;
                }
                node_infos.push_back(split_node);
                const auto split_output_consumers = split_node->output(0).get_target_inputs();
                if (split_output_consumers.size() != 1) {
                    return false;
                }
                kv_sdpa_input.emplace(*split_output_consumers.begin());
            } else {  // original static pattern
                if (past_output.get_partial_shape().is_dynamic() || new_token_shape.is_dynamic()) {
                    return false;
                }

                if (pattern_map.count(shifted_pos_idx) == 0) {
                    return false;
                }

                ov::Output<ov::Node> past_seqlen_output = pattern_map.at(past_seqlen_from_param).get_node_shared_ptr();
                // make sure idx_* starts from 0 so that the other input is the past_seqlen_node
                if (pattern_map.count(const_range_cur) > 0) {
                    ov::op::v0::Constant* idx_const = ov::as_type<ov::op::v0::Constant>(pattern_map.at(const_range_cur).get_node());
                    const auto idx_data = idx_const->cast_vector<int64_t>();
                    if (idx_data.size() < 1) {
                        return false;
                    }
                    for (size_t i = 0; i < idx_data.size(); ++i) {
                        if (idx_data[i] != static_cast<int64_t>(i)) {
                            return false;
                        }
                    }
                } else {
                    // should already be garuanteed
                }

                if (cache->present_kv_len.get_node()) {
                    seqlen_output = cache->present_kv_len;
                } else {
                    seqlen_output = past_seqlen_output;
                    is_present_len = false;
                }
            }
        }

        if (shapeof_axis && *shapeof_axis != target_axis) {
            return false;
        }
        if (neg_cur_seqlen && (new_token_shape[target_axis].is_dynamic() || new_token_shape[target_axis].get_length() * -1 != *neg_cur_seqlen)) {
            return false;
        }

        if (!kv_sdpa_input) {
            kv_sdpa_input.emplace(*kv_to_sdpa_input);
        }

        auto kv_sdpa_node = kv_sdpa_input->get_node()->shared_from_this();
        std::shared_ptr<op::SDPA> sdpa_node = ov::as_type_ptr<op::SDPA>(kv_sdpa_node);
        if (sdpa_node && kv_sdpa_input->get_index() != 1 && kv_sdpa_input->get_index() != 2) {
            return false;
        }
        if (transformation_callback(kv_sdpa_node)) {
            return false;
        }

        auto get_trimmed_mask = [&](const ov::Output<ov::Node>& full_mask, const ov::Dimension& cur_seqlen) -> std::shared_ptr<ov::Node> {
            if (gqareuse && cache->present_kv_len.get_node() && cur_seqlen.is_static()) {
                for (const auto& [len, old_mask, new_mask] : cache->trimmed_masks) {
                    if (len != cur_seqlen.get_length())
                        continue;
                    if (old_mask == full_mask) {
                        return new_mask;
                    }
                }
            }
            return {};
        };
        // mask trimming for pure-scatter_update case
        if (!is_slice_concat && !is_update_split && sdpa_node && sdpa_node->inputs().size() == 4 &&
            m_trimmed_masks.count(sdpa_node->input_value(3)) == 0) {
            const auto full_mask = sdpa_node->input_value(3);
            const auto& cur_seqlen = new_token_shape[target_axis];
            auto trimmed_mask = get_trimmed_mask(full_mask, cur_seqlen);
            if (!trimmed_mask) {
                const auto present_len_type = seqlen_output.get_element_type();
                ov::Output<ov::Node> present_len = cache->present_kv_len;
                if (present_len.get_node()) {
                    if (present_len.get_partial_shape().rank().get_length() > 1) {
                        present_len = std::make_shared<v1::Reshape>(present_len, v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}), false);
                    }
                } else {
                    OPENVINO_ASSERT(!is_present_len);
                    std::shared_ptr<ov::Node> cur_seqlen_node;
                    if (cur_seqlen.is_static()) {
                        cur_seqlen_node = v0::Constant::create(present_len_type, ov::Shape{1}, {cur_seqlen.get_length()});
                    } else {
                        const auto zero_without_shape = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
                        auto new_token_shape = std::make_shared<v3::ShapeOf>(new_token_output, present_len_type);
                        cur_seqlen_node = std::make_shared<v8::Gather>(new_token_shape,
                                                                        v0::Constant::create(ov::element::i64, ov::Shape{1}, {target_axis}),
                                                                        zero_without_shape);
                    }
                    present_len = std::make_shared<v1::Add>(seqlen_output.get_node_shared_ptr(), cur_seqlen_node);
                }
                const auto split_lengths =
                    std::make_shared<v0::Concat>(ov::OutputVector{present_len, v0::Constant::create(present_len_type, ov::Shape{1}, {-1})}, 0);
                const auto mask_split =
                    std::make_shared<v1::VariadicSplit>(full_mask, v0::Constant::create(present_len_type, ov::Shape{}, {1}), split_lengths);
                trimmed_mask = mask_split;
                if (gqareuse && cache->present_kv_len.get_node() && cur_seqlen.is_static()) {
                    cache->trimmed_masks.push_back({cur_seqlen.get_length(), full_mask, trimmed_mask});
                }
            }
            printf("@@##statelesskv: mask-trim [%s] -> [%s]\n",
                    full_mask.get_node()->get_friendly_name().c_str(),
                    trimmed_mask->get_friendly_name().c_str());
            sdpa_node->set_argument(3, trimmed_mask->output(0));
            m_trimmed_masks.insert(trimmed_mask->output(0));
        }

        static const auto env_nopos = std::getenv("nopos");
        static const auto nopos = env_nopos && std::string_view("true") == env_nopos;
        std::string posidname;
        if (pos_idx_output.get_node()) {
            posidname = pos_idx_output.get_node()->get_friendly_name();
            if (nopos) {
                posidname += "(remove)";
            }
        }
        printf("@@##statelesskv(%s): [%s][%s] %s[%s] len[%s](%s) pos[%s] clen[%s]\n",
               is_slice_concat ? "SC" : (is_update_split ? "US" : "U"),
               past_output.get_any_name().c_str(),
               result_node->get_friendly_name().c_str(),
               sdpa_node ? "sdpa" : "next",
               kv_sdpa_node->get_friendly_name().c_str(),
               seqlen_output.get_node()->get_friendly_name().c_str(),
               is_present_len ? "present" : "past",
               posidname.c_str(),
               cache->present_kv_len.get_node() ? cache->present_kv_len.get_node()->get_friendly_name().c_str() : "");
        std::shared_ptr<op::StatelessKV> stateless_kv;
        if (nopos || !pos_idx_output.get_node()) {
            stateless_kv = std::make_shared<op::StatelessKV>(past_output, new_token_output, seqlen_output, target_axis, is_present_len);
        } else {
            stateless_kv = std::make_shared<op::StatelessKV>(past_output, new_token_output, seqlen_output, pos_idx_output, target_axis, is_present_len);
        }
        stateless_kv->set_friendly_name(past_output.get_any_name() + "_stateless");
        ov::copy_runtime_info(node_infos, stateless_kv);
        stateless_kv->output(0).set_names(result_node->output(0).get_names());

        kv_sdpa_input->replace_source_output(stateless_kv->output(1));
        result_node->input(0).replace_source_output(stateless_kv->output(0));

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, "StatelessKVFusionMatcher");
    this->register_matcher(m, callback);
}

bool StatelessKVFusion::run_on_model(const std::shared_ptr<ov::Model>& m) {
    return pass::GraphRewrite::run_on_model(m);
}

StatelessKVFusion::StatelessKVFusion() {
    add_matcher<ov::intel_gpu::StatelessKVFusionMatcher>();
}

}  // namespace ov::intel_gpu
