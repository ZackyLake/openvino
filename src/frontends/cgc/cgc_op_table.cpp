// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cgc_conversion.h"

#include "openvino/core/type/element_type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"

#include <cmath>
#include <cstdint>

namespace ov {
namespace frontend {
namespace cgc {

using namespace ov::op;

template<typename T>
static std::shared_ptr<ov::Node> make_scalar_const(const T& data) {
    constexpr auto type = ov::element::from<T>();
    return v0::Constant::create(type, ov::Shape{}, {data});
}

template <typename T>
static std::shared_ptr<ov::Node> make_array_const(const std::vector<T>& data) {
    constexpr auto type = ov::element::from<T>();
    return v0::Constant::create(type, ov::Shape{data.size()}, data);
}

static std::shared_ptr<ov::Node> gather_dimensions(const ov::Output<ov::Node>& value, const std::vector<int32_t>& dims) {
    static const auto zero = make_scalar_const<int32_t>(0);
    const auto dims_const = make_array_const(dims);
    return std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(value), dims_const, zero);
}

static int64_t normalize_axis(int64_t axis, int64_t rank) {
    OPENVINO_ASSERT(rank > 0);
    if (axis < 0) {
        OPENVINO_ASSERT(rank >= -axis);
        return rank + axis;
    } else {
        OPENVINO_ASSERT(rank > axis);
        return axis;
    }
}

class NullNode : public ov::op::Op {
public:
    OPENVINO_OP("NullNode");
    NullNode() {
        set_output_size(1);
    }
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<NullNode>();
    }
};

static ov::Output<ov::Node> get_null_connection() {
    static const auto node = std::make_shared<NullNode>();
    return node;
}


// ===================================================================
// cgc_op.transpose
//   input 0 : data
//   attr "permutation" : vector<int64_t>
// ===================================================================
static ov::OutputVector translate_transpose(const CgcNodeContext& ctx,
                                            const ov::OutputVector& inputs,
                                            const NodeOutputInfos&) {
    auto data = inputs[0];
    auto perm = ctx.get_attribute<std::vector<int64_t>>("permutation");
    auto perm_const = make_array_const(perm);
    auto node = std::make_shared<v1::Transpose>(data, perm_const);
    node->set_friendly_name(ctx.get_op_name());
    return node->outputs();
}

// ===================================================================
// cgc_op.gemm
//   input 0 : A
//   input 1 : B
//   input 2 : C
//
// Maps to v0::MatMul on the last two dims.
// If C is present, adds bias via Add.
// ===================================================================
static ov::OutputVector translate_gemm(const CgcNodeContext& ctx,
                                       const ov::OutputVector& inputs,
                                       const NodeOutputInfos&) {
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    const auto& c = inputs[2];
    std::shared_ptr<ov::Node> ret = std::make_shared<v0::MatMul>(a, b, false, false);
    ret->set_friendly_name(ctx.get_op_name());

    if (c.get_node()) {
        ret = std::make_shared<v1::Add>(ret, c);
        ret->set_friendly_name(ctx.get_op_name() + "/add_bias");
    }
    return ret->outputs();
}

// return reshaped shape for scale/zp if needed
static std::vector<int64_t> VerifyQDQ(const ov::Output<ov::Node>& data,
                                           const ov::Output<ov::Node>& scale,
                                           const ov::Output<ov::Node>& zero_point,
                                           const ov::element::Type quant_type,
                                           const ov::element::Type dequant_type,
                                           const int64_t axis,
                                           const int64_t block_size) {
    const auto in_shape = data.get_partial_shape();
    const auto in_rank = in_shape.rank();
    OPENVINO_ASSERT(in_rank.is_static());
    const auto in_rank_ = in_rank.get_length();

    const auto scale_shape = scale.get_partial_shape();
    const auto scale_rank = scale_shape.rank();
    OPENVINO_ASSERT(scale_rank.is_static());
    const auto scale_rank_ = scale_rank.get_length();
    OPENVINO_ASSERT(scale.get_element_type() == dequant_type);

    if (zero_point.get_node()) {
        const auto zp_shape = zero_point.get_partial_shape();
        const auto zp_rank = zp_shape.rank();
        OPENVINO_ASSERT(zp_rank.is_static());
        const auto zp_rank_ = zp_rank.get_length();
        OPENVINO_ASSERT(scale_rank_ == zp_rank_);
        OPENVINO_ASSERT(zp_shape == scale_shape);
        OPENVINO_ASSERT(zero_point.get_element_type() == quant_type);
    }

    if (in_rank == 1) {
        // only per-tensor allowed
        OPENVINO_ASSERT(block_size == 0);
        OPENVINO_ASSERT(scale_rank_ == 1);
        OPENVINO_ASSERT(scale_shape[0].get_length() == 1);
        // ignoring axis even if it may exceed rank
    } else {
        // do not support per-block for now
        OPENVINO_ASSERT(block_size == 0);
        OPENVINO_ASSERT(scale_rank_ == 1);
        const auto target_dim = normalize_axis(axis, in_rank_);
        const auto in_dim_size = in_shape[target_dim].get_length();
        OPENVINO_ASSERT(scale_shape[0].get_length() == 1 || scale_shape[0].get_length() == in_dim_size);
        if (target_dim != in_rank_ - 1) {
            std::vector<int64_t> reshape_dims(in_rank_, 1);
            reshape_dims[target_dim] = scale_shape[0].get_length(); // -1
            return reshape_dims;
        }
    }
    return {scale_shape[0].get_length()};
}

template<typename T>
static std::optional<T> try_get_constant(v0::Constant* node) {
    if (node) {
        const auto vec = node->cast_vector<T>(1);
        return vec[0];
    }
    return {};
}
template <typename T>
static std::optional<T> try_get_constant(ov::Node* node) {
    if (node) {
        return try_get_constant<T>(ov::as_type<v0::Constant>(node));
    }
    return {};
}

static std::tuple<std::shared_ptr<v0::Constant>, std::shared_ptr<v0::Constant>> get_output_bounds(
    const ov::element::Type& type,
    const ov::element::Type& node_dtype) {
    switch (type) { 
    case ov::element::i4:
        return {
            std::make_shared<v0::Constant>(node_dtype, ov::Shape{}, -8),
            std::make_shared<v0::Constant>(node_dtype, ov::Shape{}, 7),
        };
    case ov::element::u4:
        return {
            std::make_shared<v0::Constant>(node_dtype, ov::Shape{}, 0),
            std::make_shared<v0::Constant>(node_dtype, ov::Shape{}, 15),
        };
    case ov::element::i8:
        return {
            std::make_shared<v0::Constant>(node_dtype, ov::Shape{}, -128),
            std::make_shared<v0::Constant>(node_dtype, ov::Shape{}, 127),
        };
    case ov::element::u8:
        return {
            std::make_shared<v0::Constant>(node_dtype, ov::Shape{}, 0),
            std::make_shared<v0::Constant>(node_dtype, ov::Shape{}, 255),
        };
    case ov::element::i16:
        return {
            std::make_shared<v0::Constant>(node_dtype, ov::Shape{}, -32768),
            std::make_shared<v0::Constant>(node_dtype, ov::Shape{}, 32767),
        };
    case ov::element::u16:
        return {
            std::make_shared<v0::Constant>(node_dtype, ov::Shape{}, 0),
            std::make_shared<v0::Constant>(node_dtype, ov::Shape{}, 65535),
        };
    default:
        OPENVINO_ASSERT(false, "unsupported qlinear type");
        return {};
    }
}
static std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> get_input_bounds(
    std::shared_ptr<ov::Node> low,
    std::shared_ptr<ov::Node> high,
    ov::Output<ov::Node> zero_point, 
    ov::Output<ov::Node> scale,
    const std::vector<int64_t>& shape) {
    if (zero_point.get_node()) {
        auto zp = std::make_shared<v0::Convert>(zero_point, low->get_output_element_type(0));
        low = std::make_shared<v1::Subtract>(low, zp);
        high = std::make_shared<v1::Subtract>(high, zp);
    }
    if (scale.get_element_type() != low->get_output_element_type(0)) {
        scale = std::make_shared<v0::Convert>(scale, low->get_output_element_type(0));
    }
    low = std::make_shared<v1::Multiply>(low, scale);
    high = std::make_shared<v1::Multiply>(high, scale);
    const auto reshape = make_array_const(shape);
    low = std::make_shared<v1::Reshape>(low, reshape, false);
    high = std::make_shared<v1::Reshape>(high, reshape, false);
    if (auto constant = ov::util::get_constant_from_source(low)) {
        low = constant;
    }
    if (auto constant = ov::util::get_constant_from_source(high)) {
        high = constant;
    }
    return {low, high};
}

// ===================================================================
// cgc_op.dequantize_linear
//   input 0 : data (quantized)
//   input 1 : scale
//   input 2 : zero_point
//   attr "axis" : int64_t
//   attr "block_size" : int64_t
//
// Semantics: output = (data - zero_point) * scale
// Mapped as:  Convert(data) -> Subtract(zero_point) -> Multiply(scale)
// ===================================================================
static ov::OutputVector translate_dequantize_linear(const CgcNodeContext& ctx,
                                                    const ov::OutputVector& inputs,
                                                    const NodeOutputInfos& outputs) {
    auto data = inputs[0];
    auto scale = inputs[1];
    auto zero_point = inputs[2];
    const auto axis = ctx.get_attribute<int64_t>("axis");
    const auto block_size = ctx.get_attribute<int64_t>("block_size");
    OPENVINO_ASSERT(outputs.size() == 1);
    const auto dequant_type = outputs[0].first;

    const auto scale_const = try_get_constant<float>(scale.get_node());
    const auto zp_const = try_get_constant<int32_t>(zero_point.get_node());

    if (dequant_type != scale.get_element_type()) {
        scale = std::make_shared<v0::Convert>(scale, dequant_type);
    }

    const auto reshape_shape =
        VerifyQDQ(data, scale, zero_point, data.get_element_type(), dequant_type, axis, block_size);

    data = std::make_shared<v0::Convert>(data, dequant_type);
    if (zero_point.get_node()) {
        zero_point = std::make_shared<v0::Convert>(zero_point, dequant_type);
    }
    if (reshape_shape.size() > 1) {
        scale = std::make_shared<v1::Reshape>(scale, make_array_const(reshape_shape), false);
        if (zero_point.get_node()) {
            zero_point = std::make_shared<v1::Reshape>(zero_point, make_array_const(reshape_shape), false);
        }
    }

    auto sub = std::make_shared<v1::Subtract>(data, zero_point);
    auto mul = std::make_shared<v1::Multiply>(sub, scale);
    mul->set_friendly_name(ctx.get_op_name());

    auto [in_low, in_high] = get_output_bounds(inputs[0].get_element_type(), ov::element::i32);
    const auto in_low_const = in_low->get_vector<int32_t>()[0], in_high_const = in_high->get_vector<int32_t>()[0];
    std::optional<float> out_low, out_high;
    if (scale_const && zp_const) {
        out_low = (in_low_const - *zp_const) * *scale_const;
        out_high = (in_high_const - *zp_const) * *scale_const;
    }
    printf("deqlinear[%s]: scale[%s] zp[%s] -> [%s ~ %s]\n",
           ctx.get_op_name().c_str(),
           scale_const ? std::to_string(*scale_const).c_str() : "?",
           zp_const ? std::to_string(*zp_const).c_str() : "?",
           out_low ? std::to_string(*out_low).c_str() : "?",
           out_high ? std::to_string(*out_high).c_str() : "?");
    return mul->outputs();
}


// ===================================================================
// cgc_op.quantize_linear
//   input 0 : data (i32|f32|f16)
//   input 1 : scale (f32|f16)
//   input 2 : zero_point (null|i16|u16|i8|u8|i4|u4|i2|u2)
//   attr "axis" : int64_t
//   attr "block_size" : int64_t
//
// Semantics: output = clamp(round(data / scale) + zero_point)
// Mapped as approximate:  data / scale → round → add zp → convert
// For simplicity, use FakeQuantize or direct math.
// Here we use: Convert(Round(data / scale) + zp, target_type)
// ===================================================================
static ov::OutputVector translate_quantize_linear(const CgcNodeContext& ctx,
                                                  const ov::OutputVector& inputs,
                                                  const NodeOutputInfos& outputs) {
    auto data = inputs[0];
    auto scale = inputs[1];
    auto zero_point = inputs[2];
    const auto axis = ctx.get_attribute<int64_t>("axis");
    const auto block_size = ctx.get_attribute<int64_t>("block_size");
    OPENVINO_ASSERT(outputs.size() == 1);
    const auto quant_type = outputs[0].first;
    auto dequant_type = scale.get_element_type();
    const bool postfix = dequant_type == ov::element::f16 && quant_type == ov::element::u16;
    //if (postfix) {
    //    dequant_type = ov::element::f32;
    //    scale = std::make_shared<v0::Convert>(scale, dequant_type);
    //}
    const auto opname = ctx.get_op_name();

    const auto reshape_shape = VerifyQDQ(data, scale, zero_point, quant_type, dequant_type, axis, block_size);

    if (data.get_element_type() != dequant_type) {
        data = std::make_shared<v0::Convert>(data, dequant_type);
        data.get_node()->set_friendly_name(opname + "_pre_convert");
    }

    auto [out_low, out_high] = get_output_bounds(quant_type, postfix ? ov::element::f32 : dequant_type);
    auto [in_low, in_high] = get_input_bounds(out_low, out_high, zero_point, scale, reshape_shape);
    const std::size_t levels = static_cast<size_t>(1) << quant_type.bitwidth();
    auto ret = std::make_shared<v0::Convert>(
        std::make_shared<v0::FakeQuantize>(data, in_low, in_high, out_low, out_high, levels),
        quant_type);
    ret->set_friendly_name(opname + "_post_convert");
    ret->get_input_node_ptr(0)->set_friendly_name(opname);

    const auto in_low_const = try_get_constant<float>(in_low.get());
    const auto in_high_const = try_get_constant<float>(in_high.get());
    printf("qlinear[%s]: (%s) in[%s ~ %s] out[%d ~ %d]\n",
           opname.c_str(),
           postfix ? "fix" : "",
           in_low_const ? std::to_string(*in_low_const).c_str() : "?",
           in_high_const ? std::to_string(*in_high_const).c_str() : "?",
           out_low->cast_vector<int32_t>()[0],
           out_high->cast_vector<int32_t>()[0]);
    return {ret};
}

// ===================================================================
// cgc_op.add
//   input 0, 1 : tensors with numpy-style broadcasting
// ===================================================================
static ov::OutputVector translate_add(const CgcNodeContext& ctx,
                                      const ov::OutputVector& inputs,
                                      const NodeOutputInfos&) {
    auto node = std::make_shared<v1::Add>(inputs[0], inputs[1]);
    node->set_friendly_name(ctx.get_op_name());
    return node->outputs();
}

// ===================================================================
// cgc_op.reshape
//   input 0 : data
//   input 1 : shape (1-D integer tensor)
// ===================================================================
static ov::OutputVector translate_reshape(const CgcNodeContext& ctx,
                                          const ov::OutputVector& inputs,
                                          const NodeOutputInfos&) {
    auto node = std::make_shared<v1::Reshape>(inputs[0], inputs[1], false);
    node->set_friendly_name(ctx.get_op_name());
    return node->outputs();
}

// ===================================================================
// cgc_op.cast
//   input 0 : data
// ===================================================================
static ov::OutputVector translate_cast(const CgcNodeContext& ctx,
                                          const ov::OutputVector& inputs,
                                          const NodeOutputInfos& outputs) {
    OPENVINO_ASSERT(outputs.size() == 1);
    auto node = std::make_shared<v0::Convert>(inputs[0], outputs[0].first);
    node->set_friendly_name(ctx.get_op_name());
    return node->outputs();
}

// ===================================================================
// cgc_op.slice
//   input 0 : data
//   attrs: starts, ends, axes, steps (all vector<int64_t>)
// ===================================================================
static ov::OutputVector translate_slice(const CgcNodeContext& ctx,
                                        const ov::OutputVector& inputs,
                                        const NodeOutputInfos&) {
    const auto& data = inputs[0];
    auto starts = ctx.get_attribute<std::vector<int64_t>>("starts");
    auto ends = ctx.get_attribute<std::vector<int64_t>>("ends");
    auto axes = ctx.get_attribute<std::vector<int64_t>>("axes");
    auto step = ctx.get_attribute<std::vector<int64_t>>("step");

    auto starts_c = make_array_const(starts);
    auto ends_c = make_array_const(ends);
    auto step_c = make_array_const(step);
    auto axes_c = make_array_const(axes);

    auto node = std::make_shared<v8::Slice>(data, starts_c, ends_c, step_c, axes_c);
    node->set_friendly_name(ctx.get_op_name());
    return node->outputs();
}

// ===================================================================
// cgc_op.rotary_embedding
//   input 0 : input
//   input 1 : cos_cache
//   input 2 : sin_cache
//   input 3 : position_ids
//   input 4 : seqlens_k
//   attrs: interleaved, num_heads, rotary_embedding_dim, (rope_theta), (factor), (attention_factor)
//
// For now, decompose into the mathematical definition:
//   Split data into rotated_half and the rest,
//   apply cos/sin rotation on the rotated portion.
// ===================================================================

//Inputs:
//
//input: Token embeddings as a 3D tensor (batch_size, sequence_length, hidden_size). num_heads attribute must be provided.
//cos_cache (optional): Cosine values for rotation. Shape (max_position_id_plus_1, head_size/2) or (max_position_id_plus_1, rotary_embedding_dim/2) for partial rotation when position_ids provided; or (batch_size, sequence_length, head_size/2) or (batch_size, sequence_length, rotary_embedding_dim/2) when position_ids not provided. When null, the implementation computes cos values at runtime using rope_theta and factor (if provided). Both cos_cache and sin_cache must be null or both non-null.
//sin_cache (optional): Sine values for rotation with same shape requirements as cos_cache. When null, the implementation computes sin values at runtime using rope_theta and factor (if provided).
//position_ids (optional): 2D int64 tensor with shape (batch_size, sequence_length) for explicit per-token positions. Use for non-contiguous or arbitrary position orderings (e.g., speculative decoding, sparse attention).
//seqlens_k (optional): 1D int32 tensor of shape (batch_size) containing the past sequence length (starting position_id) per batch element. For contiguous decoding, position_id[b, i] = seqlens_k[b] + i. Consistent with GQA convention.
//At least one of seqlens_k or position_ids must be provided. When both are provided, the backend implementation chooses which to consume.
//
//Attributes:
//
//interleaved: If 1, rotate using interleaved pattern. Default is 0 (False).
//num_heads: Number of attention heads. Must be provided when input is 3D.
//rotary_embedding_dim: Dimension for partial rotary embeddings. Default 0 means full rotation.
//rope_theta (optional): Base frequency for RoPE computation (e.g., 10000.0). Required when cos_cache/sin_cache are null (runtime compute mode).
//factor (optional): Used with all rope types except ‘default’. The scaling factor to apply to the RoPE embeddings. In most scaling types, a factor of x will enable the model to handle sequences of length x * original maximum pre-trained length.
//attention_factor (optional, float): Used with 'yarn' and 'longrope' RoPE types. The scaling factor applied to the attention computation. If unspecified, defaults to the value recommended by the implementation, using factor to infer the suggested value.

static ov::OutputVector translate_rotary_embedding(const CgcNodeContext& ctx,
                                                   const ov::OutputVector& inputs,
                                                   const NodeOutputInfos&) {
    auto input = inputs[0];
    auto cos = inputs[1];
    auto sin = inputs[2];
    auto pos_ids = inputs[3];
    auto seq_k = inputs[4];
    FRONT_END_GENERAL_CHECK(bool(cos.get_node()) == bool(sin.get_node()),
                            "Both cos_cache and sin_cache must be null or both non-null");
    FRONT_END_GENERAL_CHECK(pos_ids.get_node() || seq_k.get_node(),
                            "At least one of position_ids or seqlens_k must be provided");
    const auto in_shape = input.get_partial_shape();
    const auto in_rank = in_shape.rank();
    FRONT_END_GENERAL_CHECK(in_rank.is_static() && in_rank.get_length() == 3,
                            "input should be (batch_size, sequence_length, hidden_size)");

    const bool interleaved = ctx.get_attribute<int64_t>("interleaved", 0);
    const auto num_heads = ctx.get_attribute<int64_t>("num_heads");
    const auto dim = ctx.get_attribute<int64_t>("rotary_embedding_dim", 0);

    const auto batch_dim = in_shape[0];
    FRONT_END_GENERAL_CHECK(batch_dim.is_static());
    const auto batch_size = batch_dim.get_length();
    const auto hidden_dim = in_shape[2];
    FRONT_END_GENERAL_CHECK(hidden_dim.is_static() && hidden_dim.get_length() % num_heads == 0);
    const auto head_size = hidden_dim.get_length() / num_heads;
    FRONT_END_GENERAL_CHECK(head_size % 2 == 0);
    const auto half_head_size = head_size / 2;
    FRONT_END_GENERAL_CHECK(dim == 0 || dim == head_size, "only full rotation supported for now");

    const auto zero = make_array_const<int64_t>({0});
    const auto one = make_array_const<int64_t>({1});
    const auto two = make_array_const<int64_t>({2});
    const auto neg = make_array_const<int64_t>({-1});
    const auto zero_scalar = make_scalar_const<int64_t>(0);
    const auto one_scalar = make_scalar_const<int64_t>(1);
    const auto neg_scalar = make_scalar_const<int64_t>(-1);

    const bool has_pos_ids = pos_ids.get_node();
    ov::Output<ov::Node> seq_k_2d;
    if (has_pos_ids) {
        const auto pos_shape = pos_ids.get_partial_shape();
        const auto pos_rank = pos_shape.rank();
        FRONT_END_GENERAL_CHECK(pos_rank.is_static() && pos_rank.get_length() == 2,
                                "position_ids must have shape (batch_size, sequence_length)");
    } else {
        const auto seq_shape = seq_k.get_partial_shape();
        FRONT_END_GENERAL_CHECK(seq_shape.is_static());
        FRONT_END_GENERAL_CHECK(ov::shape_size(seq_shape.to_shape()) == batch_size,
                                "seqlens_k element count must be batch_size");
        seq_k = std::make_shared<v0::Convert>(seq_k, ov::element::i64);
        const auto seq_rank = seq_shape.rank().get_length();
        if (seq_rank == 1) {
            seq_k_2d = std::make_shared<v0::Unsqueeze>(seq_k, one_scalar);
        } else if (seq_rank == 2 && seq_shape[0].get_length() == batch_size) {
            seq_k_2d = seq_k;
        } else {
            seq_k_2d = std::make_shared<v1::Reshape>(seq_k, make_array_const<int64_t>({batch_size, 1}), false);
        }
    }
    
    const auto perm = make_array_const<int32_t>({0, 2, 1, 3});  // BSNH <-> BNSH
    const auto batch_seq_node = gather_dimensions(input, {0, 1});
    const auto head_size_node = make_array_const<int64_t>({num_heads, head_size});
    const auto input_reshape = std::make_shared<v0::Concat>(ov::NodeVector{batch_seq_node, head_size_node}, 0);
    const auto head_half_node = make_array_const<int64_t>({num_heads, half_head_size});
    const auto input_half_shape =
        std::make_shared<v0::Concat>(ov::NodeVector{batch_seq_node, head_half_node}, 0);
    auto input_4d = std::make_shared<v1::Reshape>(input, input_reshape, false)->output(0);
    input_4d = std::make_shared<v1::Transpose>(input_4d, perm); // BNSH

    ov::Output<ov::Node> position_ids = pos_ids;
    if (!has_pos_ids) {
        const auto current_seqlen = gather_dimensions(input, {1});
        const auto current_seqlen_scalar = std::make_shared<v0::Squeeze>(current_seqlen, zero);
        ov::Output<ov::Node> position_range =
            std::make_shared<v4::Range>(zero_scalar, current_seqlen_scalar, one_scalar, ov::element::i64);
        position_range = std::make_shared<v0::Unsqueeze>(position_range, zero);
        position_ids = std::make_shared<v1::Add>(seq_k_2d, position_range);
    }

    if (cos.get_node()) {
        const auto cos_shape = cos.get_partial_shape();
        const auto sin_shape = sin.get_partial_shape();
        FRONT_END_GENERAL_CHECK(cos_shape == sin_shape);
        const auto cos_rank = cos_shape.rank();
        FRONT_END_GENERAL_CHECK(cos_rank.is_static() && (cos_rank.get_length() == 2 || cos_rank.get_length() == 3),
                                "cos_cache and sin_cache must have rank 2 or 3");
        const auto last_dim = cos_shape[cos_rank.get_length() - 1];
        FRONT_END_GENERAL_CHECK(last_dim.is_static() && last_dim.get_length() == half_head_size,
                                "only full rotation supported for now");

        if (cos_rank.get_length() == 2) {
            cos = std::make_shared<v8::Gather>(cos, position_ids, zero_scalar);
            sin = std::make_shared<v8::Gather>(sin, position_ids, zero_scalar);
        }
    } else {
        const auto rope_theta = ctx.get_attribute<float>("rope_theta");
        const auto factor = ctx.get_attribute<float>("factor", 1.0f);
        FRONT_END_GENERAL_CHECK(rope_theta > 0.0f, "rope_theta must be positive");
        FRONT_END_GENERAL_CHECK(factor > 0.0f, "factor must be positive");

        std::vector<float> inv_freq(static_cast<size_t>(half_head_size));
        for (int64_t i = 0; i < half_head_size; ++i) {
            const auto exponent = static_cast<float>(2 * i) / static_cast<float>(head_size);
            inv_freq[static_cast<size_t>(i)] = 1.0f / (factor * std::pow(rope_theta, exponent));
        }

        auto positions = std::make_shared<v0::Convert>(position_ids, ov::element::f32);

        const auto inv_freq_const = make_array_const(inv_freq);
        const auto positions_3d = std::make_shared<v0::Unsqueeze>(positions, neg_scalar);
        const auto angles = std::make_shared<v1::Multiply>(positions_3d, inv_freq_const);
        cos = std::make_shared<v0::Cos>(angles);
        sin = std::make_shared<v0::Sin>(angles);
    }

    if (cos.get_element_type() != input.get_element_type()) {
        cos = std::make_shared<v0::Convert>(cos, input.get_element_type());
    }
    if (sin.get_element_type() != input.get_element_type()) {
        sin = std::make_shared<v0::Convert>(sin, input.get_element_type());
    }
    cos = std::make_shared<v0::Unsqueeze>(cos, one_scalar);
    sin = std::make_shared<v0::Unsqueeze>(sin, one_scalar);

    const auto apply_rope = [&](ov::Output<ov::Node> data, bool is_5d) -> std::shared_ptr<ov::Node> {
        const auto split = std::make_shared<v1::Split>(data, neg_scalar, 2);
        auto data0 = split->output(0);
        auto data1 = split->output(1);
        if (is_5d) {
            data0 = std::make_shared<v0::Squeeze>(data0, neg_scalar);
            data1 = std::make_shared<v0::Squeeze>(data1, neg_scalar);
        }

        std::shared_ptr<ov::Node> res0 = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(data0, cos),
                                                                        std::make_shared<v1::Multiply>(data1, sin));
        std::shared_ptr<ov::Node> res1 = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(data0, sin),
                                                                   std::make_shared<v1::Multiply>(data1, cos));
        if (is_5d) {
            res0 = std::make_shared<v0::Unsqueeze>(res0, neg_scalar);
            res1 = std::make_shared<v0::Unsqueeze>(res1, neg_scalar);
        }
        const auto concat = std::make_shared<v0::Concat>(ov::NodeVector{res0, res1}, -1);
        if (is_5d) {
            return std::make_shared<v1::Reshape>(concat, std::make_shared<v3::ShapeOf>(input_4d), false);
        } else {
            return concat;
        }
    };
    ov::Output<ov::Node> ret;
    if (interleaved) {
        const auto input_reshape5d = std::make_shared<v0::Concat>(ov::NodeVector{input_half_shape, two}, 0);
        auto input_5d = std::make_shared<v1::Reshape>(input_4d, input_reshape5d, false);
        ret = apply_rope(input_5d, true);
    } else {
        ret = apply_rope(input_4d, false);
    }
    ret = std::make_shared<v1::Transpose>(ret, perm);
    ret = std::make_shared<v1::Reshape>(ret, std::make_shared<v3::ShapeOf>(input), false);
    ret.get_node_shared_ptr()->set_friendly_name(ctx.get_op_name());
    return {ret};
}

// ===================================================================
// cgc_op.convolution
//   input 0 : data      (N, C_in, spatial...)
//   input 1 : weights   (C_out, C_in, kH, kW)
//   input 2 : bias      (may be null)
//   attrs: strides, start_padding, end_padding
// ===================================================================
static ov::OutputVector translate_convolution(const CgcNodeContext& ctx,
                                              const ov::OutputVector& inputs,
                                              const NodeOutputInfos&) {
    const auto& data = inputs[0];
    const auto& weights = inputs[1];

    auto strides_v = ctx.get_attribute<std::vector<int64_t>>("strides");
    auto pads_begin_v = ctx.get_attribute<std::vector<int64_t>>("start_padding");
    auto pads_end_v = ctx.get_attribute<std::vector<int64_t>>("end_padding");

    ov::Strides strides(strides_v.begin(), strides_v.end());
    ov::CoordinateDiff pads_begin(pads_begin_v.begin(), pads_begin_v.end());
    ov::CoordinateDiff pads_end(pads_end_v.begin(), pads_end_v.end());
    ov::Strides dilations(strides.size(), 1);

    auto conv = std::make_shared<v1::Convolution>(
        data, weights, strides, pads_begin, pads_end, dilations);
    conv->set_friendly_name(ctx.get_op_name());

    // Add bias if present
    if (inputs.size() > 2) {
        const auto& bias_node = inputs[2].get_node_shared_ptr();
        if (bias_node) {
            auto add = std::make_shared<v1::Add>(conv, inputs[2]);
            add->set_friendly_name(ctx.get_op_name() + "/add_bias");
            return add->outputs();
        }
    }
    return conv->outputs();
}

// ===================================================================
// cgc_op.lp_normalization
//   input 0 : data
//   attrs: axis (int64), epsilon (float), p (int64)
//
// For p==2 L2 normalization → v0::NormalizeL2
// For p==1 L1 normalization → reduce_sum(abs(x))
// ===================================================================
static ov::OutputVector translate_lp_normalization(const CgcNodeContext& ctx,
                                                   const ov::OutputVector& inputs,
                                                   const NodeOutputInfos&) {
    const auto& data = inputs[0];
    auto axis = ctx.get_attribute<int64_t>("axis");
    auto p = ctx.get_attribute<int64_t>("p");
    FRONT_END_GENERAL_CHECK(p == 1 || p == 2, "Only L1/L2 normalization is supported, got p=", p);

    const auto axes_const = make_scalar_const(axis);
    std::shared_ptr<ov::Node> node;
    if (p == 2) {
        auto eps = ctx.get_attribute<float>("epsilon", 1e-12f);
        node = std::make_shared<v0::NormalizeL2>(data, axes_const, eps, ov::op::EpsMode::ADD);
    } else {
        node = std::make_shared<v0::Abs>(data);
        node = std::make_shared<v1::ReduceSum>(node, axes_const, true);
    }
    node->set_friendly_name(ctx.get_op_name());
    return node->outputs();
}

// ===================================================================
// cgc_op.multiply
//   input 0, 1 : tensors with numpy-style broadcasting
// ===================================================================
static ov::OutputVector translate_multiply(const CgcNodeContext& ctx,
                                           const ov::OutputVector& inputs,
                                           const NodeOutputInfos&) {
    auto node = std::make_shared<v1::Multiply>(inputs[0], inputs[1]);
    node->set_friendly_name(ctx.get_op_name());
    return node->outputs();
}

// ===================================================================
// cgc_op.sigmoid
//   input 0 : data
// ===================================================================
static ov::OutputVector translate_sigmoid(const CgcNodeContext& ctx,
                                          const ov::OutputVector& inputs,
                                          const NodeOutputInfos&) {
    auto node = std::make_shared<v0::Sigmoid>(inputs[0]);
    node->set_friendly_name(ctx.get_op_name());
    return node->outputs();
}

// ===================================================================
// cgc_op.group_query_attention
//   input 0  : query
//   input 1  : key
//   input 2  : value
//   input 3  : past_key
//   input 4  : past_value
//   input 5  : seqlens_k
//   input 6  : total_sequence_length
//   input 7  : cos_cache
//   input 8  : sin_cache
//   input 9  : position_ids
//   input 10 : attention_bias
//   input 11 : head_sink
//   input 12 : k_scale
//   input 13 : v_scale
// ===================================================================

static ov::OutputVector translate_gqa(const CgcNodeContext& ctx,
                                      const ov::OutputVector& inputs,
                                      const NodeOutputInfos&) {
    FRONT_END_GENERAL_CHECK(inputs.size() == 14,
                            "cgc_op.group_query_attention expects 14 inputs, but got ",
                            inputs.size());

    const auto num_heads = ctx.get_attribute<int64_t>("num_heads");
    const auto kv_num_heads = ctx.get_attribute<int64_t>("kv_num_heads");
    const auto scale = ctx.get_attribute<float>("scale", 0.0f);
    const auto do_rotary = ctx.get_attribute<int64_t>("do_rotary", 0) != 0;
    const auto rotary_interleaved = ctx.get_attribute<int64_t>("rotary_interleaved", 0) != 0;

    auto query = inputs[0];
    auto key = inputs[1];
    auto value = inputs[2];
    auto past_key = inputs[3];
    auto past_value = inputs[4];
    const auto key_node = key.get_node_shared_ptr();
    const auto value_node = value.get_node_shared_ptr();
    const auto past_key_node = past_key.get_node_shared_ptr();
    const auto past_value_node = past_value.get_node_shared_ptr();

    FRONT_END_GENERAL_CHECK(static_cast<bool>(key_node) == static_cast<bool>(value_node),
                            "cgc_op.group_query_attention expects key and value to be both present or both null");
    FRONT_END_GENERAL_CHECK(
        static_cast<bool>(past_key_node) == static_cast<bool>(past_value_node),
        "cgc_op.group_query_attention expects past_key and past_value to be both present or both null");

    // kv: B, S, N, H
    const auto perm = make_array_const<int32_t>({0, 2, 1, 3});  // BSNH <-> BNSH
    const auto batch_size_node = gather_dimensions(query, {0});
    const auto current_seqlen_size_node = gather_dimensions(query, {1});
    const auto hidden_size_node = gather_dimensions(query, {2});

    ov::OutputVector gqa_inputs;
    if (!key_node) {
        const auto total_num_heads_node = make_array_const<int64_t>({num_heads + kv_num_heads + kv_num_heads});
        const auto head_size_node = std::make_shared<v1::Divide>(hidden_size_node, total_num_heads_node);
        const auto packed_qkv_shape = std::make_shared<v0::Concat>(
            ov::NodeVector{batch_size_node, current_seqlen_size_node, total_num_heads_node, head_size_node},
            0);

        auto packed_qkv = std::make_shared<v1::Reshape>(query, packed_qkv_shape, false)->output(0);
        packed_qkv = std::make_shared<v1::Transpose>(packed_qkv, perm);

        const auto split_axis = make_scalar_const<int64_t>(1);
        const auto split_lengths = make_array_const<int64_t>({num_heads, kv_num_heads, kv_num_heads});
        const auto split = std::make_shared<v1::VariadicSplit>(packed_qkv, split_axis, split_lengths);
        const auto split_outputs = split->outputs();
        gqa_inputs.insert(gqa_inputs.end(), split_outputs.begin(), split_outputs.end());
    } else {
        const auto num_heads_node = make_array_const<int64_t>({num_heads});
        const auto head_size_node = std::make_shared<v1::Divide>(hidden_size_node, num_heads_node);
        const auto q_shape = std::make_shared<v0::Concat>(
            ov::NodeVector{batch_size_node, current_seqlen_size_node, num_heads_node, head_size_node},
            0);

        query = std::make_shared<v1::Reshape>(query, q_shape, false);
        query = std::make_shared<v1::Transpose>(query, perm);
        gqa_inputs.push_back(query);

        const auto kv_num_heads_node = make_array_const<int64_t>({kv_num_heads});
        const auto kv_shape = std::make_shared<v0::Concat>(
            ov::NodeVector{batch_size_node, current_seqlen_size_node, kv_num_heads_node, head_size_node},
            0);

        key = std::make_shared<v1::Reshape>(key, kv_shape, false);
        value = std::make_shared<v1::Reshape>(value, kv_shape, false);
        key = std::make_shared<v1::Transpose>(key, perm);
        value = std::make_shared<v1::Transpose>(value, perm);
        gqa_inputs.push_back(key);
        gqa_inputs.push_back(value);
    }

    if (past_key_node) {
        past_key = std::make_shared<v1::Transpose>(past_key, perm);
        past_value = std::make_shared<v1::Transpose>(past_value, perm);
        gqa_inputs.push_back(past_key);
        gqa_inputs.push_back(past_value);
    } else {
        gqa_inputs.push_back(get_null_connection());
        gqa_inputs.push_back(get_null_connection());
    }

    for (size_t input_index = 5; input_index < inputs.size(); ++input_index) {
        gqa_inputs.push_back(inputs[input_index].get_node() ? inputs[input_index] : get_null_connection());
    }
    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(gqa_inputs,
                                                                       num_heads,
                                                                       kv_num_heads,
                                                                       scale,
                                                                       do_rotary,
                                                                       rotary_interleaved);
    gqa->set_friendly_name(ctx.get_op_name());
    ov::OutputVector rets;
    rets.push_back(gqa->output(0));
    rets.push_back(std::make_shared<v1::Transpose>(gqa->output(1), perm));
    rets.push_back(std::make_shared<v1::Transpose>(gqa->output(2), perm));
    rets.push_back(get_null_connection()); // output_qk_matrix(optional)
    return rets;
}


const CgcTranslatorMap& get_supported_ops() {
    static const CgcTranslatorMap ops = {
        {"cgc_op.transpose",                translate_transpose},
        {"cgc_op.gemm",                     translate_gemm},
        {"cgc_op.dequantize_linear",        translate_dequantize_linear},
        {"cgc_op.quantize_linear",          translate_quantize_linear},
        {"cgc_op.add",                      translate_add},
        {"cgc_op.reshape",                  translate_reshape},
        {"cgc_op.cast",                     translate_cast},
        {"cgc_op.slice",                    translate_slice},
        {"cgc_op.rotary_embedding",         translate_rotary_embedding},
        {"cgc_op.convolution",              translate_convolution},
        {"cgc_op.lp_normalization",         translate_lp_normalization},
        {"cgc_op.multiply",                 translate_multiply},
        {"cgc_op.sigmoid",                  translate_sigmoid},
        {"cgc_op.group_query_attention",    translate_gqa},
    };
    return ops;
}

}  // namespace dxgml
}  // namespace frontend
}  // namespace ov
