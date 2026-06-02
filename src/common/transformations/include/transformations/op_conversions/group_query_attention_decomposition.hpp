// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API GroupQueryAttentionDecomposition;

}  // namespace pass
}  // namespace ov

class ov::pass::GroupQueryAttentionDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GroupQueryAttentionDecomposition");
    GroupQueryAttentionDecomposition();

private:
    ov::OutputVector decompose(std::shared_ptr<ov::op::internal::GroupQueryAttention> node);
    std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<op::v3::ShapeOf>& shape,
                                             const std::vector<int>& dims);
    std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims);
    std::shared_ptr<ov::Node> rotaryEmbedding(ov::Output<ov::Node> input,
                                              ov::Output<ov::Node> cos,
                                              ov::Output<ov::Node> sin,
                                              bool interleaved);
    struct CachedNodes {
        ov::Output<ov::Node> pos_ids;
        ov::Output<ov::Node> kv_slices;
        std::shared_ptr<ov::Node> concat_kv_len;
        std::map<ov::Output<ov::Node>, ov::Output<ov::Node>> rotary_cache;
    };
    std::map<ov::Output<ov::Node>, CachedNodes> m_seqk_cache;
};
