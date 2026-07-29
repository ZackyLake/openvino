// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <winrt/base.h>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/decoder.hpp"
#include "openvino/frontend/exception.hpp"


namespace ov {
namespace frontend {
namespace cgc {

class CgcInputModel;

struct CgcNodeContext {
    std::string_view m_op_type;
    std::shared_ptr<DecoderBase> m_decoder;

    CgcNodeContext(std::string_view op_type, const std::shared_ptr<DecoderBase>& decoder)
        : m_op_type(op_type),
          m_decoder(decoder) {}

    const std::string& get_op_name() const {
        return m_decoder->get_op_name();
    }

    template <class T>
    T get_attribute(const std::string& name) const {
        auto any = m_decoder->get_attribute(name);
        FRONT_END_GENERAL_CHECK(!any.empty(), "Attribute '", name, "' not found on ", m_op_type);
        return any.as<T>();
    }

    template <class T>
    std::optional<T> try_get_attribute(const std::string& name) const {
        auto any = m_decoder->get_attribute(name);
        if (!any.empty())
            return any.as<T>();
        return {};
    }

    template <class T>
    T get_attribute(const std::string& name, const T& def) const {
        auto any = m_decoder->get_attribute(name);
        return any.empty() ? def : any.as<T>();
    }

    bool has_attribute(const std::string& name) const {
        return !m_decoder->get_attribute(name).empty();
    }

};


using NodeOutputInfos = std::vector<std::pair<ov::element::Type, ov::PartialShape>>;
using CgcTranslatorFunction = std::function<ov::OutputVector(const CgcNodeContext&, const ov::OutputVector&, const NodeOutputInfos&)>;
using CgcTranslatorMap = std::unordered_map<std::string_view, CgcTranslatorFunction>;

const CgcTranslatorMap& get_supported_ops();


std::shared_ptr<ov::Model> translate_dxgml_model(const std::shared_ptr<CgcInputModel>& input_model);

}  // namespace cgc
}  // namespace frontend
}  // namespace ov
