// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cgc_conversion.h"

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/frontend.hpp"
#include "openvino/frontend/graph_iterator.hpp"
#include "openvino/frontend/input_model.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/place.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "openvino/util/mmap_object.hpp"

#define NOMINMAX
#include "cgc_ir_templates.h"
#include "cgc_ir.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>


#ifdef OPENVINO_STATIC_LIBRARY
#    define CGC_FRONTEND_C_API
#else
#    ifdef openvino_cgc_frontend_EXPORTS
#        define CGC_FRONTEND_C_API OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS
#    else
#        define CGC_FRONTEND_C_API OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS
#    endif
#endif

namespace ov {
namespace frontend {
namespace cgc {

using namespace cgc_ir_templates;

template <typename T>
static std::string val_to_string(const T& val) {
    return std::to_string(val);
}
template <>
static std::string val_to_string<>(const std::string_view& val) {
    return std::string(val);
}
template <>
static std::string val_to_string<>(const std::string& val) {
    return val;
}
template <>
static std::string val_to_string<>(const bool& val) {
    return val ? "true" : "false";
}
template <typename T>
static std::string val_to_string(const std::span<T>& sp) {
    std::string ret = "[";
    bool is_first = true;
    for (const auto& v : sp) {
        if (!is_first) {
            ret.append(", ");
        }
        ret.append(val_to_string(v));
        is_first = false;
    }
    ret.append("]");
    return ret;
}
template <typename T>
static std::string val_to_string(const std::vector<T>& vec) {
    return val_to_string(std::span{vec});
}

namespace detail {
template <typename T>
struct is_span : std::false_type {};
template <typename ElementType, size_t Extent>
struct is_span<std::span<ElementType, Extent>> : std::true_type {};
struct string_hash {
    using is_transparent = void;
    size_t operator()(std::string_view sv) const {
        return std::hash<std::string_view>{}(sv);
    }
};
}  // namespace detail
template <typename T>
using is_span = detail::is_span<std::remove_cvref_t<T>>;
template <typename T>
inline constexpr bool is_span_v = detail::is_span<std::remove_cvref_t<T>>::value;
template <typename V, typename K = std::string>
using KVDict = std::unordered_map<K, V, detail::string_hash, std::equal_to<>>;

template<ov::element::Type_t T>
static std::vector<ov::fundamental_type_for<T>> convert_n_ele(const std::byte* data, size_t count) {
    auto it = ov::element::iterator<T>(reinterpret_cast<const int8_t*>(data));
    std::vector<ov::fundamental_type_for<T>> ret(it, it + count);
    return ret;
}

static std::string serialize_n_ele(std::span<const std::byte> data, ov::element::Type type, size_t n = 4) {
    const auto count = std::min<size_t>(data.size() * 8 / type.bitwidth(), n);
    switch (type) {
#define CastStr(T) return val_to_string(std::span{reinterpret_cast<const T*>(data.data()), count})
#define CvtStr(T)  return val_to_string(convert_n_ele<ov::element::Type_t::T>(data.data(), count));
    case ov::element::boolean:
        CastStr(bool);
    case ov::element::i4:
        CvtStr(i4);
    case ov::element::i8:
        CastStr(int8_t);
    case ov::element::i16:
        CastStr(int16_t);
    case ov::element::i32:
        CastStr(int32_t);
    case ov::element::i64:
        CastStr(int64_t);
    case ov::element::u4:
        CvtStr(u4);
    case ov::element::u8:
        CastStr(uint8_t);
    case ov::element::u16:
        CastStr(uint16_t);
    case ov::element::u32:
        CastStr(uint32_t);
    case ov::element::u64:
        CastStr(uint64_t);
    case ov::element::f16:
        CastStr(ov::float16);
    case ov::element::bf16:
        CastStr(ov::bfloat16);
    case ov::element::f32:
        CastStr(float);
    case ov::element::f64:
        CastStr(double);
    case ov::element::f8e5m2:
        CastStr(ov::float8_e5m2);
    case ov::element::f8e4m3:
        CastStr(ov::float8_e4m3);
    case ov::element::f8e8m0:
        CastStr(ov::float8_e8m0);
    default:
        return "<" + type.get_type_name() + ">";
    }
}

static std::string_view trim_left(std::string_view text) {
    size_t idx = 0;
    while (idx < text.size()) {
        if (std::isspace(text[idx]))
            idx++;
        else
            break;
    }
    text.remove_prefix(idx);
    return text;
}
static std::string_view trim_right(std::string_view text) {
    size_t idx = text.size();
    while (idx--) {
        if (!std::isspace(text[idx])) {
            idx++;
            break;
        }
    }
    text.remove_suffix(text.size() - idx);
    return text;
}

// return {line_break, line_end}
static std::pair<size_t, size_t> find_line_break(std::string_view text) {
    auto pos = text.find_first_of("\r\n");
    if (pos < text.size() - 1 && text[pos] == '\r' && text[pos + 1] == '\n') {
        return {pos + 1, pos};
    }
    return {pos, pos};
}

static inline std::optional<uint32_t> parse_hex(char ch) {
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
        return ch - 'a' + 10;
    }
    if (ch >= 'A' && ch <= 'F') {
        return ch - 'A' + 10;
    }
    return {};
}

static inline std::optional<std::byte> hex_to_byte(char ch0, char ch1) {
    const auto hi = parse_hex(ch0);
    const auto lo = parse_hex(ch1);
    if (hi && lo) {
        return std::byte(*lo + ((*hi) << 4));
    }
    return {};
}

static std::vector<std::byte> hex_to_bytes(std::string_view str) {
    std::vector<std::byte> data;
    data.reserve(str.size() / 2);
    while (!str.empty()) {
        const auto byte = hex_to_byte(str[0], str[1]);
        if (!byte)
            return {};
        data.push_back(*byte);
        str.remove_prefix(2);
    }
    return data;
}

/*
{-#
  dialect_resources: {
    cgc: {
      __model_Constant_output_0: "0x010000000000000000000000",
      __model_layers.1_mlp_Mul_output_0_zero_point: "0x01000000CC22",
      __model_layers.1_mlp_Mul_output_0_scale: "0x0100000054A92A3D",
      __model_layers.1_mlp_down_proj_Conv_output_0_zero_point: "0x01000000B37F"
    }
  }
#-}
*/
static std::optional<KVDict<std::vector<std::byte>>> parse_cgc_resources(std::string_view content) {
    KVDict<std::string_view, std::string_view> pending;

    std::vector<std::string_view> scopes;
    bool is_cgc_scope = false;
    const auto update_scope = [&]() {
        is_cgc_scope = scopes.size() == 3 && scopes[1] == "dialect_resources" && scopes[2] == "cgc";
    };
    size_t line_count = 0;
    const auto logerr = [&](const char* fmt, auto&&... args) {
        printf("at line[%zu](", line_count > 0 ? line_count - 1 : line_count);
        for (size_t i = 0; i < scopes.size(); ++i) {
            printf("%s%s", i == 0 ? "" : " > ", std::string(scopes[i]).c_str());
        }
        printf("), ");
        printf(fmt, args...);
    };
    constexpr auto npos = std::string_view::npos;

    while (!content.empty()) {
        const auto [line_break, line_end] = find_line_break(content);
        auto line = content.substr(0, line_end);
        if (line_end == npos) {
            content = {};
        } else {
            line_count++;
            content = content.substr(line_break + 1);
        }

        line = trim_left(line);
        if (line.empty())
            continue;

        if (scopes.empty()) {
            if (line.starts_with("{-#")) {
                scopes.push_back(line.substr(0, 3));
                // ignore rest of line
                continue;
            }
            logerr("no start mark: %s\n", std::string(line).c_str());
            return {};
        }
        if (scopes.size() == 1) {
            if (line.starts_with("#-}")) {
                scopes.pop_back();
                // ignore rest of line
                continue;
            }
        }

        if (line[0] == '}') {  // end of a scape
            line.remove_prefix(1);
            line = trim_left(line);
            if (!line.empty()) {
                logerr("not empty after right brace: %s\n", std::string(line).c_str());
                return {};
            }
            scopes.pop_back();
            update_scope();
            continue;
        }

        const auto comma_pos = line.find(':');
        if (comma_pos == npos || comma_pos == 0) {
            logerr("expects a name: %s\n", std::string(line).c_str());
            return {};
        }
        const auto name = line.substr(0, comma_pos);
        line = line.substr(comma_pos + 1);
        line = trim_left(line);

        if (line.empty()) {
            logerr("nothing after a name\n");
            return {};
        }
        if (line[0] == '{') {  // a new scope
            line.remove_prefix(1);
            line = trim_left(line);
            if (!line.empty()) {
                logerr("not empty after left brace: %s\n", std::string(line).c_str());
                return {};
            }
            scopes.push_back(name);
            update_scope();
            continue;
        }

        if (!is_cgc_scope) {
            logerr("ignored content: %s\n", std::string(line).c_str());
            continue;
        }

        if (line[0] != '"') {
            logerr("expects a string to represent data: %s\n", std::string(line).c_str());
            return {};
        }
        line.remove_prefix(1);

        const auto quota_pos = line.find('"');
        if (quota_pos == npos) {
            logerr("string not closed within the line: %s\n", std::string(line).c_str());
            return {};
        }
        auto value = line.substr(0, quota_pos);
        line = line.substr(quota_pos + 1);
        if (!line.empty() && line[0] == ',') {
            line.remove_prefix(1);
        }
        line = trim_left(line);
        if (!line.empty()) {
            logerr("not empty after a kv: %s\n", std::string(line).c_str());
            return {};
        }

        if (!value.starts_with("0x") || value.size() % 2 != 0) {
            logerr("expects hex bytes value: %s\n", std::string(value).c_str());
            return {};
        }
        value.remove_prefix(2);
        if (value.size() <= 8) {
            logerr("expects a UINT32 version and some data: %s\n", std::string(value).c_str());
            return {};
        }
        const auto version = value.substr(0, 8);
        value.remove_prefix(8);
        const auto ver_data = hex_to_bytes(version);
        if (ver_data.size() != 4) {
            logerr("expects a UINT32 version: %s\n", std::string(version).c_str());
            return {};
        }
        if (!pending.insert_or_assign(name, value).second) {
            logerr("key[%s] already added, will override!\n", std::string(name).c_str());
        }
    }

    if (pending.empty()) {
        return {};
    }
    printf("==> collected %zu] resoucres.\n", pending.size());

    const auto max_threads = std::max(1, ov::get_number_of_cpu_cores());
    const auto stream_count = std::min(max_threads, static_cast<int>(pending.size()));
    auto executor = std::make_shared<ov::threading::CPUStreamsExecutor>(
        ov::threading::IStreamsExecutor::Config{"CgcResourceHexDecoder", stream_count, 1});

    KVDict<std::vector<std::byte>> resources;
    std::atomic_bool failed = false;
    std::vector<ov::threading::Task> tasks;
    tasks.reserve(pending.size());
    for (const auto& [name_, value] : pending) {
        auto& data = resources[std::string(name_)];
        tasks.emplace_back([&]() {
            data = hex_to_bytes(value);
            std::string name(name_);
            if (data.empty()) {
                failed = true;
                printf("==> failed to load resource [%s]: %s\n", name.c_str(),
                       std::string(value).c_str());
            } else {
                printf("==> load resource [%s]: [%zu] bytes\n", name.c_str(), data.size());
            }
        });
    }
    executor->run_and_wait(tasks);
    if (failed) {
        return {};
    }
    return resources;
}

static ov::element::Type map_element_type(CGC_IR_TYPE type) {
    switch (type) {
    case CGC_IR_TYPE_BOOL8:         return ov::element::boolean;
    case CGC_IR_TYPE_INT4:          return ov::element::i4;
    case CGC_IR_TYPE_INT8:          return ov::element::i8;
    case CGC_IR_TYPE_INT16:         return ov::element::i16;
    case CGC_IR_TYPE_INT32:         return ov::element::i32;
    case CGC_IR_TYPE_INT64:         return ov::element::i64;
    case CGC_IR_TYPE_UINT4:         return ov::element::u4;
    case CGC_IR_TYPE_UINT8:         return ov::element::u8;
    case CGC_IR_TYPE_UINT16:        return ov::element::u16;
    case CGC_IR_TYPE_UINT32:        return ov::element::u32;
    case CGC_IR_TYPE_UINT64:        return ov::element::u64;
    case CGC_IR_TYPE_FLOAT16:       return ov::element::f16;
    case CGC_IR_TYPE_BFLOAT16:      return ov::element::bf16;
    case CGC_IR_TYPE_FLOAT32:       return ov::element::f32;
    case CGC_IR_TYPE_FLOAT64:       return ov::element::f64;
    case CGC_IR_TYPE_FLOAT8E5M2:    return ov::element::f8e5m2;
    case CGC_IR_TYPE_FLOAT8E4M3:    return ov::element::f8e4m3;
    case CGC_IR_TYPE_FLOAT8E8M0FNU: return ov::element::f8e8m0;
    default:                        return ov::element::dynamic;
    }
}

static std::string_view attr_type_to_string(CGC_IR_ATTRIBUTE type) {
    switch (type) {
    case CGC_IR_ATTRIBUTE_UNKNOWN:    return "unknown";
    case CGC_IR_ATTRIBUTE_ENUM:       return "enum";
    case CGC_IR_ATTRIBUTE_STRING:     return "string";
    case CGC_IR_ATTRIBUTE_SCALAR:     return "scalar";
    case CGC_IR_ATTRIBUTE_MULTIDIM:   return "multidim";
    default:                          return "";
    }
}


static std::string attr_type_to_string(Attribute<> attr) {
    const auto type = attr.GetKind();
    std::string typestr(attr_type_to_string(type));
    if (type == CGC_IR_ATTRIBUTE_SCALAR || type == CGC_IR_ATTRIBUTE_MULTIDIM) {
        CGC_IR_TYPE dtype = CGC_IR_TYPE_UNKNOWN;
        std::optional<size_t> count;
        if (type == CGC_IR_ATTRIBUTE_MULTIDIM) {
            MultiDimAttribute<> multi(attr.m_impl.as<ICgcMlirMultiDimAttribute>());
            dtype = multi.GetType().GetDataType();
            count = multi.GetValueSize();
        } else {
            ScalarAttribute<> scalar(attr.m_impl.as<ICgcMlirScalarAttribute>());
            dtype = scalar.GetType().GetKind();
        }
        const auto ovtype = map_element_type(dtype);
        typestr.append("<").append(ovtype.to_string());
        if (count) {
            typestr.append("[").append(std::to_string(*count)).append("]");
        }
        typestr.append(">");
    }
    return typestr;
}

static ov::PartialShape get_partial_shape(ICgcMlirMultiDimType* mdt) {
    if (!mdt)
        return ov::PartialShape::dynamic();

    winrt::com_ptr<ICgcMlirSpan> shape_span;
    if (FAILED(mdt->GetShape(IID_PPV_ARGS(&shape_span))))
        return ov::PartialShape::dynamic();

    auto rank = mdt->GetRank();
    auto* data = static_cast<const int64_t*>(shape_span->GetData());
    std::vector<ov::Dimension> dims;
    dims.reserve(rank);
    for (size_t i = 0; i < rank; ++i)
        dims.emplace_back(data[i]);
    return ov::PartialShape(dims);
}

static std::pair<ov::element::Type, ov::PartialShape> get_type_layout(const MultiDimType<>& mdt) {
    const auto ele_type = map_element_type(mdt.GetDataType());
    const auto shape = mdt.GetShape().AsSpan<int64_t>();
    std::vector<ov::Dimension> dims;
    for (const auto dim : shape)
        dims.push_back(dim);
    return {ele_type, ov::PartialShape(dims)};
}
static std::pair<ov::element::Type, ov::PartialShape> get_type_layout(const Type<>& type) {
    if (winrt::com_ptr<ICgcMlirMultiDimType> mdt; SUCCEEDED(type.Get()->QueryInterface(IID_PPV_ARGS(&mdt)))) {
        return get_type_layout(std::move(mdt));
    }
    const auto ele_type = map_element_type(type.GetKind());
    return {ele_type, ov::PartialShape()};
}


class CgcPlace : public Place {
public:
    CgcPlace(const std::string& name, ov::element::Type et, ov::PartialShape shape)
        : m_name(name), m_element_type(et), m_shape(std::move(shape)) {}

    std::vector<std::string> get_names() const override { return {m_name}; }
    bool is_input() const override { return m_is_input; }
    bool is_output() const override { return m_is_output; }

    ov::element::Type get_element_type() const { return m_element_type; }
    ov::PartialShape get_shape() const { return m_shape; }

    void set_input(bool v) { m_is_input = v; }
    void set_output(bool v) { m_is_output = v; }

private:
    std::string m_name;
    ov::element::Type m_element_type;
    ov::PartialShape m_shape;
    bool m_is_input = false;
    bool m_is_output = false;
};


struct ValueProducerInfo {
    std::string producer_name;
    size_t output_port_index = 0;
};

struct NodeOutputInfo {
    enum class TargetType { Null, Input, Constant, Op };
    TargetType target = TargetType::Null;
    uint32_t node_idx = 0;
    uint32_t port_idx = 0;
};

template<typename F>
static auto get_attr_val(Attribute<> attr, F&& func) {
    if (!attr) {
        return func(std::nullopt);
    }
    switch (attr.GetKind()) {
    case CGC_IR_ATTRIBUTE_SCALAR: {
        ScalarAttribute<> scalar(attr.m_impl.as<ICgcMlirScalarAttribute>());
        switch (scalar.GetType().GetKind()) {
        case CGC_IR_TYPE_INT64:
            return func(scalar.GetInt64Value());
        case CGC_IR_TYPE_INT32:
            return func(scalar.GetInt32Value());
        case CGC_IR_TYPE_INT16:
            return func(scalar.GetInt16Value());
        case CGC_IR_TYPE_INT8:
            return func(scalar.GetInt8Value());
        case CGC_IR_TYPE_BOOL8:
            return func(scalar.GetBoolValue());
        case CGC_IR_TYPE_FLOAT32:
            return func(scalar.GetFloat32Value());
        case CGC_IR_TYPE_FLOAT64:
            return func(scalar.GetFloat64Value());
        default:
            return func(std::nullopt);
        }
    }
    case CGC_IR_ATTRIBUTE_MULTIDIM: {
        MultiDimAttribute<> multi(attr.m_impl.as<ICgcMlirMultiDimAttribute>());
        switch (multi.GetType().GetDataType()) {
        case CGC_IR_TYPE_INT64:
            return func(multi.AsSpan<int64_t>());
        case CGC_IR_TYPE_INT32:
            return func(multi.AsSpan<int32_t>());
        case CGC_IR_TYPE_INT16:
            return func(multi.AsSpan<int16_t>());
        case CGC_IR_TYPE_INT8:
            return func(multi.AsSpan<int8_t>());
        case CGC_IR_TYPE_BOOL8:
            return func(multi.AsSpan<bool>());
        case CGC_IR_TYPE_FLOAT32:
            return func(multi.AsSpan<float>());
        case CGC_IR_TYPE_FLOAT64:
            return func(multi.AsSpan<double>());
        default:
            return func(std::nullopt);
        }
    }
    case CGC_IR_ATTRIBUTE_STRING: {
        StringAttribute<> str(attr.m_impl.as<ICgcMlirStringAttribute>());
        return func(str.AsStringView());
    }
    case CGC_IR_ATTRIBUTE_ENUM: {
        EnumAttribute<> str(attr.m_impl.as<ICgcMlirEnumAttribute>());
        return func(str.AsStringView());
    }
    default:
        return func(std::nullopt);
    }
}


static std::string attr_val_to_string(Attribute<> attr) {
    return get_attr_val(attr, [](auto&& val) -> std::string {
        using T = std::remove_cvref_t<decltype(val)>;
        if constexpr (std::is_same_v<T, std::nullopt_t>) {
            return {};
        } else {
            return val_to_string(val);
        }
    });
}

class CgcDecoder : public DecoderBase {
public:
    CgcDecoder(Op<> op, const std::string& node_name, std::vector<ValueProducerInfo>&& inputs)
        : m_op(std::move(op)),
          m_op_type(m_op.GetName().AsStringView()),
          m_op_name(node_name),
          m_inputs(std::move(inputs)) {}

    const std::string& get_op_type() const override {
        return m_op_type;
    }
    const std::string& get_op_name() const override {
        return m_op_name;
    }

    size_t get_input_size() const override {
        return m_op.GetOperands().size();
    }

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        std::string& producer_output_port_name,
                        size_t& producer_output_port_index) const override {
        producer_name.clear();
        producer_output_port_name.clear();
        producer_output_port_index = 0;
        const auto [name, idx] = m_inputs[input_port_idx];
        producer_name = name;
        producer_output_port_index = idx;
    }

    ov::Any get_attribute(const std::string& name) const override {
        return get_attr_val(m_op.TryGetAttribute(name), [](auto&& val) -> ov::Any {
            using T = std::remove_cvref_t<decltype(val)>;
            if constexpr (std::is_same_v<T, std::nullopt_t>) {
                return {};
            } else if constexpr (std::is_signed_v<T>) {
                return ov::Any(int64_t(val));
            } else if constexpr (std::is_unsigned_v<T>) {
                return ov::Any(uint64_t(val));
            } else if constexpr (is_span_v<T>) {
                using E = typename T::value_type;
                using U = std::conditional_t<std::is_integral_v<E>, std::conditional_t<std::is_signed_v<E>, int64_t, uint64_t>, E>;
                return ov::Any(std::vector<U>(val.begin(), val.end()));
            } else {
                return ov::Any(std::move(val));
            }
        });
    }

private:
    mutable Op<> m_op;
    std::string m_op_type;
    std::string m_op_name;
    std::vector<ValueProducerInfo> m_inputs;
};



class CgcGraphIterator : public GraphIterator {
public:
    CgcGraphIterator(const EntryPoint<>& entry_point,
                     std::span<const std::string> input_names,
                     std::span<const std::string> output_names)
        : m_entry_point(entry_point) {
        build_maps(input_names, output_names);
    }

    size_t size() const override { return m_nodes.size(); }
    void reset() override { m_current_index = 0; }
    void next() override { ++m_current_index; }
    bool is_end() const override {
        return m_current_index >= m_nodes.size();
    }

    std::shared_ptr<DecoderBase> get_decoder() const override {
        if (m_current_index >= m_nodes.size())
            return nullptr;
        auto node = m_nodes[m_current_index];
        return get_decoder(node, m_node_names[m_current_index]);
    }

    std::shared_ptr<GraphIterator> get_body_graph_iterator(const std::string&) const override {
        return nullptr;
    }

    std::vector<std::string> get_input_names() const override { return m_input_names; }
    std::vector<std::string> get_output_names() const override { 
        std::vector<std::string> names;
        names.reserve(m_outputs.size());
        for (const auto& info : m_outputs) {
            names.emplace_back(value_info_to_string(info));
        }
        return names; 
    }

    NodeOutputInfo get_value_info(void* handle) const {
        const auto it = m_value_map.find(handle);
        if (it != m_value_map.end())
            return it->second;
        return {};
    }

    std::string_view get_target_name(const NodeOutputInfo& info) const {
        switch (info.target) { 
        case NodeOutputInfo::TargetType::Null:
            return "";
        case NodeOutputInfo::TargetType::Input:
            return m_input_names[info.node_idx];
        default:
            return m_node_names[info.node_idx];
        }
    }

    std::string value_info_to_string(const NodeOutputInfo& info) const {
        if (info.target == NodeOutputInfo::TargetType::Null)
            return {};
        std::string ret(get_target_name(info));
        if (info.target == NodeOutputInfo::TargetType::Op) {
            ret.append(":").append(std::to_string(info.port_idx));
        }
        return ret;
    }

    std::shared_ptr<DecoderBase> get_decoder(Op<> node, std::string node_name) const {
        std::vector<ValueProducerInfo> input_infos;
        {
            for (const auto opr : node.GetOperands()) {
                const auto info = get_value_info(opr.GetHandle());
                if (info.target != NodeOutputInfo::TargetType::Null) {
                    input_infos.push_back({std::string(get_target_name(info)), info.port_idx});
                } else {
                    input_infos.push_back({});
                }
            }
        }
        return std::make_shared<CgcDecoder>(node, node_name, std::move(input_infos));
    }
    std::shared_ptr<DecoderBase> get_decoder(Op<> node) const {
        const auto info = get_value_info(node.GetResults()[0].GetHandle());
        return get_decoder(node, std::string(get_target_name(info)));
    }

    EntryPoint<> m_entry_point;
    std::vector<Op<>> m_nodes;
    std::vector<Op<>> m_ops;
    std::vector<Constant<>> m_constants;
    std::vector<std::string> m_node_names;
    size_t m_current_index = 0;
    std::unordered_map<void*, NodeOutputInfo> m_value_map;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
    std::vector<NodeOutputInfo> m_outputs;

private:
    void build_maps(std::span<const std::string> input_names, std::span<const std::string> output_names) {
        // model inputs
        {
            const auto args = m_entry_point.GetArguments();
            m_input_names.reserve(args.size());
            if (!input_names.empty()) {
                FRONT_END_GENERAL_CHECK(input_names.size() == args.size(), "Invalid input model for CGC frontend.");
                m_input_names.assign(input_names.begin(), input_names.end());
            }
            for (uint32_t i = 0; i < args.size(); ++i) {
                const auto& arg = args[i];
                if (input_names.empty()) {
                    m_input_names.push_back("arg_" + std::to_string(i));
                }
                m_value_map[arg.GetHandle()] = {NodeOutputInfo::TargetType::Input, i, 0};
            }
        }

        // child operations
        {
            auto ops = m_entry_point.GetOps();
            m_node_names.reserve(ops.size());
            m_nodes.reserve(ops.size());
            for (uint32_t i = 0; i < ops.size(); ++i) {
                auto op = ops[i];
                std::string name;
                const auto type = op.GetName().AsStringView();
                if (type == "cgc_op.constant") {
                    auto& cop = m_constants.emplace_back(op.As<ICgcMlirConstantOp>());
                    if (const auto datakey = cop.TryGetDataKey(); datakey) {
                        name = datakey->AsStringView();
                    } else {
                        name = "constant_" + std::to_string(i);
                    }
                    m_value_map[cop.GetResults()[0].GetHandle()] = {NodeOutputInfo::TargetType::Constant, i, 0};
                } else if (type == "cgc.return") {
                    auto rets = op.GetOperands();
                    for (const auto& ret : rets) {
                        m_outputs.emplace_back(get_value_info(ret.GetHandle()));
                    }
                } else if (type == "cgc_op.null_ptr") {
                } else {
                    m_ops.emplace_back(op);
                    const auto results = op.GetResults();
                    for (uint32_t j = 0; j < results.size(); ++j) {
                        m_value_map[results[j].GetHandle()] = {NodeOutputInfo::TargetType::Op, i, j};
                    }
                }
                if (name.empty())
                    name = "node_" + std::to_string(i);
                m_nodes.emplace_back(op);
                m_node_names.emplace_back(name);
            }
        }

        {
            const auto rets = m_entry_point.GetReturnTypes();
            FRONT_END_GENERAL_CHECK(m_outputs.size() == rets.size());
            if (!output_names.empty()) {
                FRONT_END_GENERAL_CHECK(output_names.size() == rets.size(), "Invalid output model for CGC frontend.");
                m_output_names.assign(output_names.begin(), output_names.end());
            }
        }
    }

};

struct CgcExtraData {
    std::span<const std::string> input_names;
    std::span<const std::string> output_names;
    std::map<std::string, std::span<const std::byte>, std::less<>> initializers;
};

class CgcInputModel : public InputModel {
public:
    CgcInputModel(winrt::com_ptr<ICgcMlirContext> context,
                  winrt::com_ptr<ICgcMlirModuleOp> module,
                  const CgcExtraData& extra_data,
                  std::filesystem::path model_path)
        : m_context(std::move(context)),
          m_module(std::move(module)),
          m_entry_point(m_module.GetEntryPoint()),
          m_graph_iterator(
              std::make_shared<CgcGraphIterator>(m_entry_point, extra_data.input_names, extra_data.output_names)),
          m_model_path(std::move(model_path)) {
        initialize(extra_data);
        print_model_info();
    }

    std::vector<Place::Ptr> get_inputs() const override { return m_inputs; }
    std::vector<Place::Ptr> get_outputs() const override { return m_outputs; }

    Place::Ptr get_place_by_tensor_name(const std::string& tensor_name) const override {
        auto it = m_tensor_name_to_place.find(tensor_name);
        return (it != m_tensor_name_to_place.end()) ? it->second : nullptr;
    }

    ov::PartialShape get_partial_shape(const Place::Ptr& place) const override {
        auto dp = std::dynamic_pointer_cast<CgcPlace>(place);
        return dp ? dp->get_shape() : ov::PartialShape::dynamic();
    }

    ov::element::Type get_element_type(const Place::Ptr& place) const override {
        auto dp = std::dynamic_pointer_cast<CgcPlace>(place);
        return dp ? dp->get_element_type() : ov::element::dynamic;
    }

    const std::shared_ptr<CgcGraphIterator>& get_graph_iterator() const {
        return m_graph_iterator;
    }

    const std::filesystem::path& get_model_path() const {
        return m_model_path;
    }

    std::span<const std::byte> get_external_resource_data(std::string_view resource_name) const {
        if (const auto it = m_initializers.find(resource_name); it != m_initializers.end()) {
            return it->second;
        }
        
        if (!m_resource_data) {
            load_resources();
        }
        
        if (const auto it = m_resource_data->find(resource_name); it != m_resource_data->end()) {
            return it->second;
        }
        return {};
    }

private:
    void initialize(const CgcExtraData& extra_data) {
        // inits
        {
            for (const auto& [k,v] : extra_data.initializers) {
                m_initializers.emplace(k, v);
            }
            for (const auto& [k, v] : extra_data.initializers) {
                std::string name;
                std::transform(k.begin(), k.end(), std::back_inserter(name), [](const char ch) {
                    return ch == '/' ? '_' : ch;
                });
                m_initializers.try_emplace(name, v);
                m_initializers.try_emplace("_" + name, v);
            }
        }

        // inputs
        {
            auto types = m_entry_point.GetArgumentTypes<ICgcMlirMultiDimType>();
            OPENVINO_ASSERT(types.size() == m_graph_iterator->m_input_names.size());
            m_inputs.reserve(types.size());
            for (size_t i = 0; i < types.size(); ++i) {
                const auto& name = m_graph_iterator->m_input_names[i];
                const auto type = types[i];
                const auto [ele_type, shape] = get_type_layout(type);
                auto place = std::make_shared<CgcPlace>(name, ele_type, shape);
                place->set_input(true);
                m_inputs.push_back(place);
                m_tensor_name_to_place[name] = place;
            }
        }

        // outputs
        {
            auto types = m_entry_point.GetReturnTypes<ICgcMlirMultiDimType>();
            OPENVINO_ASSERT(types.size() == m_graph_iterator->m_outputs.size());
            m_outputs.reserve(types.size());
            for (size_t i = 0; i < types.size(); ++i) {
                const auto& info = m_graph_iterator->m_outputs[i];
                const auto type = types[i];
                const auto [ele_type, shape] = get_type_layout(type);
                const auto name = m_graph_iterator->m_output_names.empty()
                                      ? m_graph_iterator->value_info_to_string(info)
                                      : m_graph_iterator->m_output_names[i];
                auto place = std::make_shared<CgcPlace>(name, ele_type, shape);
                place->set_output(true);
                m_outputs.push_back(place);
                m_tensor_name_to_place[name] = place;
            }
        }

        if (std::any_of(m_graph_iterator->m_constants.begin(), m_graph_iterator->m_constants.end(), [](auto& cop) {
                return cop->HasExternalData() == TRUE;
            })) {
            load_resources();
        }
    }

    void print_model_info() const {
        printf("=== DXGML Model Info ===\n");

        // -- Inputs --
        printf("Inputs (%zu):\n", m_inputs.size());
        for (size_t i = 0; i < m_inputs.size(); ++i) {
            auto dp = std::dynamic_pointer_cast<CgcPlace>(m_inputs[i]);
            printf("  [%3zu] %-15s  %s %s\n",
                   i, dp->get_names()[0].c_str(),
                   dp->get_element_type().get_type_name().c_str(),
                   dp->get_shape().to_string().c_str());
        }

        // -- Outputs --
        printf("Outputs (%zu):\n", m_outputs.size());
        for (size_t i = 0; i < m_outputs.size(); ++i) {
            auto dp = std::dynamic_pointer_cast<CgcPlace>(m_outputs[i]);
            printf("  [%3zu] %-15s  %s %s\n",
                   i, dp->get_names()[0].c_str(),
                   dp->get_element_type().get_type_name().c_str(),
                   dp->get_shape().to_string().c_str());
        }

        // -- Constants --
        printf("Constants (%zu):\n", m_graph_iterator->m_constants.size());
        for (auto& cop : m_graph_iterator->m_constants) {
            const auto ret = cop.GetResults()[0];
            const auto type = ret.GetType<ICgcMlirMultiDimType>();
            const auto info = m_graph_iterator->get_value_info(ret.GetHandle());
            const auto name = m_graph_iterator->get_target_name(info);
            const auto datakey = cop.TryGetDataKey();
            const auto [ele_type, shape] = get_type_layout(type);
            std::span<const std::byte> data;
            if (cop->HasExternalData() == FALSE) {
                data = cop.GetData().AsByteSpan();
            } else {
                OPENVINO_ASSERT(datakey.has_value(), "External constant should have a data key.");
                data = get_external_resource_data(datakey->AsStringView());
            }
            printf("  %-80s  %s %s",
                   datakey ? datakey->AsStringView().data() : std::string("(inline)").append(name).c_str(),
                   ele_type.get_type_name().c_str(),
                   shape.to_string().c_str());
            if (data.empty()) {
                printf("  not found\n");
            } else {
                printf("  (%zu)  %s\n", data.size(), serialize_n_ele(data, ele_type).c_str());
            }
        }

        // -- Operators --
        printf("Operators (%zu):\n", m_graph_iterator->m_ops.size());
        constexpr auto print_types = [](Range<ICgcMlirType, Type<>> types) {
            size_t i = 0;
            for (auto type : types) {
                winrt::com_ptr<ICgcMlirMultiDimType> mdt;
                if (SUCCEEDED(type.Get()->QueryInterface(IID_PPV_ARGS(&mdt)))) {
                    const auto [ele_type, shape] = get_type_layout(mdt);
                    printf("%s %s%s",
                           ele_type.get_type_name().c_str(),
                           shape.to_string().c_str(),
                           ++i == types.size() ? "" : ", ");
                } else {
                    const auto ele_type = map_element_type(type.GetKind());
                    printf("%s%s", ele_type.get_type_name().c_str(), ++i == types.size() ? "" : ", ");
                }
            }
        };
        size_t opidx = 0;
        for (auto& op : m_graph_iterator->m_ops) {
            auto oprts = op.GetOperandTypes();
            auto retts = op.GetResultTypes();
            const auto info = m_graph_iterator->get_value_info(op.GetResults()[0].GetHandle());
            std::string name(m_graph_iterator->get_target_name(info));
            printf("  [%zu]  %-40s (%s)\n    [%zu,%zu] (",
                   opidx++,
                   op.GetName().AsStringView().data(),
                   name.c_str(),
                   oprts.size(),
                   retts.size());
            print_types(oprts);
            printf(") -> (");
            print_types(retts);
            printf(")\n");
            auto attrs = op.GetAttributes();
            for (auto nattr : attrs) {
                const auto attr = nattr.GetAttribute();
                const auto typestr = attr_type_to_string(attr);
                const auto namestr = nattr.GetName().AsStringView();
                printf("    @@  [%s](%s): %s\n", namestr.data(), typestr.c_str(), attr_val_to_string(attr).c_str());
            }
            auto oprs = op.GetOperands();
            for (const auto opr : oprs) {
                const auto info = m_graph_iterator->get_value_info(opr.GetHandle());
                printf("    >>  %s", m_graph_iterator->value_info_to_string(info).c_str());
                if (info.target == NodeOutputInfo::TargetType::Op) {
                    const auto& prev_node = m_graph_iterator->m_nodes[info.node_idx];
                    printf(" (%s)\n", prev_node.GetName().AsStringView().data());
                } else {
                    printf("\n");
                }
            }
        }

        printf("========================\n");
        getchar();
    }

    std::filesystem::path get_resource_path() const {
        return m_model_path.parent_path() / "resources.mlir";
    }

    void load_resources() const {
        if (m_resource_data.has_value()) {
            return;
        }
        const auto resource_path = get_resource_path();
        if (!std::filesystem::exists(resource_path)) {
            printf("resource file [%s] not exists, will skip loading\n", resource_path.string().c_str());
            m_resource_data.emplace();
            return;
        }

        printf("loading resource file [%s]...\n", resource_path.string().c_str());
        const auto res = ov::load_mmap_object(resource_path);
        std::string_view content(res->data(), res->size());

        m_resource_data = parse_cgc_resources(content);
        FRONT_END_GENERAL_CHECK(m_resource_data.has_value(),
                                "Failed to parse CGC resources file: ",
                                resource_path.string());

        printf("pasred [%zu] tensor from external resources\n", m_resource_data->size());
    }

    Context<> m_context;
    Module<> m_module;
    EntryPoint<> m_entry_point;
    KVDict<std::span<const std::byte>> m_initializers;
    std::shared_ptr<CgcGraphIterator> m_graph_iterator;
    std::filesystem::path m_model_path;
    std::vector<Place::Ptr> m_inputs;
    std::vector<Place::Ptr> m_outputs;
    std::unordered_map<std::string, Place::Ptr> m_tensor_name_to_place;
    mutable std::optional<KVDict<std::vector<std::byte>>> m_resource_data;
};


std::shared_ptr<ov::Model> translate_dxgml_model(const std::shared_ptr<CgcInputModel>& input_model) {
    auto graph_iter = input_model->get_graph_iterator();
    const auto& translator_map = get_supported_ops();

    ov::ParameterVector parameters;
    std::unordered_map<uint32_t, std::shared_ptr<ov::op::v0::Constant>> constant_output_map;
    std::unordered_map<uint32_t, ov::OutputVector> op_output_map;

    const auto retrieve_output = [&](const NodeOutputInfo& info) -> ov::Output<ov::Node> {
        switch (info.target) {
        case NodeOutputInfo::TargetType::Null:
            return {};
        case NodeOutputInfo::TargetType::Input:
            return parameters[info.node_idx];
        case NodeOutputInfo::TargetType::Constant: {
            auto it = constant_output_map.find(info.node_idx);
            OPENVINO_ASSERT(it != constant_output_map.end());
            return it->second;
        }
        default: {
            auto it = op_output_map.find(info.node_idx);
            OPENVINO_ASSERT(it != op_output_map.end());
            return it->second[info.port_idx];
        }
        }
    };

    // inputs
    {
        const auto& inputs = input_model->get_inputs();
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto dp = std::dynamic_pointer_cast<CgcPlace>(inputs[i]);
            auto param = std::make_shared<ov::op::v0::Parameter>(dp->get_element_type(), dp->get_shape());
            auto name = dp->get_names()[0];
            param->set_friendly_name("input_" + std::to_string(i));
            param->output(0).set_names({name});
            parameters.push_back(param);
        }
        printf("===>created [%zu] inputs\n", inputs.size());
    }

    // constants
    {
        auto& constants = graph_iter->m_constants;
        for (size_t i = 0; i < constants.size(); ++i) {
            auto& cop = constants[i];
            const auto type = cop.GetResultTypes<ICgcMlirMultiDimType>()[0];
            const auto [ele_type, shape] = get_type_layout(type);
            const auto info = graph_iter->get_value_info(cop.GetResults()[0].GetHandle());
            std::string name(graph_iter->get_target_name(info));
            std::span<const std::byte> data;
            if (cop->HasExternalData() == FALSE) {
                data = cop.GetData().AsByteSpan();
            } else {
                const auto datakey = cop.TryGetDataKey();
                OPENVINO_ASSERT(datakey.has_value(), "External constant should have a data key.");
                data = input_model->get_external_resource_data(datakey->AsStringView());
            }
            //printf("===>constant[%s] [%zu](%p)\n", name.c_str(), data.size(), data.data());
            auto ov_const = data.empty()
                                ? std::make_shared<ov::op::v0::Constant>(ele_type, shape.get_shape())
                                : std::make_shared<ov::op::v0::Constant>(ele_type, shape.get_shape(), data.data());
            ov_const->set_friendly_name(name);
            ov_const->output(0).set_names({name});
            constant_output_map[info.node_idx] = ov_const;
        }
        printf("===>created [%zu] constants\n", constants.size());
    }

    // ops
    for (size_t idx = 0; idx < graph_iter->m_ops.size(); ++idx) {
        auto& op = graph_iter->m_ops[idx];
        //const auto& node_name = graph_iter->m_node_names[idx];

        const auto op_type = op.GetName().AsStringView();

        if (op_type == "cgc_op.return" || op_type == "cgc_op.null_ptr" || op_type == "cgc_op.constant") {
            printf("should be skipped: [%s]!\n", op_type.data());
            continue;
        }
        printf("===>creating [%zu](%s)\n", idx, op_type.data());

        ov::OutputVector inputs;
        {
            auto oprs = op.GetOperands();
            for (const auto opr : oprs) {
                const auto info = graph_iter->get_value_info(opr.GetHandle());
                auto input = retrieve_output(info);
                if (info.target == NodeOutputInfo::TargetType::Null) {
                    printf("  >> in[%zu] null\n", inputs.size());
                } else {
                    const auto [ele_type, shape] = get_type_layout(opr.GetType<>());
                    const auto real_type = input.get_element_type();
                    const auto realshape = input.get_partial_shape();
                    printf("  >> in[%zu] def[%s %s] real[%s %s](%s)\n",
                           inputs.size(),
                           ele_type.get_type_name().c_str(),
                           shape.to_string().c_str(),
                           real_type.get_type_name().c_str(),
                           realshape.to_string().c_str(),
                           input.get_node()->get_friendly_name().c_str());
                    if (shape.is_static()) {
                        OPENVINO_ASSERT(shape.compatible(realshape));
                    }
                }
                inputs.push_back(input);
            }
        }
        NodeOutputInfos outputs;
        {
            auto rets = op.GetResults();
            for (uint32_t i = 0; i < rets.size(); ++i) {
                outputs.push_back(get_type_layout(rets[i].GetType()));
            }
        }
        const auto decoder = graph_iter->get_decoder(op);

        auto out_name_base = decoder->get_op_name() + ":";

        auto translator_it = translator_map.find(op_type);
        OPENVINO_ASSERT(translator_it != translator_map.end(), "unsupported op");
        CgcNodeContext ctx(op_type, decoder);
        auto output_vec = translator_it->second(ctx, inputs, outputs);
        OPENVINO_ASSERT(output_vec.size() == outputs.size());
        for (uint32_t i = 0; i < outputs.size(); ++i) {
            const auto [ele_type, shape] = outputs[i];
            auto& out = output_vec[i];
            if (shape.is_static()) {
                const auto real_type = out.get_element_type();
                const auto realshape = out.get_partial_shape();
                printf("  << out[%u](%s) def[%s %s] real[%s %s]\n",
                       i,
                       out.get_node()->get_friendly_name().c_str(),
                       ele_type.get_type_name().c_str(),
                       shape.to_string().c_str(),
                       real_type.get_type_name().c_str(),
                       realshape.to_string().c_str());
                OPENVINO_ASSERT(shape.compatible(realshape));
                out.set_names({out_name_base + std::to_string(i)});
            } else {
                printf("  << out[%u](%s) def[%s %s]\n",
                       i,
                       out.get_node()->get_friendly_name().c_str(),
                       ele_type.get_type_name().c_str(),
                       shape.to_string().c_str());
            }
        }

        const auto info = graph_iter->get_value_info(op.GetResults()[0].GetHandle());
        std::string name(graph_iter->get_target_name(info));
        op_output_map[info.node_idx] = output_vec;
    }
    printf("===>created [%zu] ops\n", graph_iter->m_ops.size());

    // outputs
    ov::ResultVector results;
    {
        const auto& outs = graph_iter->m_outputs;
        for (size_t i = 0; i < outs.size(); ++i) {
            const auto& info = outs[i];
            const auto input = retrieve_output(info);
            auto result = std::make_shared<ov::op::v0::Result>(input);
            if (!graph_iter->m_output_names.empty()) {
                const auto& name = graph_iter->m_output_names[i];
                result->set_friendly_name(name);
                result->output(0).set_names({name});
            }
            results.push_back(result);
        }
    }

    auto model = std::make_shared<ov::Model>(results, parameters, "cgc_model");
    printf("===translate finished\n");
    {
        auto xml_path = input_model->get_model_path();
        xml_path.replace_extension(".xml");
        auto bin_path = xml_path;
        bin_path.replace_extension(".bin");
        ov::pass::Serialize serialize(xml_path, bin_path);
        serialize.run_on_model(model);
    }
    getchar();
    return model;
}


class CgcFrontEnd : public FrontEnd {
public:
    CgcFrontEnd() = default;
    std::string get_name() const override { return "cgc"; }

protected:
    bool supported_impl(const std::vector<ov::Any>& variants) const override {
        if (variants.empty()) return false;
        if (variants[0].is<std::string>()) {
            auto ext = std::filesystem::path(variants[0].as<std::string>()).extension().string();
            return ext == ".mlir";
        }
        if (variants[0].is<std::filesystem::path>()) {
            auto ext = variants[0].as<std::filesystem::path>().extension().string();
            return ext == ".mlir";
        }
        if (variants[0].is<std::istream*>()) {
            const auto stream = variants[0].as<std::istream*>();
            stream->seekg(0, stream->beg);
            char tmp[128] = {0};
            stream->read(tmp, 120);
            stream->seekg(0, stream->beg);
            return std::string_view(tmp).find("cgc.module") != std::string_view::npos;
        }
        return false;
    }

    std::shared_ptr<ov::Model> convert(const InputModel::Ptr& model) const override {
        auto dxgml_model = std::dynamic_pointer_cast<CgcInputModel>(model);
        FRONT_END_GENERAL_CHECK(dxgml_model, "Invalid input model for CGC frontend.");
        return translate_dxgml_model(dxgml_model);
    }

    template<typename T = void, typename D>
    static void extract_variant(const std::vector<ov::Any>& variants, size_t idx, D& dst) {
        using U = std::conditional_t<std::is_void_v<T>, std::remove_cvref_t<D>, T>;
        if (variants.size() > idx) {
            FRONT_END_GENERAL_CHECK(variants[idx].is<U>(),
                                    "Unsupported variant type for CGC extra data ",
                                    std::to_string(idx));
            dst = variants[idx].as<U>();
        }
    };

    InputModel::Ptr load_impl(const std::vector<ov::Any>& variants) const override {
        FRONT_END_GENERAL_CHECK(!variants.empty(), "No arguments provided to load CGC model.");

        std::filesystem::path model_path;
        std::istream* model_stream = nullptr;
        CgcExtraData extra_data;
        if (variants[0].is<std::string>())
            model_path = variants[0].as<std::string>();
        else if (variants[0].is<std::filesystem::path>())
            model_path = variants[0].as<std::filesystem::path>();
        else if (variants[0].is<std::istream*>()) {
            model_stream = variants[0].as<std::istream*>();
            if (variants.size() > 1) {
                extract_variant<std::string>(variants, 1, model_path);
                extract_variant<std::vector<std::string>>(variants, 2, extra_data.input_names);
                extract_variant<std::vector<std::string>>(variants, 3, extra_data.output_names);
                extract_variant(variants, 4, extra_data.initializers);
            }
        }
        else
            FRONT_END_GENERAL_CHECK(false, "Unsupported variant type for CGC frontend.");
        
        std::optional<std::ifstream> file;
        if (!model_stream) {
            validate_path(model_path);
            file.emplace(model_path, std::ios::binary);
            FRONT_END_GENERAL_CHECK(file->is_open(), "Cannot open CGC model file: " + model_path.string());
            model_stream = &*file;
        }
        std::vector<char> model_text;
        model_stream->seekg(0, model_stream->end);
        const auto fsize = model_stream->tellg();
        model_stream->seekg(0, std::ios::beg);
        model_text.resize(static_cast<size_t>(fsize));
        model_stream->read(model_text.data(), fsize);
        FRONT_END_GENERAL_CHECK(model_stream->good(), "Failed to read CGC model content.");

        winrt::com_ptr<ICgcMlirContext> context;
        THROW_IF_FAILED(CgcCreateMlirContext(IID_PPV_ARGS(&context)));

        winrt::com_ptr<ICgcMlirModuleOp> module;
        THROW_IF_FAILED(
            context->Deserialize(model_text.data(), static_cast<UINT64>(model_text.size()), IID_PPV_ARGS(&module)));

        return std::make_shared<CgcInputModel>(std::move(context), std::move(module), extra_data, model_path);
    }
};

}  // namespace cgc
}  // namespace frontend
}  // namespace ov


CGC_FRONTEND_C_API ov::frontend::FrontEndVersion get_api_version() {
    return OV_FRONTEND_API_VERSION;
}

CGC_FRONTEND_C_API void* get_front_end_data() {
    auto* res = new ov::frontend::FrontEndPluginInfo();
    res->m_name = "cgc";
    res->m_creator = []() {
        return std::make_shared<ov::frontend::cgc::CgcFrontEnd>();
    };
    return res;
}
