// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

// --------------------------------------------------------------------------------------------------------------------
// CGC IR C++ Helper API Templates
//
// THIS API NOT GUARANTEED TO BE STABLE AND MAY HAVE BREAKING CHANGES.
//
// This API complements the CGC IR C++ Helper API defined in CGC_ir.h by providing C++ template wrappers for common
// patterns of usage. These templates aim to improve usability and type safety when working with CGC IR COM-style
// interfaces.
// --------------------------------------------------------------------------------------------------------------------

#include "cgc_ir.h"

#include <span>
#include <vector>
#include <wil/result.h>
#include <winrt/base.h>

namespace cgc_ir_templates
{

// clang-format off

constexpr inline size_t GetDataTypeSizeInBits(CGC_IR_TYPE dataType)
{
    switch (dataType)
    {
    case CGC_IR_TYPE_BOOL8:
        return 8;

    case CGC_IR_TYPE_INT2:
    case CGC_IR_TYPE_UINT2:
        return 2;

    case CGC_IR_TYPE_INT4:
    case CGC_IR_TYPE_UINT4:
    case CGC_IR_TYPE_FLOAT4E2M1FN:
        return 4;

    case CGC_IR_TYPE_FLOAT6E2M3FN:
    case CGC_IR_TYPE_FLOAT6E3M2FN:
        return 6;

    case CGC_IR_TYPE_INT8:
    case CGC_IR_TYPE_UINT8:
    case CGC_IR_TYPE_FLOAT8E5M2:
    case CGC_IR_TYPE_FLOAT8E4M3:
    case CGC_IR_TYPE_FLOAT8E4M3FN:
    case CGC_IR_TYPE_FLOAT8E5M2FNUZ:
    case CGC_IR_TYPE_FLOAT8E4M3FNUZ:
    case CGC_IR_TYPE_FLOAT8E4M3B11FNUZ:
    case CGC_IR_TYPE_FLOAT8E3M4:
    case CGC_IR_TYPE_FLOAT8E8M0FNU:
        return 8;

    case CGC_IR_TYPE_INT16:
    case CGC_IR_TYPE_UINT16:
    case CGC_IR_TYPE_FLOAT16:
    case CGC_IR_TYPE_BFLOAT16:
        return 16;

    case CGC_IR_TYPE_INT32:
    case CGC_IR_TYPE_UINT32:
    case CGC_IR_TYPE_FLOAT32:
        return 32;

    case CGC_IR_TYPE_INT64:
    case CGC_IR_TYPE_UINT64:
    case CGC_IR_TYPE_FLOAT64:
        return 64;

    default:
        throw std::invalid_argument("Invalid data type");
    };
}

template <typename Interface> struct Context;
template <typename Interface> struct Type;
template <typename Interface> struct Value;
template <typename Interface> struct Blob;
template <typename Interface> struct MultiDimType;

template <typename Interface> struct Op;
template <typename Interface> struct Module;
template <typename Interface> struct Function;
template <typename Interface> struct EntryPoint;
template <typename Interface> struct Program;
template <typename Interface> struct Invoke;
template <typename Interface> struct Partition;
template <typename Interface> struct Subgraph;
template <typename Interface> struct FunctionalDefinition;
template <typename Interface> struct Constant;
template <typename Interface> struct ViewMemref;
template <typename Interface> struct Copy;

template <typename Interface> struct Attribute;
template <typename Interface> struct TypedAttribute;
template <typename Interface> struct ScalarAttribute;
template <typename Interface> struct StringAttribute;
template <typename Interface> struct EnumAttribute;
template <typename Interface> struct MultiDimAttribute;
template <typename Interface> struct MemoryLayoutAttribute;
template <typename Interface> struct PhysicalLayout;
template <typename Interface> struct NamedAttribute;

template <typename T>
using OpWrapper = 
    std::conditional_t<std::is_base_of_v<ICgcMlirModuleOp, T>, Op<T>,
    std::conditional_t<std::is_base_of_v<ICgcMlirEntryPointOp, T>, EntryPoint<T>,
    std::conditional_t<std::is_base_of_v<ICgcMlirProgramOp, T>, Program<T>,
    std::conditional_t<std::is_base_of_v<ICgcMlirInvokeOp, T>, Invoke<T>, 
    std::conditional_t<std::is_base_of_v<ICgcMlirPartitionOp, T>, Partition<T>,
    std::conditional_t<std::is_base_of_v<ICgcMlirSubgraphOp, T>, Subgraph<T>,
    std::conditional_t<std::is_base_of_v<ICgcMlirFunctionalDefinitionOp, T>, FunctionalDefinition<T>,
    std::conditional_t<std::is_base_of_v<ICgcMlirConstantOp, T>, Constant<T>, 
    std::conditional_t<std::is_base_of_v<ICgcMlirViewMemrefOp, T>, ViewMemref<T>,
    std::conditional_t<std::is_base_of_v<ICgcMlirCopyOp, T>, Copy<T>,
    Op<T>
>>>>>>>>>>;

template <typename T>
using TypeWrapper = 
    std::conditional_t<std::is_base_of_v<ICgcMlirMultiDimType, T>, MultiDimType<T>, Type<T>>;

template <typename T> constexpr CGC_IR_OPERATION GetOpTypeEnum()
{
    if constexpr (std::is_same_v<T, ICgcMlirModuleOp>)          return CGC_IR_OPERATION_MODULE;
    else if constexpr (std::is_same_v<T, ICgcMlirEntryPointOp>) return CGC_IR_OPERATION_ENTRY_POINT;
    else if constexpr (std::is_same_v<T, ICgcMlirProgramOp>)    return CGC_IR_OPERATION_PROGRAM;
    else if constexpr (std::is_same_v<T, ICgcMlirInvokeOp>)     return CGC_IR_OPERATION_INVOKE;
    else if constexpr (std::is_same_v<T, ICgcMlirPartitionOp>)  return CGC_IR_OPERATION_PARTITION;
    else if constexpr (std::is_same_v<T, ICgcMlirSubgraphOp>)   return CGC_IR_OPERATION_SUBGRAPH;
    else if constexpr (std::is_same_v<T, ICgcMlirFunctionalDefinitionOp>) return CGC_IR_OPERATION_FUNCTIONAL_DEFINITION;
    else if constexpr (std::is_same_v<T, ICgcMlirConstantOp>)   return CGC_IR_OPERATION_CONSTANT;
    else if constexpr (std::is_same_v<T, ICgcMlirViewMemrefOp>) return CGC_IR_OPERATION_VIEW_MEMREF;
    else if constexpr (std::is_same_v<T, ICgcMlirCopyOp>)     return CGC_IR_OPERATION_COPY;

    return CGC_IR_OPERATION_UNKNOWN;
};

template <typename T>
using AttributeWrapper = 
    std::conditional_t<std::is_base_of_v<ICgcMlirMemoryLayoutAttribute, T>, MemoryLayoutAttribute<T>,
    std::conditional_t<std::is_base_of_v<ICgcMlirStringAttribute, T>, StringAttribute<T>,
    std::conditional_t<std::is_base_of_v<ICgcMlirEnumAttribute, T>, EnumAttribute<T>,
    std::conditional_t<std::is_base_of_v<ICgcMlirScalarAttribute, T>, ScalarAttribute<T>,
    std::conditional_t<std::is_base_of_v<ICgcMlirMultiDimAttribute, T>, MultiDimAttribute<T>,
    std::conditional_t<std::is_base_of_v<ICgcMlirTypedAttribute, T>, TypedAttribute<T>,
    Attribute<T>
>>>>>>;

template <typename T, typename Wrapper> class Range
{
    winrt::com_ptr<ICgcMlirRange> range_;

  public:
    explicit Range(ICgcMlirRange *range)
    {
        range_.copy_from(range);
    }

    explicit Range(winrt::com_ptr<ICgcMlirRange> &&range) : range_(std::move(range))
    {
    }

    ICgcMlirRange* Get() const
    {
        return range_.get();
    }

    class Iterator
    {
        winrt::com_ptr<ICgcMlirRange> range_;
        UINT32 index_;

      public:
        using iterator_category = std::input_iterator_tag;
        using value_type = winrt::com_ptr<T>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type *;
        using reference = value_type &;

        Iterator(ICgcMlirRange *range, UINT32 index) : index_(index)
        {
            range_.copy_from(range);
        }

        Wrapper operator*() const
        {
            winrt::com_ptr<T> item;
            THROW_IF_FAILED(range_->Get(index_, IID_PPV_ARGS(&item)));
            return Wrapper(item);
        }

        Iterator &operator++()
        {
            ++index_;
            return *this;
        }
        Iterator operator++(int)
        {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const Iterator &other) const
        {
            return index_ == other.index_;
        }
        bool operator!=(const Iterator &other) const
        {
            return !(*this == other);
        }
    };

    Iterator begin() const
    {
        return Iterator(range_.get(), 0);
    }

    Iterator end() const
    {
        return Iterator(range_.get(), UINT32(range_->GetSize()));
    }

    SIZE_T size() const
    {
        return range_->GetSize();
    }

    Wrapper operator[](SIZE_T index) const
    {
        winrt::com_ptr<T> item;
        THROW_IF_FAILED(range_->Get(static_cast<UINT32>(index), IID_PPV_ARGS(&item)));
        return Wrapper(std::move(item));
    }
};

//
// Convenience wrapper for an ICgcMlirString-derived interfaces.
//
template <typename Interface = ICgcMlirString> struct String
{
    winrt::com_ptr<Interface> m_impl;

    String(Interface *impl) { m_impl.copy_from(impl); }
    String(const winrt::com_ptr<Interface> &impl) : m_impl(impl) {}
    String(winrt::com_ptr<Interface> &&impl) : m_impl(std::move(impl)) {}

    // View the string data as a string view. The memory lifetime is tied to the ICgcMlirContext; so long as the context
    // is alive (or there are refs to it through ICgcMlirContextChild-derived interfaces), the data pointer is valid.
    std::string_view AsStringView() const
    {
        return std::string_view(m_impl->GetData(), m_impl->GetDataSize());
    }
};

//
// Convenience wrapper for an ICgcMlirSpan-derived interfaces.
//
template <typename Interface = ICgcMlirSpan> struct Span
{
    winrt::com_ptr<Interface> m_impl;

    Span(Interface *impl) { m_impl.copy_from(impl); }
    Span(const winrt::com_ptr<Interface> &impl) : m_impl(impl) {}
    Span(winrt::com_ptr<Interface> &&impl) : m_impl(std::move(impl)) {}

    // View the span data as a span of T. The memory lifetime is tied to the ICgcMlirContext; so long as the context
    // is alive (or there are refs to it through ICgcMlirContextChild-derived interfaces), the data pointer is valid.
    template <typename T> std::span<const T> AsSpan() const
    {
        return std::span<const T>(static_cast<const T *>(m_impl->GetData()), m_impl->GetDataSizeInBytes() / sizeof(T));
    }

    // View the span data as a span of bytes. The memory lifetime is tied to the ICgcMlirContext; so long as the context
    // is alive (or there are refs to it through ICgcMlirContextChild-derived interfaces), the data pointer is valid.
    std::span<const std::byte> AsByteSpan() const
    {
        return AsSpan<std::byte>();
    }

    // View the span data as a string view. The memory lifetime is tied to the ICgcMlirContext; so long as the context
    // is alive (or there are refs to it through ICgcMlirContextChild-derived interfaces), the data pointer is valid.
    std::string_view AsStringView() const
    {
        return std::string_view(static_cast<const char *>(m_impl->GetData()), m_impl->GetDataSizeInBytes());
    }
};

//
// Convenience wrapper for an ICgcBlob-derived interfaces.
//
template <typename Interface = ICgcBlob> struct Blob
{
    winrt::com_ptr<Interface> m_impl;

    Blob(Interface *impl) { m_impl.copy_from(impl); }
    Blob(const winrt::com_ptr<Interface> &impl) : m_impl(impl) {}
    Blob(winrt::com_ptr<Interface> &&impl) : m_impl(std::move(impl)) {}

    // View the blob data as a span of T. The memory lifetime is tied to the blob itself, so the caller must keep a
    // reference on the ICgcBlob instance while using the returned span.
    template <typename T> std::span<const T> AsSpan() const
    {
        return std::span<const T>(static_cast<const T *>(m_impl->GetData()), m_impl->GetDataSizeInBytes() / sizeof(T));
    }

    // View the blob data as a span of bytes. The memory lifetime is tied to the blob itself, so the caller must keep a
    // reference on the ICgcBlob instance while using the returned span.
    std::span<const std::byte> AsByteSpan() const
    {
        return AsSpan<std::byte>();
    }

    // View the blob data as a string view. The memory lifetime is tied to the blob itself, so the caller must keep a
    // reference on the ICgcBlob instance while using the returned span.
    std::string_view AsStringView() const
    {
        return std::string_view(static_cast<const char *>(m_impl->GetData()), m_impl->GetDataSizeInBytes());
    }

    // View the blob data as a new string.
    std::string AsString() const
    {
        return std::string(AsStringView());
    }
};

//
// Convenience wrapper for an ICgcMlirContext-derived interfaces.
//
template <typename Interface = ICgcMlirContext> struct Context
{
    winrt::com_ptr<Interface> m_impl;

    Context(Interface *impl) { m_impl.copy_from(impl); }
    Context(const winrt::com_ptr<Interface> &impl) : m_impl(impl) {}
    Context(winrt::com_ptr<Interface> &&impl) : m_impl(std::move(impl)) {}

    auto Deserialize(std::span<const std::byte> data) const
    {
        winrt::com_ptr<ICgcMlirModuleOp> moduleOp;
        THROW_IF_FAILED(m_impl->Deserialize(data.data(), data.size(), IID_PPV_ARGS(&moduleOp)));
        return OpWrapper<ICgcMlirModuleOp>(std::move(moduleOp));
    }

    auto Deserialize(const Blob<> &blob) const
    {
        return Deserialize(blob.AsByteSpan());
    }
};

//
// Convenience wrapper for an ICgcMlirType-derived interfaces.
//
template <typename Interface = ICgcMlirType> struct Type
{
    winrt::com_ptr<Interface> m_impl;

    Type(Interface *impl) { m_impl.copy_from(impl); }
    Type(const winrt::com_ptr<Interface> &impl) : m_impl(impl) {}
    Type(winrt::com_ptr<Interface> &&impl) : m_impl(std::move(impl)) {}

    Interface *Get() const
    {
        return m_impl.get();
    }

    operator bool() const
    {
        return m_impl != nullptr;
    }

    // Returns the type's kind or CGC_IR_TYPE_UNKNOWN if it's not recognized as a more specific type.
    CGC_IR_TYPE GetKind() const
    {
        return m_impl->GetKind();
    }

    // Serializes the wrapped MLIR object to the specified encoding format.
    auto Serialize(CGC_ENCODING encoding) const
    {
        winrt::com_ptr<ICgcBlob> blob;
        THROW_IF_FAILED(m_impl->Serialize(encoding, IID_PPV_ARGS(&blob)));
        return Blob(std::move(blob));
    }
};

//
// Convenience wrapper for an ICgcMlirMultiDimType-derived interfaces.
//
template <typename Interface = ICgcMlirMultiDimType> struct MultiDimType : public Type<Interface>
{
    MultiDimType(Interface *impl) : Type<Interface>(impl) {}
    MultiDimType(const winrt::com_ptr<Interface> &impl) : Type<Interface>(impl) {}
    MultiDimType(winrt::com_ptr<Interface> &&impl) : Type<Interface>(std::move(impl)) {}

    // Returns the rank (number of dimensions).
    SIZE_T GetRank() const
    {
        return this->m_impl->GetRank();
    }

    // Returns the total size in bytes.
    SIZE_T GetSizeInBytes() const
    {
        return this->m_impl->GetSizeInBytes();
    }

    // Returns the total number of elements (product of dimensions).
    SIZE_T GetSizeInElements() const
    {
        return this->m_impl->GetSizeInElements();
    }

    // Returns the data type of the elements.
    CGC_IR_TYPE GetDataType() const
    {
        return this->m_impl->GetDataType();
    }

    // Returns the data type of dimension sizes.
    CGC_IR_TYPE GetIndexType() const
    {
        return this->m_impl->GetIndexType();
    }

    // Returns the size of all dimensions.
    auto GetShape() const
    {
        winrt::com_ptr<ICgcMlirSpan> shape;
        THROW_IF_FAILED(this->m_impl->GetShape(IID_PPV_ARGS(&shape)));
        return Span(std::move(shape));
    }

    // Returns whether the type is marked as constant.
    bool IsConstant() const
    {
        return this->m_impl->IsConstant() != FALSE;
    }
};

//
// Convenience wrapper for an ICgcMlirAttribute-derived interfaces.
//
template <typename Interface = ICgcMlirAttribute> struct Attribute
{
    winrt::com_ptr<Interface> m_impl;

    Attribute(Interface *impl) { m_impl.copy_from(impl); }
    Attribute(const winrt::com_ptr<Interface> &impl) : m_impl(impl) {}
    Attribute(winrt::com_ptr<Interface> &&impl) : m_impl(std::move(impl)) {}

    operator bool() const
    {
        return this->m_impl != nullptr;
    }

    Interface *Get() const
    {
        return m_impl.get();
    }

    CGC_IR_ATTRIBUTE GetKind() const
    {
        return m_impl->GetKind();
    }

    auto GetAttributeTypeName() const
    {
        winrt::com_ptr<ICgcMlirString> str;
        THROW_IF_FAILED(m_impl->GetAttributeTypeName(str.put()));
        return String(std::move(str));
    }

    // Serializes the wrapped MLIR object to the specified encoding format.
    auto Serialize(CGC_ENCODING encoding) const
    {
        winrt::com_ptr<ICgcBlob> blob;
        THROW_IF_FAILED(m_impl->Serialize(encoding, IID_PPV_ARGS(&blob)));
        return Blob(std::move(blob));
    }
};

//
// Convenience wrapper for an ICgcMlirTypedAttribute-derived interfaces.
//
template <typename Interface = ICgcMlirTypedAttribute> struct TypedAttribute : public Attribute<Interface>
{
    TypedAttribute(Interface *impl) : Attribute<Interface>(impl) {}
    TypedAttribute(const winrt::com_ptr<Interface> &impl) : Attribute<Interface>(impl) {}
    TypedAttribute(winrt::com_ptr<Interface> &&impl) : Attribute<Interface>(std::move(impl)) {}

    template <typename T = ICgcMlirType> auto GetType() const
    {
        winrt::com_ptr<T> type;
        THROW_IF_FAILED(this->m_impl->GetType(IID_PPV_ARGS(&type)));
        return TypeWrapper<T>(std::move(type));
    }
};

//
// Convenience wrapper for an ICgcMlirScalarAttribute-derived interfaces.
//
template <typename Interface = ICgcMlirScalarAttribute> struct ScalarAttribute : public TypedAttribute<Interface>
{
    ScalarAttribute(Interface *impl) : TypedAttribute<Interface>(impl) {}
    ScalarAttribute(const winrt::com_ptr<Interface> &impl) : TypedAttribute<Interface>(impl) {}
    ScalarAttribute(winrt::com_ptr<Interface> &&impl) : TypedAttribute<Interface>(std::move(impl)) {}

    bool GetBoolValue() const
    {
        BOOL value = FALSE;
        THROW_IF_FAILED(this->m_impl->GetBoolValue(&value));
        return static_cast<bool>(value);
    }

    int64_t GetInt64Value() const
    {
        int64_t value = 0;
        THROW_IF_FAILED(this->m_impl->GetInt64Value(&value));
        return value;
    }

    int32_t GetInt32Value() const
    {
        int32_t value = 0;
        THROW_IF_FAILED(this->m_impl->GetInt32Value(&value));
        return value;
    }

    int16_t GetInt16Value() const
    {
        int16_t value = 0;
        THROW_IF_FAILED(this->m_impl->GetInt16Value(&value));
        return value;
    }

    int8_t GetInt8Value() const
    {
        int8_t value = 0;
        THROW_IF_FAILED(this->m_impl->GetInt8Value(&value));
        return value;
    }

    uint64_t GetUInt64Value() const
    {
        uint64_t value = 0;
        THROW_IF_FAILED(this->m_impl->GetUint64Value(&value));
        return value;
    }

    uint32_t GetUInt32Value() const
    {
        uint32_t value = 0;
        THROW_IF_FAILED(this->m_impl->GetUint32Value(&value));
        return value;
    }

    uint16_t GetUInt16Value() const
    {
        uint16_t value = 0;
        THROW_IF_FAILED(this->m_impl->GetUint16Value(&value));
        return value;
    }

    uint8_t GetUInt8Value() const
    {
        uint8_t value = 0;
        THROW_IF_FAILED(this->m_impl->GetUint8Value(&value));
        return value;
    }

    double GetFloat64Value() const
    {
        double value = 0.0;
        THROW_IF_FAILED(this->m_impl->GetFloat64Value(&value));
        return value;
    }

    float GetFloat32Value() const
    {
        float value = 0.0f;
        THROW_IF_FAILED(this->m_impl->GetFloat32Value(&value));
        return value;
    }
};

//
// Convenience wrapper for an ICgcMlirMultiDimAttribute-derived interfaces.
//
template <typename Interface = ICgcMlirMultiDimAttribute> struct MultiDimAttribute : public TypedAttribute<Interface>
{
    MultiDimAttribute(Interface *impl) : TypedAttribute<Interface>(impl) {}
    MultiDimAttribute(const winrt::com_ptr<Interface> &impl) : TypedAttribute<Interface>(impl) {}
    MultiDimAttribute(winrt::com_ptr<Interface> &&impl) : TypedAttribute<Interface>(std::move(impl)) {}

    const void* GetValue() const
    {
        return this->m_impl->GetValue();
    }

    SIZE_T GetValueSize() const
    {
        return this->m_impl->GetValueSize();
    }

    template <typename T = ICgcMlirMultiDimType> auto GetType() const
    {
        winrt::com_ptr<T> type;
        THROW_IF_FAILED(this->m_impl->GetType(IID_PPV_ARGS(&type)));
        return TypeWrapper<T>(std::move(type));
    }

    // View the data as a span of T.
    template <typename T> std::span<const T> AsSpan() const
    {
        return std::span<const T>(static_cast<const T*>(GetValue()), GetValueSize());
    }
};

//
// Convenience wrapper for an ICgcMlirStringAttribute-derived interfaces.
//
template <typename Interface = ICgcMlirStringAttribute> struct StringAttribute : public Attribute<Interface>
{
    StringAttribute(Interface *impl) : Attribute<Interface>(impl) {}
    StringAttribute(const winrt::com_ptr<Interface> &impl) : Attribute<Interface>(impl) {}
    StringAttribute(winrt::com_ptr<Interface> &&impl) : Attribute<Interface>(std::move(impl)) {}

    std::string_view AsStringView() const
    {
        return std::string_view(this->m_impl->GetValue(), this->m_impl->GetValueSize());
    }

    std::string AsString() const
    {
        return std::string(AsStringView());
    }
};

//
// Convenience wrapper for an ICgcMlirEnumAttribute-derived interfaces.
//
template <typename Interface = ICgcMlirEnumAttribute> struct EnumAttribute : public Attribute<Interface>
{
    EnumAttribute(Interface *impl) : Attribute<Interface>(impl) {}
    EnumAttribute(const winrt::com_ptr<Interface> &impl) : Attribute<Interface>(impl) {}
    EnumAttribute(winrt::com_ptr<Interface> &&impl) : Attribute<Interface>(std::move(impl)) {}

    std::string_view AsStringView() const
    {
        return std::string_view(this->m_impl->GetStringValue(), this->m_impl->GetStringValueSize());
    }

    std::string AsString() const
    {
        return std::string(AsStringView());
    }
};

//
// Convenience wrapper for an ICgcMlirMemoryLayoutAttribute-derived interfaces.
//
template <typename Interface = ICgcMlirMemoryLayoutAttribute> struct MemoryLayoutAttribute : public Attribute<Interface>
{
    MemoryLayoutAttribute(Interface *impl) : Attribute<Interface>(impl) {}
    MemoryLayoutAttribute(const winrt::com_ptr<Interface> &impl) : Attribute<Interface>(impl) {}
    MemoryLayoutAttribute(winrt::com_ptr<Interface> &&impl) : Attribute<Interface>(std::move(impl)) {}

    bool IsIdentity() const
    {
        return this->m_impl->IsIdentity() != FALSE;
    }

    std::optional<Span<>> TryGetElementAlignments() const
    {
        winrt::com_ptr<ICgcMlirSpan> span;
        if (SUCCEEDED(this->m_impl->TryGetElementAlignments(IID_PPV_ARGS(&span))) && span)
        {
            return Span(std::move(span));
        }
        return std::nullopt;
    }

    std::optional<Span<>> TryGetByteAlignments() const
    {
        winrt::com_ptr<ICgcMlirSpan> span;
        if (SUCCEEDED(this->m_impl->TryGetByteAlignments(IID_PPV_ARGS(&span))) && span)
        {
            return Span(std::move(span));
        }
        return std::nullopt;
    }

    template <typename T = ICgcMlirScalarAttribute> std::optional<AttributeWrapper<T>> TryGetFillValue() const
    {
        winrt::com_ptr<T> attr;
        if (SUCCEEDED(this->m_impl->TryGetFillValue(IID_PPV_ARGS(&attr))) && attr)
        {
            return AttributeWrapper<T>(std::move(attr));
        }
        return std::nullopt;
    }

    std::vector<CGC_IR_PHYSICAL_ORDER_ENTRY> GetPhysicalOrderEntries() const
    {
        std::vector<CGC_IR_PHYSICAL_ORDER_ENTRY> entries(this->m_impl->GetPhysicalOrderEntryCount());
        if (!entries.empty())
        {
            THROW_IF_FAILED(this->m_impl->GetPhysicalOrderEntries(entries.data(), entries.size()));
        }
        return entries;
    }
};

//
// Convenience wrapper for an ICgcMlirPhysicalLayout-derived interfaces.
//
template <typename Interface = ICgcMlirPhysicalLayout> struct PhysicalLayout
{
    winrt::com_ptr<Interface> m_impl;

    PhysicalLayout(Interface *impl) { m_impl.copy_from(impl); }
    PhysicalLayout(const winrt::com_ptr<Interface> &impl) : m_impl(impl) {}
    PhysicalLayout(winrt::com_ptr<Interface> &&impl) : m_impl(std::move(impl)) {}

    operator bool() const
    {
        return m_impl != nullptr;
    }

    Interface *Get() const
    {
        return m_impl.get();
    }

    SIZE_T GetRank() const
    {
        return m_impl->GetRank();
    }

    auto GetShape() const
    {
        winrt::com_ptr<ICgcMlirSpan> span;
        THROW_IF_FAILED(m_impl->GetShape(IID_PPV_ARGS(&span)));
        return Span(std::move(span));
    }

    SIZE_T GetSizeInElements() const
    {
        return m_impl->GetSizeInElements();
    }

    SIZE_T GetSizeInBytes() const
    {
        return m_impl->GetSizeInBytes();
    }
};

//
// Convenience wrapper for an ICgcMlirNamedAttribute-derived interfaces.
//
template <typename Interface = ICgcMlirNamedAttribute> struct NamedAttribute
{
    winrt::com_ptr<Interface> m_impl;

    NamedAttribute(Interface *impl) { m_impl.copy_from(impl); }
    NamedAttribute(const winrt::com_ptr<Interface> &impl) : m_impl(impl) {}
    NamedAttribute(winrt::com_ptr<Interface> &&impl) : m_impl(std::move(impl)) {}

    operator bool() const
    {
        return this->m_impl != nullptr;
    }

    Interface *Get() const
    {
        return m_impl.get();
    }

    auto GetName() const
    {
        winrt::com_ptr<ICgcMlirString> str;
        THROW_IF_FAILED(m_impl->GetName(str.put()));
        return String(std::move(str));
    }

    template <typename T = ICgcMlirAttribute> auto GetAttribute() const
    {
        winrt::com_ptr<T> attribute;
        THROW_IF_FAILED(m_impl->GetAttribute(IID_PPV_ARGS(&attribute)));
        return AttributeWrapper<T>(std::move(attribute));
    }
};

//
// Convenience wrapper for an ICgcMlirValue-derived interfaces.
//
template <typename Interface = ICgcMlirValue> struct Value
{
    winrt::com_ptr<Interface> m_impl;

    Value(Interface *impl) { m_impl.copy_from(impl); }
    Value(const winrt::com_ptr<Interface> &impl) : m_impl(impl) {}
    Value(winrt::com_ptr<Interface> &&impl) : m_impl(std::move(impl)) {}

    Interface *Get() const
    {
        return m_impl.get();
    }

    Interface *operator->() const
    {
        return m_impl.get();
    }

    template <typename T = ICgcMlirOperation> auto GetDefiningOp() const
    {
        winrt::com_ptr<T> operation;
        (void)m_impl->GetDefiningOp(IID_PPV_ARGS(&operation));
        return OpWrapper<T>(std::move(operation));
    }

    // Returns the type of this value.
    template <typename T = ICgcMlirType> auto GetType() const
    {
        winrt::com_ptr<T> type;
        THROW_IF_FAILED(m_impl->GetType(IID_PPV_ARGS(&type)));
        return TypeWrapper<T>(std::move(type));
    }

    // Try to get the type of this value. Returns null TypeWrapper if the type doesn't match.
    template <typename T = ICgcMlirType> auto TryGetType() const
    {
        winrt::com_ptr<T> type;
        m_impl->GetType(IID_PPV_ARGS(&type));
        return TypeWrapper<T>(std::move(type));
    }

    // Serializes the wrapped MLIR object to the specified encoding format.
    auto Serialize(CGC_ENCODING encoding) const
    {
        winrt::com_ptr<ICgcBlob> blob;
        THROW_IF_FAILED(m_impl->Serialize(encoding, IID_PPV_ARGS(&blob)));
        return Blob(std::move(blob));
    }

    // It's not possible to directly compare two ICgcMlirValue* for equality since they are lightweight wrappers and
    // don't own the underlying MLIR data. This method provides a way to get a unique handle to the wrapped MLIR value 
    // that can be used for identity comparisons and hashing.
    void *GetHandle() const
    {
        return m_impl->GetHandle();
    }
};

//
// Convenience wrapper for an ICgcMlirOperation-derived interfaces.
//
template <typename Interface = ICgcMlirOperation> struct Op
{
    winrt::com_ptr<Interface> m_impl;

    Op(Interface *impl) { m_impl.copy_from(impl); }
    Op(const winrt::com_ptr<Interface> &impl) : m_impl(impl) {}
    Op(winrt::com_ptr<Interface> &&impl) : m_impl(std::move(impl)) {}

    Interface *operator->() const
    {
        return m_impl.get();
    }

    Interface *Get() const
    {
        return m_impl.get();
    }

    explicit operator bool() const
    {
        return m_impl != nullptr;
    }

    // Cast the operation to the specified interface. Throws if the cast fails.
    template <typename T = ICgcMlirContextChild> OpWrapper<T> As()
    {
        winrt::com_ptr<T> casted;
        THROW_IF_FAILED(m_impl->QueryInterface(IID_PPV_ARGS(&casted)));
        return OpWrapper<T>(std::move(casted));
    }

    // Try to cast the operation to the specified interface. Returns null OpWrapper if the cast fails.
    template <typename T = ICgcMlirContextChild> OpWrapper<T> TryAs()
    {
        winrt::com_ptr<T> casted;
        m_impl->QueryInterface(IID_PPV_ARGS(&casted));
        return OpWrapper<T>(std::move(casted));
    }

    // Returns the parent context of this object.
    auto GetContext() const
    {
        winrt::com_ptr<ICgcMlirContext> context;
        THROW_IF_FAILED(m_impl->GetContext(IID_PPV_ARGS(&context)));
        return Context(std::move(context));
    }

    // Returns the operation's kind or CGC_IR_OPERATION_UNKNOWN if it's not recognized as a more specific kind.
    CGC_IR_OPERATION GetKind() const
    {
        return m_impl->GetKind();
    }

    // Returns the operation's type as its fully qualified MLIR dialect & operation name.
    auto GetName() const
    {
        winrt::com_ptr<ICgcMlirString> str;
        THROW_IF_FAILED(m_impl->GetName(str.put()));
        return String(std::move(str));
    }

    // Returns the operation's symbol name if any. Returns empty string if none.
    auto GetSymbolName() const
    {
        winrt::com_ptr<ICgcMlirString> str;
        THROW_IF_FAILED(m_impl->GetSymbolName(str.put()));
        return String(std::move(str));
    }

    // Returns the tree of operations of this operation as a range of operations.
    template <typename T = ICgcMlirOperation> auto WalkOps(CGC_IR_TRAVERSAL_ORDER order = CGC_IR_TRAVERSAL_ORDER_PRE)
    {
        winrt::com_ptr<ICgcMlirRange> range;
        THROW_IF_FAILED(m_impl->WalkOps(GetOpTypeEnum<T>(), order, IID_PPV_ARGS(&range)));
        return Range<T, OpWrapper<T>>(std::move(range));
    }

    // Returns the direct child operations of this operation as a range of operations.
    template <typename T = ICgcMlirOperation> auto GetOps()
    {
        winrt::com_ptr<ICgcMlirRange> range;
        THROW_IF_FAILED(m_impl->GetOps(GetOpTypeEnum<T>(), IID_PPV_ARGS(&range)));
        return Range<T, OpWrapper<T>>(std::move(range));
    }

    // Returns the operation's operands as a range of values.
    template <typename T = ICgcMlirValue> auto GetOperands()
    {
        winrt::com_ptr<ICgcMlirRange> range;
        THROW_IF_FAILED(m_impl->GetOperands(IID_PPV_ARGS(&range)));
        return Range<T, Value<T>>(std::move(range));
    }

    // Returns the operation's operand types as a range of types.
    template <typename T = ICgcMlirType> auto GetOperandTypes()
    {
        winrt::com_ptr<ICgcMlirRange> range;
        THROW_IF_FAILED(m_impl->GetOperandTypes(IID_PPV_ARGS(&range)));
        return Range<T, TypeWrapper<T>>(std::move(range));
    }

    // Returns the operation's results as a range of values.
    template <typename T = ICgcMlirValue> auto GetResults()
    {
        winrt::com_ptr<ICgcMlirRange> range;
        THROW_IF_FAILED(m_impl->GetResults(IID_PPV_ARGS(&range)));
        return Range<T, Value<T>>(std::move(range));
    }

    // Returns the operation's result types as a range of types.
    template <typename T = ICgcMlirType> auto GetResultTypes()
    {
        winrt::com_ptr<ICgcMlirRange> range;
        THROW_IF_FAILED(m_impl->GetResultTypes(IID_PPV_ARGS(&range)));
        return Range<T, TypeWrapper<T>>(std::move(range));
    }

    template <typename T = ICgcMlirAttribute> auto GetAttribute(std::string_view name) const
    {
        winrt::com_ptr<T> attribute;
        THROW_IF_FAILED(m_impl->GetAttribute(name.data(), IID_PPV_ARGS(&attribute)));
        return AttributeWrapper<T>(std::move(attribute));
    }

    template <typename T = ICgcMlirAttribute> auto TryGetAttribute(std::string_view name) const
    {
        winrt::com_ptr<T> attribute;
        (void)(m_impl->GetAttribute(name.data(), IID_PPV_ARGS(&attribute)));
        return AttributeWrapper<T>(std::move(attribute));
    }

    // Returns the operation's attributes as a range of named attributes.
    template <typename T = ICgcMlirNamedAttribute> auto GetAttributes()
    {
        winrt::com_ptr<ICgcMlirRange> range;
        THROW_IF_FAILED(m_impl->GetAttributes(IID_PPV_ARGS(&range)));
        return Range<T, NamedAttribute<T>>(std::move(range));
    }

    // Serializes the wrapped MLIR object to the specified encoding format.
    auto Serialize(CGC_ENCODING encoding) const
    {
        winrt::com_ptr<ICgcBlob> blob;
        THROW_IF_FAILED(m_impl->Serialize(encoding, IID_PPV_ARGS(&blob)));
        return Blob(std::move(blob));
    }
};

//
// Convenience wrapper for an ICgcMlirModuleOp-derived interfaces.
//
template <typename Interface = ICgcMlirModuleOp> struct Module : public Op<Interface>
{
    Module(Interface *impl) : Op<Interface>(impl) {}
    Module(const winrt::com_ptr<Interface> &impl) : Op<Interface>(impl) {}
    Module(winrt::com_ptr<Interface> &&impl) : Op<Interface>(std::move(impl)) {}

    // Returns GUID defining the semantics of the module (input IR, output IR, partition IR, etc.).
    GUID GetMlirInterface() const
    {
        return this->m_impl->GetMlirInterface();
    }

    // Returns the first program op in the module, if any, or throws.
    auto GetProgram()
    {
        winrt::com_ptr<ICgcMlirRange> range;
        THROW_IF_FAILED(this->m_impl->GetOps(GetOpTypeEnum<ICgcMlirProgramOp>(), IID_PPV_ARGS(&range)));
        THROW_HR_IF_MSG(E_FAIL, range->GetSize() == 0, "No program operation found in module.");

        winrt::com_ptr<ICgcMlirProgramOp> programOp;
        THROW_IF_FAILED(range->Get(0, IID_PPV_ARGS(&programOp)));

        return OpWrapper<ICgcMlirProgramOp>(std::move(programOp));
    }

    // Returns the first entrypoint op in the module, if any, or throws.
    auto GetEntryPoint()
    {
        winrt::com_ptr<ICgcMlirRange> range;
        THROW_IF_FAILED(this->m_impl->GetOps(GetOpTypeEnum<ICgcMlirEntryPointOp>(), IID_PPV_ARGS(&range)));
        THROW_HR_IF_MSG(E_FAIL, range->GetSize() == 0, "No entrypoint operation found in module.");

        winrt::com_ptr<ICgcMlirEntryPointOp> entryPointOp;
        THROW_IF_FAILED(range->Get(0, IID_PPV_ARGS(&entryPointOp)));

        return OpWrapper<ICgcMlirEntryPointOp>(std::move(entryPointOp));
    }
};

//
// Convenience wrapper for an ICgcMlirFunctionOp-derived interfaces.
//
template <typename Interface = ICgcMlirFunctionOp> struct Function : public Op<Interface>
{
    Function(Interface *impl) : Op<Interface>(impl) {}
    Function(const winrt::com_ptr<Interface> &impl) : Op<Interface>(impl) {}
    Function(winrt::com_ptr<Interface> &&impl) : Op<Interface>(std::move(impl)) {}

    // Returns the block arguments of the function.
    template <typename T = ICgcMlirValue> auto GetArguments()
    {
        winrt::com_ptr<ICgcMlirRange> range;
        THROW_IF_FAILED(this->m_impl->GetArguments(IID_PPV_ARGS(&range)));
        return Range<T, Value<T>>(std::move(range));
    }

    // Returns the types of the block arguments of the function.
    template <typename T = ICgcMlirType> auto GetArgumentTypes()
    {
        winrt::com_ptr<ICgcMlirRange> range;
        THROW_IF_FAILED(this->m_impl->GetArgumentTypes(IID_PPV_ARGS(&range)));
        return Range<T, TypeWrapper<T>>(std::move(range));
    }

    // Returns the types of the block return values of the function.
    template <typename T = ICgcMlirType> auto GetReturnTypes()
    {
        winrt::com_ptr<ICgcMlirRange> range;
        THROW_IF_FAILED(this->m_impl->GetReturnTypes(IID_PPV_ARGS(&range)));
        return Range<T, TypeWrapper<T>>(std::move(range));
    }
};

//
// Convenience wrapper for an ICgcMlirEntryPointOp-derived interfaces.
//
template <typename Interface = ICgcMlirEntryPointOp> struct EntryPoint : public Function<Interface>
{
    EntryPoint(Interface *impl) : Function<Interface>(impl) {}
    EntryPoint(const winrt::com_ptr<Interface> &impl) : Function<Interface>(impl) {}
    EntryPoint(winrt::com_ptr<Interface> &&impl) : Function<Interface>(std::move(impl)) {}
};

//
// Convenience wrapper for an ICgcMlirProgramOp-derived interfaces.
//
template <typename Interface = ICgcMlirProgramOp> struct Program : public Function<Interface>
{
    Program(Interface *impl) : Function<Interface>(impl) {}
    Program(const winrt::com_ptr<Interface> &impl) : Function<Interface>(impl) {}
    Program(winrt::com_ptr<Interface> &&impl) : Function<Interface>(std::move(impl)) {}
};

//
// Convenience wrapper for an ICgcMlirInvokeOp-derived interfaces.
//
template <typename Interface = ICgcMlirInvokeOp> struct Invoke : public Op<Interface>
{
    Invoke(Interface *impl) : Op<Interface>(impl) {}
    Invoke(const winrt::com_ptr<Interface> &impl) : Op<Interface>(impl) {}
    Invoke(winrt::com_ptr<Interface> &&impl) : Op<Interface>(std::move(impl)) {}

    // Returns the operation called by this invoke op.
    template <typename T = ICgcMlirOperation> auto GetCallee()
    {
        winrt::com_ptr<T> callee;
        THROW_IF_FAILED(this->m_impl->GetCallee(IID_PPV_ARGS(&callee)));
        return OpWrapper<T>(std::move(callee));
    }
};

//
// Convenience wrapper for an ICgcMlirPartitionOp-derived interfaces.
//
template <typename Interface = ICgcMlirPartitionOp> struct Partition : public Function<Interface>
{
    Partition(Interface *impl) : Function<Interface>(impl) {}
    Partition(const winrt::com_ptr<Interface> &impl) : Function<Interface>(impl) {}
    Partition(winrt::com_ptr<Interface> &&impl) : Function<Interface>(std::move(impl)) {}

    // Clones the partition and its dependencies (subgraphs) into a new self-contained module.
    auto Isolate()
    {
        winrt::com_ptr<ICgcMlirModuleOp> module;
        THROW_IF_FAILED(this->m_impl->Isolate(IID_PPV_ARGS(&module)));
        return OpWrapper<ICgcMlirModuleOp>(std::move(module));
    }
};

//
// Convenience wrapper for an ICgcMlirSubgraphOp-derived interfaces.
//
template <typename Interface = ICgcMlirSubgraphOp> struct Subgraph : public Op<Interface>
{
    Subgraph(Interface *impl) : Op<Interface>(impl) {}
    Subgraph(const winrt::com_ptr<Interface> &impl) : Op<Interface>(impl) {}
    Subgraph(winrt::com_ptr<Interface> &&impl) : Op<Interface>(std::move(impl)) {}
};

//
// Convenience wrapper for an ICgcMlirFunctionalDefinitionOp-derived interfaces.
//
template <typename Interface = ICgcMlirFunctionalDefinitionOp> struct FunctionalDefinition : public Function<Interface>
{
    FunctionalDefinition(Interface *impl) : Function<Interface>(impl) {}
    FunctionalDefinition(const winrt::com_ptr<Interface> &impl) : Function<Interface>(impl) {}
    FunctionalDefinition(winrt::com_ptr<Interface> &&impl) : Function<Interface>(std::move(impl)) {}
};

//
// Convenience wrapper for an ICgcMlirConstantOp-derived interfaces.
//
template <typename Interface = ICgcMlirConstantOp> struct Constant : public Op<Interface>
{
    Constant(Interface *impl) : Op<Interface>(impl) {}
    Constant(const winrt::com_ptr<Interface> &impl) : Op<Interface>(impl) {}
    Constant(winrt::com_ptr<Interface> &&impl) : Op<Interface>(std::move(impl)) {}

    // Returns the constant data.
    auto GetData()
    {
        winrt::com_ptr<ICgcMlirSpan> span;
        THROW_IF_FAILED(this->m_impl->GetData(IID_PPV_ARGS(&span)));
        return Span(std::move(span));
    }

    // Returns the constant resource's human-readable key, or std::nullopt if not available (e.g., inline constants).
    std::optional<String<>> TryGetDataKey()
    {
        winrt::com_ptr<ICgcMlirString> str;
        if (SUCCEEDED(this->m_impl->GetDataKey(str.put())))
        {
            return String(std::move(str));
        }
        return std::nullopt;
    }

    // Returns the constant result value.
    auto GetValue()
    {
        winrt::com_ptr<ICgcMlirValue> value;
        THROW_IF_FAILED(this->m_impl->GetValue(IID_PPV_ARGS(&value)));
        return Value(std::move(value));
    }
};

//
// Convenience wrapper for an ICgcMlirViewMemrefOp-derived interfaces.
//
template <typename Interface = ICgcMlirCopyOp> struct Copy : public Op<Interface>
{
    Copy(Interface *impl) : Op<Interface>(impl) {}
    Copy(const winrt::com_ptr<Interface> &impl) : Op<Interface>(impl) {}
    Copy(winrt::com_ptr<Interface> &&impl) : Op<Interface>(std::move(impl)) {}

    // Returns the source value being copied from.
    auto GetSource()
    {
        winrt::com_ptr<ICgcMlirValue> source;
        THROW_IF_FAILED(this->m_impl->GetSource(IID_PPV_ARGS(&source)));
        return Value(std::move(source));
    }

    // Returns the target value being copied to.
    auto GetTarget()
    {
        winrt::com_ptr<ICgcMlirValue> target;
        THROW_IF_FAILED(this->m_impl->GetTarget(IID_PPV_ARGS(&target)));
        return Value(std::move(target));
    }
};

template <typename Interface = ICgcMlirViewMemrefOp> struct ViewMemref : public Op<Interface>
{
    ViewMemref(Interface *impl) : Op<Interface>(impl) {}
    ViewMemref(const winrt::com_ptr<Interface> &impl) : Op<Interface>(impl) {}
    ViewMemref(winrt::com_ptr<Interface> &&impl) : Op<Interface>(std::move(impl)) {}

    // Returns the value viewed by this operation.
    auto GetSource()
    {
        winrt::com_ptr<ICgcMlirValue> source;
        THROW_IF_FAILED(this->m_impl->GetSource(IID_PPV_ARGS(&source)));
        return Value(std::move(source));
    }

    // Returns the view offset in bytes.
    uint64_t GetOffset()
    {
        uint64_t offset = 0;
        THROW_IF_FAILED(this->m_impl->GetOffset(&offset));
        return offset;
    }
};

} // namespace cgc_ir_templates