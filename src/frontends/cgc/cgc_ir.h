// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

// --------------------------------------------------------------------------------------------------------------------
// CGC IR C++ Helper API
// 
// THIS API NOT GUARANTEED TO BE STABLE AND MAY HAVE BREAKING CHANGES.
//
// This API provides CGC IR inspection in a limited capacity without depending directly on MLIR. This may be useful 
// for tools or components that need a subset of MLIR functionality. This API does not aim to cover all use-cases such 
// as generating IR, mutating IR, or expressing all aspects of the IR.
//
// Design notes:
//
// - This header contains ABI-stable interfaces that generally mirror/wrap MLIR C++ types. Using COM-style interfaces
//   directly can sometimes be verbose, so helper templates are provided in CGC_ir_util.h.
//
// - Most interfaces inherit from ICgcMlirContextChild to provide access to their parent context, which manages the
//   lifetime of the underlying MLIR data. Nano-COM objects are lightweight references to data owned by the context.
//   Users do not need to keep an explicit reference to the context as long as they hold references to objects created 
//   from it.
//
// - Parameters for return values are typically expressed as "REFIID riid, _COM_Outptr_ void** ppv" instead of concrete 
//   interface types to prioritize extensibility and flexibility over static type safety. For example, ICgcMlirRange::Get 
//   can directly instantiate an IUnknown, ICgcMlirOperation, ICgcMlirType, etc. instead of having type-specific 
//   interfaces like ICgcMlirOperationRange, ICgcMlirTypeRange, etc. In cases where a single return type seems obvious,
//   the more generic pattern is used for consistency and to avoid having to introduce new interfaces later.
//
// - When interfaces need to return variable-sized contiguous data it is often returned as a span instead of requiring 
//   a two-step calling pattern and copying into caller-allocated buffers. For example, strings and arrays are usually
//   returned as spans.
//
// --------------------------------------------------------------------------------------------------------------------

#include "cgc.h"

// Creates a ICgcMlirContext.
// Expected output interface(s): ICgcMlirModuleOp
STDAPI CgcCreateMlirContext(REFIID riid, _COM_Outptr_ void** ppv);

enum CGC_IR_TRAVERSAL_ORDER
{
    CGC_IR_TRAVERSAL_ORDER_PRE,
    CGC_IR_TRAVERSAL_ORDER_POST,
};

struct CGC_IR_EDGE
{
    UINT32 SourceNodeIndex;
    UINT32 TargetNodeIndex;
};

enum CGC_IR_TYPE;

// Encapsulates an MLIR context that can process CGC dialects.
interface CGC_DECLARE_INTERFACE("88002289-5c2e-4698-adec-bc3d09e55a14") ICgcMlirContext : public IUnknown
{
    // Deserializes a IR from the provided MLIR data.
    // Expected output interface(s): ICgcMlirModuleOp
    IFACEMETHOD(Deserialize)(
        const void* data,
        UINT64 dataSizeInBytes,
        REFIID riid,
        _COM_Outptr_ void** ppv
    ) = 0;

    // Creates a subgraph configuration builder.
    // Expected output interface(s): ICgcMlirSubgraphConfigBuilder
    IFACEMETHOD(CreateSubgraphConfigBuilder)(
        REFIID riid,
        _COM_Outptr_ void** ppv
    ) = 0;
};

// Most interfaces in this API are lightweight references to memory that is owned by the context, which mirrors the 
// structure of the wrapped MLIR C++ types (mlir::Operation*, mlir::Type, etc.). Objects created from a context inherit
// this interface to provide access to their parent context; furthermore, these objects keep a reference on the context
// which frees the client from managing the context's lifetime explicitly.
interface CGC_DECLARE_INTERFACE("9a47dae8-76b3-4b48-8241-88ad360f0dcb") ICgcMlirContextChild : public IUnknown
{
    // Returns the parent context of this object.
    // Expected output interface(s): ICgcMlirModuleOp
    IFACEMETHOD(GetContext)(REFIID riid, _COM_Outptr_ void** ppv) = 0;
};

// Helper for external targets to build an MLIR subgraph specialization response.
interface CGC_DECLARE_INTERFACE("cab22b4c-7ce5-434b-9013-45ea91ee9f9e") ICgcMlirSubgraphConfigBuilder : public ICgcMlirContextChild
{
    IFACEMETHOD_(void, SetPriorityLevel)(INT32 priorityLevel) = 0;

    IFACEMETHOD(SetScratchMemoryRequirements)(
        _In_reads_(segmentCount) const INT64* sizesInBytes,
        _In_reads_(segmentCount) const INT64* alignments,
        SIZE_T segmentCount
    ) = 0;

    IFACEMETHOD_(void, SetForeignConfigEntry)(
        const char* key,
        const char* value
    ) = 0;

    // Builds the supported configuration and places it as the sole entry in a 
    // cgc_subgraph_specialization.specialization_transform_results op in a new module.
    // Expected output interface(s): ICgcMlirModuleOp
    IFACEMETHOD(BuildSpecializationResults)(
        const char* symbolName,
        REFIID riid, 
        _COM_Outptr_ void** ppv
    ) = 0;
};

// --------------------------------------------------------------------------------------------------------------------
// Views and Iterators
// --------------------------------------------------------------------------------------------------------------------

// Encapsulates a range of objects (e.g., ops, types, values) whose storage is tied to the MLIR context.
interface CGC_DECLARE_INTERFACE("bb4aaa2a-d765-41b1-8fa7-a6c776596718") ICgcMlirRange : public ICgcMlirContextChild
{
    // Returns the number of elements in the range. An empty range (size 0) is valid.
    IFACEMETHOD_(SIZE_T, GetSize)() = 0;

    // Retrieves the element at the specified index in the range.
    // Expected output interface(s): ICgcMlirOperation, ICgcMlirType, ICgcMlirValue, or derivatives.
    IFACEMETHOD(Get)(UINT32 index, REFIID riid, _COM_Outptr_ void** element) = 0;
};

// Encapsulates a span of primitive data (e.g., chars, integers, bytes) whose storage is tied to the MLIR context.
interface CGC_DECLARE_INTERFACE("69c1c7ba-5b62-4819-8e90-fa3dfef56814") ICgcMlirSpan : public ICgcMlirContextChild
{
    // Returns a pointer to the spans's data. It is only safe to access this pointer while the span is alive, so it
    // should not be copied or cached for later use.
    IFACEMETHOD_(const void*, GetData)() = 0;

    // Returns the data type of the data elements.
    IFACEMETHOD_(CGC_IR_TYPE, GetDataType)() = 0;

    // Returns the size in elements.
    IFACEMETHOD_(SIZE_T, GetDataSize)() = 0;

    // Returns the size in bytes.
    IFACEMETHOD_(SIZE_T, GetDataSizeInBytes)() = 0;
};

// Encapsulates a graph (dataflow, dependency, etc.). Edge semantics are determined by the origin of the graph.
interface CGC_DECLARE_INTERFACE("60aa47c1-0864-4073-851f-380805d59f15") ICgcMlirGraph : public ICgcMlirContextChild
{
    // Returns the number of nodes in the graph.
    IFACEMETHOD_(SIZE_T, GetNodeCount)() = 0;

    // Retrieves the node at the specified index.
    // Expected output interface(s): ICgcMlirOperation or derivatives for operation graphs.
    IFACEMETHOD(GetNode)(UINT32 nodeIndex, REFIID riid, _COM_Outptr_ void** node) = 0;

    // Returns the number of edges in the graph.
    IFACEMETHOD_(SIZE_T, GetEdgeCount)() = 0;

    // Retrieves the edge at the specified graph edge index.
    IFACEMETHOD(GetEdge)(UINT32 edgeIndex, _Out_ CGC_IR_EDGE* edge) = 0;
};

// Encapsulates a UTF-8 string whose storage is tied to the MLIR context.
interface CGC_DECLARE_INTERFACE("cbb71059-9c72-4333-a2f7-b61e68ecb9a8") ICgcMlirString : public ICgcMlirContextChild
{
    // Returns the string data as a null-terminated UTF-8 string.
    IFACEMETHOD_(const char*, GetData)() = 0;

    // Returns the size of the string in characters (not including null terminator).
    IFACEMETHOD_(SIZE_T, GetDataSize)() = 0;
};

// --------------------------------------------------------------------------------------------------------------------
// Types
// --------------------------------------------------------------------------------------------------------------------

enum CGC_IR_TYPE
{
    CGC_IR_TYPE_UNKNOWN,
    CGC_IR_TYPE_BOOL8,
    CGC_IR_TYPE_INT2,
    CGC_IR_TYPE_INT4,
    CGC_IR_TYPE_INT8,
    CGC_IR_TYPE_INT16,
    CGC_IR_TYPE_INT32,
    CGC_IR_TYPE_INT64,
    CGC_IR_TYPE_UINT2,
    CGC_IR_TYPE_UINT4,
    CGC_IR_TYPE_UINT8,
    CGC_IR_TYPE_UINT16,
    CGC_IR_TYPE_UINT32,
    CGC_IR_TYPE_UINT64,
    CGC_IR_TYPE_FLOAT64,
    CGC_IR_TYPE_FLOAT32,
    CGC_IR_TYPE_FLOAT16,
    CGC_IR_TYPE_BFLOAT16,
    CGC_IR_TYPE_FLOAT8E5M2,
    CGC_IR_TYPE_FLOAT8E4M3,
    CGC_IR_TYPE_FLOAT8E4M3FN,
    CGC_IR_TYPE_FLOAT8E5M2FNUZ,
    CGC_IR_TYPE_FLOAT8E4M3FNUZ,
    CGC_IR_TYPE_FLOAT8E4M3B11FNUZ,
    CGC_IR_TYPE_FLOAT8E3M4,
    CGC_IR_TYPE_FLOAT4E2M1FN,
    CGC_IR_TYPE_FLOAT6E2M3FN,
    CGC_IR_TYPE_FLOAT6E3M2FN,
    CGC_IR_TYPE_FLOAT8E8M0FNU,
};

// Describes one entry in the physical order permutation.
//
// Every entry represents a real dimension with fully resolved MSB/LSB.
// Grouped entries share the same GroupIndex (0, 1, 2, ...).
// Non-grouped entries have GroupIndex = UINT32_MAX.
struct CGC_IR_PHYSICAL_ORDER_ENTRY
{
    UINT32 DimensionIndex; // logical dimension index
    UINT32 GroupIndex;     // which group this entry belongs to, or UINT32_MAX if not grouped
    UINT32 MsbIndex;       // resolved MSB bit position of the dimension's bit range
    UINT32 LsbIndex;       // resolved LSB bit position of the dimension's bit range
};

interface CGC_DECLARE_INTERFACE("1e2a4cb4-f485-4e36-afe7-11eb49db9daf") ICgcMlirType : public ICgcMlirContextChild
{
    // Returns the type's kind or CGC_IR_TYPE_UNKNOWN if it's not recognized as a more specific type.
    IFACEMETHOD_(CGC_IR_TYPE, GetKind)() = 0;

    // Serializes the wrapped MLIR object to the specified encoding format.
    IFACEMETHOD(Serialize)(CGC_ENCODING encoding, REFIID riid, _COM_Outptr_ void** ppv) = 0;
};

interface CGC_DECLARE_INTERFACE("ec9abfdc-6d38-44a4-8296-fd50641ef726") ICgcMlirMultiDimType : public ICgcMlirType
{
    // Returns the rank (number of dimensions).
    IFACEMETHOD_(SIZE_T, GetRank)() = 0;

    // Returns the total size in bytes.
    IFACEMETHOD_(SIZE_T, GetSizeInBytes)() = 0;

    // Returns the total number of elements (product of dimensions).
    IFACEMETHOD_(SIZE_T, GetSizeInElements)() = 0;

    // Returns the data type of the elements.
    IFACEMETHOD_(CGC_IR_TYPE, GetDataType)() = 0;

    // Returns the data type of dimension sizes.
    IFACEMETHOD_(CGC_IR_TYPE, GetIndexType)() = 0;

    // Returns the size of all dimensions, which is typically a span of INT64 values.
    // Expected output interface(s): ICgcMlirSpan
    IFACEMETHOD(GetShape)(REFIID riid, _COM_Outptr_ void** shape) = 0;

    // Returns whether the type is marked as constant (immutable with compile-time known contents).
    IFACEMETHOD_(BOOL, IsConstant)() = 0;
};

interface CGC_DECLARE_INTERFACE("10fc3cd4-7023-4dd6-a602-a5606c373cf2") ICgcMlirTensorType : public ICgcMlirMultiDimType
{
};

interface CGC_DECLARE_INTERFACE("1f192cef-ac8d-46a1-bfa2-9062d7297f69") ICgcMlirMemrefType : public ICgcMlirMultiDimType
{
    // Returns the memory layout attribute of this memref type, if any.
    // Expected output interface(s): ICgcMlirMemoryLayoutAttribute
    IFACEMETHOD(GetLayout)(REFIID riid, _COM_Outptr_ void** memoryLayoutAttribute) = 0;

    // Resolves the physical shape for this memref type by applying the layout (permutation, padding,
    // byte alignment) to the logical shape. If no layout is present, the physical shape equals the
    // logical shape. Returns HRESULT_FROM_WIN32(ERROR_INVALID_DATA) for incompatible layouts or
    // if the physical shape cannot be computed.
    // Expected output interface(s): ICgcMlirPhysicalLayout
    IFACEMETHOD(ResolvePhysicalShape)(REFIID riid, _COM_Outptr_ void** physicalShape) = 0;
};

// Encapsulates the resolved physical layout of a memref type.
// Provides the physical shape, rank, and size after applying layout transformations.
interface CGC_DECLARE_INTERFACE("b4c8d2e6-f1a3-4d5b-8e7f-9a0b1c2d3e4f") ICgcMlirPhysicalLayout : public ICgcMlirContextChild
{
    // Returns the rank (number of dimensions) of the physical shape.
    IFACEMETHOD_(SIZE_T, GetRank)() = 0;

    // Returns the physical shape as a span of INT64 values.
    // Expected output interface(s): ICgcMlirSpan
    IFACEMETHOD(GetShape)(REFIID riid, _COM_Outptr_ void** shape) = 0;

    // Returns the total number of elements in the physical shape (product of physical dimensions).
    IFACEMETHOD_(SIZE_T, GetSizeInElements)() = 0;

    // Returns the total size in bytes of the physical shape.
    IFACEMETHOD_(SIZE_T, GetSizeInBytes)() = 0;
};

// --------------------------------------------------------------------------------------------------------------------
// Attributes
//
//  ICgcMlirAttribute
//  ├── ICgcMlirTypedAttribute
//  │   ├── ICgcMlirScalarAttribute
//  │   └── ICgcMlirMultiDimAttribute
//  ├── ICgcMlirStringAttribute
//  └── ICgcMlirMemoryLayoutAttribute
//
// --------------------------------------------------------------------------------------------------------------------

// Categorizes common MLIR attribute types for type-safe queries.
enum CGC_IR_ATTRIBUTE
{
    CGC_IR_ATTRIBUTE_UNKNOWN,  // generic operation that isn't mapped to a more specific type
    CGC_IR_ATTRIBUTE_ENUM,     // various enum attributes
    CGC_IR_ATTRIBUTE_STRING,   // CgcStringAttr
    CGC_IR_ATTRIBUTE_SCALAR,   // CgcIntegerAttr, CgcFloatAttr, CgcBoolAttr
    CGC_IR_ATTRIBUTE_MULTIDIM, // CgcDenseIntegerElementsAttr
    CGC_IR_ATTRIBUTE_MEMORY_LAYOUT, // cgc_memory::LayoutAttr
};

// Reference a string name & MLIR attribute pair.
interface CGC_DECLARE_INTERFACE("ed5c3a06-a5fc-48b0-9cdb-105cf8abde24") ICgcMlirNamedAttribute : public ICgcMlirContextChild
{
    // Returns the name of the attribute.
    IFACEMETHOD(GetName)(_COM_Outptr_ ICgcMlirString** ppv) = 0;

    // Expected output interface(s): ICgcMlirAttribute or derivatives.
    IFACEMETHOD(GetAttribute)(REFIID riid, _COM_Outptr_ void** ppv) = 0;
};

// References a generic MLIR attribute.
interface CGC_DECLARE_INTERFACE("6cfce9bd-c709-4962-a134-3744288f60e6") ICgcMlirAttribute : public ICgcMlirContextChild
{
    // Returns the attribute's kind or CGC_IR_ATTRIBUTE_UNKNOWN if it's not recognized as a more specific kind.
    IFACEMETHOD_(CGC_IR_ATTRIBUTE, GetKind)() = 0;

    // Returns the attribute's type as its fully qualified MLIR dialect & attribute name.
    IFACEMETHOD(GetAttributeTypeName)(_COM_Outptr_ ICgcMlirString** ppv) = 0;

    // Serializes the wrapped MLIR attribute to the specified encoding format.
    // Expected output interface(s): ICgcBlob
    IFACEMETHOD(Serialize)(CGC_ENCODING encoding, REFIID riid, _COM_Outptr_ void** ppv) = 0;
};

// Encapsulates a memory layout attribute (cgc_memory.layout).
// Exposes physical order, padding, and byte alignment sub-components.
interface CGC_DECLARE_INTERFACE("f06cf5e0-aa14-46e2-a26b-f544ea7daa21") ICgcMlirMemoryLayoutAttribute
    : public ICgcMlirAttribute
{
    // Returns TRUE if the layout is an identity layout (no transformation).
    IFACEMETHOD_(BOOL, IsIdentity)() = 0;

    // Returns the per-dimension element alignments as a span of INT64 values, if available.
    // Returns HRESULT_FROM_WIN32(ERROR_NOT_FOUND) if the layout has no padding.
    // Expected output interface(s): ICgcMlirSpan
    IFACEMETHOD(TryGetElementAlignments)(REFIID riid, _COM_Outptr_ void** elementAlignments) = 0;

    // Returns the per-dimension byte alignments as a span of INT64 values, if available.
    // Returns HRESULT_FROM_WIN32(ERROR_NOT_FOUND) if the layout has no byte alignment.
    // Expected output interface(s): ICgcMlirSpan
    IFACEMETHOD(TryGetByteAlignments)(REFIID riid, _COM_Outptr_ void** byteAlignments) = 0;

    // Returns the padding fill value as a scalar attribute, if available.
    // Returns HRESULT_FROM_WIN32(ERROR_NOT_FOUND) if padding has no constant fill value.
    // Expected output interface(s): ICgcMlirScalarAttribute
    IFACEMETHOD(TryGetFillValue)(REFIID riid, _COM_Outptr_ void** fillValue) = 0;

    // Returns the number of physical order entries. Returns 0 if the layout has no physical
    // order or no logical shape is available.
    IFACEMETHOD_(SIZE_T, GetPhysicalOrderEntryCount)() = 0;

    // Writes the physical order as a flat array of CGC_IR_PHYSICAL_ORDER_ENTRY structs into
    // a caller-allocated buffer. Every entry is a real dimension with resolved bit ranges.
    // Entries with the same GroupIndex (!= UINT32_MAX) belong to the same group.
    // The logical shape (needed to resolve open-ended MSB/LSB defaults) is captured internally
    // when the layout attribute is obtained from a memref type via GetLayout().
    // Returns HRESULT_FROM_WIN32(ERROR_NOT_FOUND) if the layout has no physical order or no
    // logical shape is available.
    IFACEMETHOD(GetPhysicalOrderEntries)(_Out_writes_(entryCount) CGC_IR_PHYSICAL_ORDER_ENTRY* entries,
                                         SIZE_T entryCount) = 0;
};

// Encapsulates an attribute associated with an enum.
interface CGC_DECLARE_INTERFACE("f46c5ea4-e7e2-43f9-8e2e-6dbb23bc282e") ICgcMlirEnumAttribute : public ICgcMlirAttribute
{
    // Returns a pointer to the string data. The pointer is valid for the lifetime of the attribute.
    IFACEMETHOD_(const char*, GetStringValue)() = 0;

    // Returns the size of the string in characters (not including null terminator).
    IFACEMETHOD_(SIZE_T, GetStringValueSize)() = 0;
};

// Encapsulates an attribute associated with a string.
interface CGC_DECLARE_INTERFACE("4621cedc-8b87-4ef8-b448-5288ec91610a") ICgcMlirStringAttribute : public ICgcMlirAttribute
{
    // Returns a pointer to the string data. The pointer is valid for the lifetime of the attribute.
    IFACEMETHOD_(const char*, GetValue)() = 0;

    // Returns the size of the string in characters (not including null terminator).
    IFACEMETHOD_(SIZE_T, GetValueSize)() = 0;
};

// References an attribute that is associated with a specific MLIR type.
interface CGC_DECLARE_INTERFACE("ffd72eb8-91bd-4d19-bbdf-15e94d0740eb") ICgcMlirTypedAttribute : public ICgcMlirAttribute
{
    // Returns the MLIR type of the attribute.
    // Expected output interface(s): ICgcMlirType or derivatives.
    IFACEMETHOD(GetType)(REFIID riid, _COM_Outptr_ void** type) = 0;
};

// References an attribute that holds a scalar value.
interface CGC_DECLARE_INTERFACE("0ca20592-8b1d-44f3-bbf0-25167cef5f22") ICgcMlirScalarAttribute : public ICgcMlirTypedAttribute
{
    IFACEMETHOD(GetBoolValue)(_Out_ BOOL* value) = 0;
    IFACEMETHOD(GetInt64Value)(_Out_ INT64* value) = 0;
    IFACEMETHOD(GetInt32Value)(_Out_ INT32* value) = 0;
    IFACEMETHOD(GetInt16Value)(_Out_ INT16* value) = 0;
    IFACEMETHOD(GetInt8Value)(_Out_ INT8* value) = 0;
    IFACEMETHOD(GetUint64Value)(_Out_ UINT64* value) = 0;
    IFACEMETHOD(GetUint32Value)(_Out_ UINT32* value) = 0;
    IFACEMETHOD(GetUint16Value)(_Out_ UINT16* value) = 0;
    IFACEMETHOD(GetUint8Value)(_Out_ UINT8* value) = 0;
    IFACEMETHOD(GetFloat64Value)(_Out_ double* value) = 0;
    IFACEMETHOD(GetFloat32Value)(_Out_ float* value) = 0;
};

// References an attribute that holds a multi-dimensional array of values.
interface CGC_DECLARE_INTERFACE("3cc657de-eaa3-46eb-a0e8-e78043a2d17a") ICgcMlirMultiDimAttribute : public ICgcMlirTypedAttribute
{
    // Returns a pointer to the data. The pointer is valid for the lifetime of the attribute.
    IFACEMETHOD_(const void*, GetValue)() = 0;

    // Returns the size in elements.
    IFACEMETHOD_(SIZE_T, GetValueSize)() = 0;
};

// --------------------------------------------------------------------------------------------------------------------
// Values
// --------------------------------------------------------------------------------------------------------------------

// Encapsulates a CGC MLIR SSA value.
interface CGC_DECLARE_INTERFACE("8b1fba8d-901b-46a3-b1e9-b2c916ed95cd") ICgcMlirValue : public ICgcMlirContextChild
{
    // Returns the operation that defines this value (producer), or nullptr for block arguments.
    // Expected output interface(s): ICgcMlirOperation
    IFACEMETHOD(GetDefiningOp)(REFIID riid, _COM_Outptr_opt_ void** operation) = 0;

    // Returns the type of this value.
    // Expected output interface(s): ICgcMlirType
    IFACEMETHOD(GetType)(REFIID riid, _COM_Outptr_ void** type) = 0;

    // Serializes the wrapped MLIR object to the specified encoding format.
    // Expected output interface(s): ICgcBlob
    IFACEMETHOD(Serialize)(CGC_ENCODING encoding, REFIID riid, _COM_Outptr_ void** ppv) = 0;

    // It's not possible to directly compare two ICgcMlirValue* for equality since they are lightweight wrappers and
    // don't own the underlying MLIR data. This method provides a way to get a unique handle to the wrapped MLIR value 
    // that can be used for identity comparisons and hashing.
    IFACEMETHOD_(void*, GetHandle)() = 0;
};

// --------------------------------------------------------------------------------------------------------------------
// Operations
// --------------------------------------------------------------------------------------------------------------------

enum CGC_IR_OPERATION
{
    CGC_IR_OPERATION_UNKNOWN,               // generic operation that isn't mapped to a more specific type
    CGC_IR_OPERATION_MODULE,                // cgc.module
    CGC_IR_OPERATION_ENTRY_POINT,           // cgc.entry_point
    CGC_IR_OPERATION_PROGRAM,               // cgc.program
    CGC_IR_OPERATION_INITIALIZER,           // cgc.initializer
    CGC_IR_OPERATION_INVOKE,                // cgc.invoke
    CGC_IR_OPERATION_PARTITION,             // cgc_partition.partition
    CGC_IR_OPERATION_SUBGRAPH,              // cgc_subgraph.subgraph
    CGC_IR_OPERATION_CONSTANT,              // cgc.constant
    CGC_IR_OPERATION_VIEW_MEMREF,           // cgc_memory.view_memref
    CGC_IR_OPERATION_FUNCTIONAL_DEFINITION, // cgc_subgraph.functional_definition
    CGC_IR_OPERATION_COPY,                  // cgc_memory.copy

    CGC_IR_OPERATION_ANY = CGC_IR_OPERATION_UNKNOWN, // match all operation types
};

// Lightweight reference to an MLIR operation whose storage is managed in the context.
interface CGC_DECLARE_INTERFACE("634bd30e-ce44-49d8-98ad-5f255671ddc1") ICgcMlirOperation : public ICgcMlirContextChild
{
    // Returns the operation's kind or CGC_IR_OPERATION_UNKNOWN if it's not recognized as a more specific kind.
    IFACEMETHOD_(CGC_IR_OPERATION, GetKind)() = 0;

    // Returns the operation's type as its fully qualified MLIR dialect & operation name.
    IFACEMETHOD(GetName)(_COM_Outptr_ ICgcMlirString** ppv) = 0;

    // Returns the operation's symbol name if any. Returns empty string if none.
    IFACEMETHOD(GetSymbolName)(_COM_Outptr_ ICgcMlirString** ppv) = 0;

    // Returns the tree of operations of this operation as a range of operations.
    // The 'type' parameter limits the returned ops to the specified type; CGC_IR_OPERATION_ANY includes all types.
    // The 'order' parameter specifies the traversal order.
    // Expected output interface(s): ICgcMlirRange
    IFACEMETHOD(WalkOps)(CGC_IR_OPERATION type, CGC_IR_TRAVERSAL_ORDER order, REFIID riid, _COM_Outptr_ void** ppv) = 0;

    // Returns the direct child operations of this operation as a range of operations.
    // The 'type' parameter limits the returned ops to the specified type; CGC_IR_OPERATION_ANY includes all types.
    // Expected output interface(s): ICgcMlirRange
    IFACEMETHOD(GetOps)(CGC_IR_OPERATION type, REFIID riid, _COM_Outptr_ void** ppv) = 0;

    // Returns the operation's operands as a range of values.
    // Expected output interface(s): ICgcMlirRange
    IFACEMETHOD(GetOperands)(REFIID riid, _COM_Outptr_ void** ppv) = 0;

    // Returns the operation's operand types as a range of types.
    // Expected output interface(s): ICgcMlirRange
    IFACEMETHOD(GetOperandTypes)(REFIID riid, _COM_Outptr_ void** ppv) = 0;

    // Returns the operation's results as a range of values.
    // Expected output interface(s): ICgcMlirRange
    IFACEMETHOD(GetResults)(REFIID riid, _COM_Outptr_ void** ppv) = 0;

    // Returns the operation's result types as a range of types.
    // Expected output interface(s): ICgcMlirRange
    IFACEMETHOD(GetResultTypes)(REFIID riid, _COM_Outptr_ void** ppv) = 0;

    // Returns all attributes of this operation as a range of attributes.
    // Expected output interface(s): ICgcMlirRange (of ICgcMlirAttribute)
    IFACEMETHOD(GetAttributes)(REFIID riid, _COM_Outptr_ void** ppv) = 0;

    // Gets a specific attribute by name.
    // Expected output interface(s): ICgcMlirAttribute
    IFACEMETHOD(GetAttribute)(_In_z_ LPCSTR name, REFIID riid, _COM_Outptr_ void** ppv) = 0;

    // Serializes the wrapped MLIR object to the specified encoding format.
    // Expected output interface(s): ICgcBlob
    IFACEMETHOD(Serialize)(CGC_ENCODING encoding, REFIID riid, _COM_Outptr_ void** ppv) = 0;
};

// Encapsulates a cgc.module operation.
interface CGC_DECLARE_INTERFACE("15232b7b-b6d8-4d79-a923-782aadbbd440") ICgcMlirModuleOp : public ICgcMlirOperation
{
    // Returns GUID defining the semantics of the module (input IR, output IR, partition IR, etc.).
    IFACEMETHOD_(GUID, GetMlirInterface)() = 0;
};

// Encapsulates an operation that implement the CGC FunctionOpInterface.
interface CGC_DECLARE_INTERFACE("63e8ad6b-27f1-4739-8422-5ccb26297619") ICgcMlirFunctionOp : public ICgcMlirOperation
{
    // Returns the block arguments of the function.
    // Expected output interface(s): ICgcMlirRange
    IFACEMETHOD(GetArguments)(REFIID riid, _COM_Outptr_ void** ppv) = 0;

    // Returns the types of the block arguments of the function.
    // Expected output interface(s): ICgcMlirRange
    IFACEMETHOD(GetArgumentTypes)(REFIID riid, _COM_Outptr_ void** ppv) = 0;

    // Returns the types of the block return values of the function.
    // Expected output interface(s): ICgcMlirRange
    IFACEMETHOD(GetReturnTypes)(REFIID riid, _COM_Outptr_ void** ppv) = 0;
};

// Encapsulates a cgc.entry_point operation.
interface CGC_DECLARE_INTERFACE("d1f795b3-a2e8-4454-9891-aaf2ac1becfd") ICgcMlirEntryPointOp : public ICgcMlirFunctionOp
{
    // Returns the dataflow graph of CGC operations in the program body.
    // Expected output interface(s): ICgcMlirGraph
    IFACEMETHOD(GetDataflowGraph)(REFIID riid, _COM_Outptr_ void** ppv) = 0;
};

// Encapsulates a cgc.program operation.
interface CGC_DECLARE_INTERFACE("50b94556-586d-4c2a-83ad-ed9a2dead2bd") ICgcMlirProgramOp : public ICgcMlirFunctionOp
{
    // Returns the dependency graph of executable operations (i.e., cgc.invoke) in the program's entry block.
    // Expected output interface(s): ICgcMlirGraph
    IFACEMETHOD(GetDependencyGraph)(REFIID riid, _COM_Outptr_ void** ppv) = 0;
};

// Encapsulates a cgc.initializer operation.
interface CGC_DECLARE_INTERFACE("e7a3f245-8b1c-4d2e-9f6a-3c5d8e2a1b4f") ICgcMlirInitializerOp : public ICgcMlirFunctionOp
{
    // Returns the dependency graph of executable operations (i.e., cgc.invoke) in the initializer's entry block.
    // Expected output interface(s): ICgcMlirGraph
    IFACEMETHOD(GetDependencyGraph)(REFIID riid, _COM_Outptr_ void** ppv) = 0;
};

// Encapsulates a cgc.invoke operation.
interface CGC_DECLARE_INTERFACE("1c81f026-ee1c-4c3d-83fb-0e5cae4e4ad1") ICgcMlirInvokeOp : public ICgcMlirOperation
{
    // Returns the operation called by this invoke op.
    // Expected output interface(s): ICgcMlirPartitionOp, ICgcMlirSubgraphOp, ICgcMlirOperation
    IFACEMETHOD(GetCallee)(REFIID riid, _COM_Outptr_ void** ppv) = 0;
};

// Encapsulates a cgc_partition.partition operation.
interface CGC_DECLARE_INTERFACE("c64f03bd-e99c-4c92-9e05-173153c07d2b") ICgcMlirPartitionOp : public ICgcMlirFunctionOp
{
    // Clones the partition and its dependencies (subgraphs) into a new self-contained module.
    // This produces the "partition IR" expected by MLIR state objects.
    // Expected output interface(s): ICgcMlirModuleOp
    IFACEMETHOD(Isolate)(REFIID riid, _COM_Outptr_ void** ppv) = 0;

    // Returns the dependency graph of executable operations (i.e., cgc.invoke) in the partition's entry block.
    // Expected output interface(s): ICgcMlirGraph
    IFACEMETHOD(GetDependencyGraph)(REFIID riid, _COM_Outptr_ void** ppv) = 0;
};

// Encapsulates a cgc_subgraph.subgraph operation.
interface CGC_DECLARE_INTERFACE("dc5d6b2e-285a-4c64-a7d7-7d135819e45e") ICgcMlirSubgraphOp : public ICgcMlirOperation
{
};

// Encapsulates a cgc_subgraph.functional_definition operation.
interface CGC_DECLARE_INTERFACE("98a31bd4-7867-4a80-9941-5fc2189bee78") ICgcMlirFunctionalDefinitionOp : public ICgcMlirFunctionOp
{
    // Returns the original dataflow graph of operations implemented by this subgraph.
    // This abstracts the body of the cgc_subgraph.functional_definition operation.
    // Expected output interface(s): ICgcMlirGraph
    IFACEMETHOD(GetDataflowGraph)(REFIID riid, _COM_Outptr_ void** ppv) = 0;
};

// Encapsulates a cgc_op.constant operation.
interface CGC_DECLARE_INTERFACE("65eb9de7-5b35-43ac-bca8-2e42d5688201") ICgcMlirConstantOp : public ICgcMlirOperation
{
    // Returns the constant data.
    // Expected output interface(s): ICgcMlirSpan
    IFACEMETHOD(GetData)(REFIID riid, _COM_Outptr_ void** ppv) = 0;

    // Returns the constant resource's human-readable key.
    IFACEMETHOD(GetDataKey)(_COM_Outptr_ ICgcMlirString** ppv) = 0;

    // Returns TRUE if this constant carries only a key with no embedded data.
    // Such constants must be resolved via an ExternalDataProvider at runtime.
    IFACEMETHOD_(BOOL, HasExternalData)() = 0;

    // Returns the constant result value.
    // Expected output interface(s): ICgcMlirValue
    IFACEMETHOD(GetValue)(REFIID riid, _COM_Outptr_ void** ppv) = 0;
};

// Encapsulates a cgc_memory.copy operation.
interface CGC_DECLARE_INTERFACE("a3b7c9d1-e5f2-4a68-9c1d-7e3f5b8a2d4c") ICgcMlirCopyOp : public ICgcMlirOperation
{
    // Returns the source value being copied from.
    // Expected output interface(s): ICgcMlirValue
    IFACEMETHOD(GetSource)(REFIID riid, _COM_Outptr_ void** ppv) = 0;

    // Returns the target value being copied to.
    // Expected output interface(s): ICgcMlirValue
    IFACEMETHOD(GetTarget)(REFIID riid, _COM_Outptr_ void** ppv) = 0;
};

// Encapsulates a cgc_memory.view_memref operation.
interface CGC_DECLARE_INTERFACE("7a8c3f2e-9d41-4b5a-b8e3-1f6c5d8a9e7b") ICgcMlirViewMemrefOp : public ICgcMlirOperation
{
    // Returns the value viewed by this operation.
    // Expected output interface(s): ICgcMlirValue
    IFACEMETHOD(GetSource)(REFIID riid, _COM_Outptr_ void** ppv) = 0;

    // Returns the view offset in bytes.
    IFACEMETHOD(GetOffset)(_Out_ UINT64* offset) = 0;
};
