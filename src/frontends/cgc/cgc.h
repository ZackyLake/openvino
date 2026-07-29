// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <comdef.h>

#ifndef CGC_DECLARE_INTERFACE
#define CGC_DECLARE_INTERFACE(iid) DECLSPEC_UUID(iid) DECLSPEC_NOVTABLE
#endif

STDAPI CgcCreateFactory(REFIID riid, _COM_Outptr_opt_ void** ppv);

interface ICgcBlob;
interface ICgcTarget;
interface ICgcParameters;
interface ICgcCompiler;
interface ICgcCompileResult;

enum CGC_ENCODING
{
    CGC_ENCODING_UNSPECIFIED,
    CGC_ENCODING_MLIR_BYTECODE,
    CGC_ENCODING_MLIR_TEXT,
};

enum CGC_TARGET_PROPERTY
{
    // Unique name of the target with respect to other targets in the same compiler.
    // This is used to map subgraph specialization requests to the matching target that
    // declared a pattern, and it is tracked in Output IR to associate partitions with 
    // specific targets.
    // Type: char* (UTF8) without null terminator
    CGC_TARGET_PROPERTY_NAME,

    // Locally unique ID of the accelerator (e.g., DXCore adapter). This is used to 
    // unambiguously identify the accelerator in case it is used across multiple targets 
    // in the same compiler.
    // Type: LUID
    CGC_TARGET_PROPERTY_ACCELERATOR_LUID,

    // The hardware accelerator represents physical hardware.
    // Type: bool
    CGC_TARGET_PROPERTY_ACCELERATOR_IS_HARDWARE,

    // Description string for the hardware accelerator.
    // Type: char* (UTF8)
    CGC_TARGET_PROPERTY_ACCELERATOR_DRIVER_DESCRIPTION,

    // Driver version for the hardware accelerator.
    // Type: char* (UTF8)
    CGC_TARGET_PROPERTY_ACCELERATOR_DRIVER_VERSION,
};

interface CGC_DECLARE_INTERFACE("412866bc-ddc2-41ec-a588-9944b696c56f") ICgcFactory : IUnknown
{
    // Creates a compiler from one or more targets.
    IFACEMETHOD(CreateCompiler)(
        UINT targetCount,
        _In_reads_(targetCount) ICgcTarget* const* targets,
        _In_opt_ ICgcParameters* parameters,
        REFIID riid, // expected: ICgcCompiler
        _COM_Outptr_opt_ void** compiler
    ) = 0;

    IFACEMETHOD(CreateParameters)(
        REFIID riid, // expected: ICgcParameters
        _COM_Outptr_ void** parameters
    ) = 0;
};

//
// Wraps a read-only data buffer. Typically used as output of API methods that produce variable-sized data.
// Callers MUST NOT write to the memory pointed to by GetData().
//
interface CGC_DECLARE_INTERFACE("c73d4169-87fc-431b-9836-f7b75a7c521a") ICgcBlob : IUnknown
{
    IFACEMETHOD_(CGC_ENCODING, GetDataEncoding)() = 0;
    IFACEMETHOD_(const void*, GetData)() = 0;
    IFACEMETHOD_(UINT64, GetDataSizeInBytes)() = 0;
};

// ----------------------------------------------------------------------------
// ICgcTarget : represents a compilation target, typically associated with a
// specific hardware accelerator.
// ----------------------------------------------------------------------------

interface CGC_DECLARE_INTERFACE("63a4f34e-f090-4eff-a364-09fc2425cfad") ICgcTarget : IUnknown
{
    // Checks support for multiple MLIR interfaces.
    IFACEMETHOD(CheckMlirInterfaceSupport)(
        UINT numInterfaces,
        _In_reads_(numInterfaces) const GUID* mlirInterfacesRequested,
        _Out_writes_(numInterfaces) BOOL* mlirInterfacesSupported
    ) = 0;

    // Returns true if the target supports the MLIR interface.
    inline BOOL SupportsMlirInterface(const GUID& mlirInterface)
    {
        BOOL supported = FALSE;
        CheckMlirInterfaceSupport(1, &mlirInterface, &supported);
        return supported;
    }

    // Used to send and/or receive structured MLIR data to/from the target.
    // See CGC_GUIDS namespace below for predefined MLIR interfaces.
    // Implementation returns E_NOINTERFACE if the target does not support
    // the requested interface.
    IFACEMETHOD(MlirExchange)(
        REFGUID mlirInterface,
        _In_reads_bytes_(inputDataSizeInBytes) const void* inputData,
        UINT64 inputDataSizeInBytes,
        _COM_Outptr_opt_ ICgcBlob** outputData
    ) = 0;

    IFACEMETHOD_(BOOL, IsPropertySupported)(
        CGC_TARGET_PROPERTY property
    ) = 0;

    IFACEMETHOD(GetProperty)(
        CGC_TARGET_PROPERTY property,
        UINT64 bufferSize,
        _Out_writes_bytes_(bufferSize) void* propertyData
    ) = 0;

    IFACEMETHOD(GetPropertySize)(
        CGC_TARGET_PROPERTY property,
        _Out_ UINT64* propertyDataSize
    ) = 0;
};

// ----------------------------------------------------------------------------
// ICgcParameters : definition of configurable compiler options.
// ----------------------------------------------------------------------------

enum CGC_PARAMETER_TYPE
{
    CGC_PARAMETER_TYPE_STRING_UTF8, // size = variable (excludes null terminator)
    CGC_PARAMETER_TYPE_BOOL8,       // size = 8 bits  / 1 byte  / sizeof(bool)
    CGC_PARAMETER_TYPE_INT32,       // size = 32 bits / 4 bytes / sizeof(int32_t)
    CGC_PARAMETER_TYPE_INT64,       // size = 64 bits / 8 bytes / sizeof(int64_t)
    CGC_PARAMETER_TYPE_UINT32,      // size = 32 bits / 4 bytes / sizeof(uint32_t)
    CGC_PARAMETER_TYPE_UINT64,      // size = 64 bits / 8 bytes / sizeof(uint64_t)
    CGC_PARAMETER_TYPE_FLOAT32,     // size = 32 bits / 4 bytes / sizeof(float)
    CGC_PARAMETER_TYPE_FLOAT64,     // size = 64 bits / 8 bytes / sizeof(double)
};

struct CGC_PARAMETER_DESC
{
    CGC_PARAMETER_TYPE type;
    const char* name;
    const char* groupName;
    const char* description;
};

interface CGC_DECLARE_INTERFACE("1da232e3-88db-4bab-9296-fc1ed8d437eb") ICgcParameters : IUnknown
{
    // Retrieve a list of all available parameters.
    // To determine the count of parameters, call this method with parameters set to null.
    IFACEMETHOD(EnumerateParameters)(
        _Inout_ SIZE_T * parameterCount,
        _Out_writes_opt_(*parameterCount) CGC_PARAMETER_DESC* parameters
    ) = 0;

    // Retrieves the value of a named parameter.
    //
    // If the parameter is a fixed-width type (e.g., CGC_PARAMETER_TYPE_BOOL8),
    // then valueSizeInBytes matches the size of the matching C++ type (e.g. sizeof(bool)).
    //
    // If the parameter is a UTF8 string, then valueSizeInBytes is the required number of 
    // bytes to hold all characters exclusive of any null terminator (e.g., std::u8string::size).
    IFACEMETHOD(GetParameterValue)(
        const char* name,
        _Inout_ SIZE_T* valueSizeInBytes,
        _Out_writes_bytes_opt_(*valueSizeInBytes) void* value
    ) = 0;

    // Sets the value of a named parameter.
    //
    // If the parameter is a fixed-width type (e.g., CGC_PARAMETER_TYPE_BOOL8), 
    // then valueSizeInBytes matches the size of the matching C++ type (e.g. sizeof(bool)).
    //
    // If the parameter is a UTF8 string, then valueSizeInBytes is the required number of 
    // bytes to hold all characters exclusive of any null terminator (e.g., std::u8string::size).
    IFACEMETHOD(SetParameterValue)(
        const char* name,
        SIZE_T valueSizeInBytes,
        _In_reads_bytes_(valueSizeInBytes) const void* value
    ) = 0;
};

// ----------------------------------------------------------------------------
// ICgcCompiler : compiles input MLIR to run on a given target.
// ----------------------------------------------------------------------------

interface CGC_DECLARE_INTERFACE("03fadf87-a7db-41a4-afa4-c4eb453b4d6e") ICgcCompiler : IUnknown
{
    // Returns the number of targets associated with this compiler.
    IFACEMETHOD_(SIZE_T, GetTargetCount)() = 0;

    // Returns the target at the given index.
    IFACEMETHOD(GetTarget)(UINT32 index, REFIID riid, _COM_Outptr_ void** target) = 0;

    // TODO: design and implement a logger interface to get error messages programatically
    // https://dev.azure.com/cga-exchange/RTML/_workitems/edit/993
    IFACEMETHOD(Compile)(
        _In_reads_bytes_(inputMlirDataSizeInBytes) const void* inputMlirData,
        UINT64 inputMlirDataSizeInBytes,
        CGC_ENCODING outputEncoding,
        REFIID riid, // expected: ICgcBlob
        _COM_Outptr_ void** outputBlob
    ) = 0;
};

// Cross-component exchanges of MLIR data (subgraph transformations, capability checks, etc.) 
// drive the CGC compilation process. MLIR interfaces define the semantics and structure 
// of serialized data exchanged between components, and each interface is associated with a 
// GUID. Compilation targets (drivers, apps) must declare support for a subset of these 
// interfaces to participate in the compilation process.

// ------------------------------------------------------------------------
// Compiler input/output data.
// ------------------------------------------------------------------------

// CGC Input IR.
// {7FC4F23F-2474-4AE8-AE7A-2722118B350E}
DEFINE_GUID(CGC_INPUT, 
0x7fc4f23f, 0x2474, 0x4ae8, 0xae, 0x7a, 0x27, 0x22, 0x11, 0x8b, 0x35, 0xe);

// CGC Output IR.
// {F53E923F-5300-4EEF-8F1E-ADA03FFF2605}
DEFINE_GUID(CGC_OUTPUT, 
0xf53e923f, 0x5300, 0x4eef, 0x8f, 0x1e, 0xad, 0xa0, 0x3f, 0xff, 0x26, 0x5);

// ------------------------------------------------------------------------
// Subgraph-related interfaces.
//
// Targets must support all SUBGRAPH_* interfaces if they wish to optimize
// subgraphs (i.e., declare patterns, validate matches, and receive
// partitions at runtime).
// ------------------------------------------------------------------------

// A partition of CGC Output IR comprising one or more subgraphs.
// {EAA277C3-190D-4049-A7B0-A611B11701B6}
DEFINE_GUID(CGC_SUBGRAPH_PARTITION, 
0xeaa277c3, 0x190d, 0x4049, 0xa7, 0xb0, 0xa6, 0x11, 0xb1, 0x17, 0x1, 0xb6);

// Compiler-initiated request to get all subgraph transformations from a target.
// {EB53032A-1116-4E71-8309-54347C1E5A26}
DEFINE_GUID(CGC_SUBGRAPH_DECLARATION_REQUEST, 
0xeb53032a, 0x1116, 0x4e71, 0x83, 0x9, 0x54, 0x34, 0x7c, 0x1e, 0x5a, 0x26);

// Target's response to SUBGRAPH_DECLARATION_REQUEST.
// {BE441609-6AB8-4FA8-B551-019ED938C298}
DEFINE_GUID(CGC_SUBGRAPH_DECLARATION, 
0xbe441609, 0x6ab8, 0x4fa8, 0xb5, 0x51, 0x1, 0x9e, 0xd9, 0x38, 0xc2, 0x98);

// Compiler-initiated request to specialize a subgraph with concrete shapes.
// {EBED5ED6-2206-457B-8BD0-ABFCC677195B}
DEFINE_GUID(CGC_SUBGRAPH_SPECIALIZATION_REQUEST, 
0xebed5ed6, 0x2206, 0x457b, 0x8b, 0xd0, 0xab, 0xfc, 0xc6, 0x77, 0x19, 0x5b);

// Target's response to SUBGRAPH_SPECIALIZATION_REQUEST.
// {F90A981E-0FCD-46D8-8AA1-166994C4DE02}
DEFINE_GUID(CGC_SUBGRAPH_SPECIALIZATION, 
0xf90a981e, 0xfcd, 0x46d8, 0x8a, 0xa1, 0x16, 0x69, 0x94, 0xc4, 0xde, 0x2);

// Compiler-initiated request to get target configuration.
// {52C3EDEA-5041-4C92-9C99-5385A4682E0A}
DEFINE_GUID(CGC_TARGET_CONFIG_REQUEST, 
0x52c3edea, 0x5041, 0x4c92, 0x9c, 0x99, 0x53, 0x85, 0xa4, 0x68, 0x2e, 0xa);

// Target's response to TARGET_CONFIG_REQUEST.
// {87EA9104-33CB-4901-9138-28D2C6697EB0}
DEFINE_GUID(CGC_TARGET_CONFIG, 
0x87ea9104, 0x33cb, 0x4901, 0x91, 0x38, 0x28, 0xd2, 0xc6, 0x69, 0x7e, 0xb0);
