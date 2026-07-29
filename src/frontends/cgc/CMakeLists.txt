# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(TARGET_NAME openvino_cgc_frontend)


set(CGC_FE_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/cgc_frontend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cgc_op_table.cpp
)

set(CGC_FE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/cgc_conversion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cgc_ir.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cgc_ir_templates.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cgc.h
)

add_library(${TARGET_NAME} ${CGC_FE_SRC} ${CGC_FE_HEADERS})
add_library(openvino::frontend::cgc ALIAS ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO}
)

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/wil/include
        ${OpenVINO_SOURCE_DIR}/src/inference/dev_api
        $<TARGET_PROPERTY:openvino::frontend::common,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        openvino::frontend::common
        openvino::frontend::common_translators
        ${CMAKE_CURRENT_SOURCE_DIR}/cgc_ir.lib
        runtimeobject.lib
    PUBLIC
        openvino::runtime
)

# Export name for install targets
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME frontend::cgc)

target_compile_options(${TARGET_NAME} PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:ReleaseWithDebInfo>>:/Oxs>
)

add_dependencies(ov_frontends ${TARGET_NAME})
