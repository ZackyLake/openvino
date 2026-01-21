// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "implementation_manager.hpp"
#include "program_node.h"
#include "primitive_inst.h"

__declspec(dllimport) void PutMarker(std::string&& txt) noexcept;

namespace cldnn {

shape_types ImplementationManager::get_shape_type(const kernel_impl_params& impl_params) {
    for (auto& in_shape : impl_params.input_layouts) {
        if (in_shape.is_dynamic()) {
            return shape_types::dynamic_shape;
        }
    }
    for (auto& out_shape : impl_params.output_layouts) {
        if (out_shape.is_dynamic()) {
            return shape_types::dynamic_shape;
        }
    }

    return shape_types::static_shape;
}

shape_types ImplementationManager::get_shape_type(const program_node& node) {
    for (auto& in_layout : node.get_input_layouts()) {
        if (in_layout.is_dynamic()) {
            return shape_types::dynamic_shape;
        }
    }
    for (auto& out_layout : node.get_output_layouts()) {
        if (out_layout.is_dynamic()) {
            return shape_types::dynamic_shape;
        }
    }

    return shape_types::static_shape;
}

bool ImplementationManager::is_supported(const program_node& node, const std::set<key_type>& supported_keys, shape_types supported_shape_type) {
    auto key_in = implementation_key()(!node.get_dependencies().empty() ? node.get_input_layout(0) : layout{ov::PartialShape{}, data_types::f32, format::any});
    if (!supported_keys.empty() && supported_keys.find(key_in) == supported_keys.end())
        return false;

    // calc_output_layouts() if layout is not valid looks redundant, but some tests fail w/o it due to
    // layout invalidation on get_input_layout() call
    auto key_out = implementation_key()(node.get_outputs_count() > 0
                                        ? node.is_valid_output_layout(0) ? node.get_output_layout(0) : node.calc_output_layouts()[0]
                                        : layout{ov::PartialShape{}, data_types::f32, format::any});
    if (!supported_keys.empty() && supported_keys.find(key_out) == supported_keys.end())
        return false;

    return true;
}


struct RunRecord
{
    std::map<std::thread::id, std::vector<std::pair<const char*, uint64_t>>> RunMap;
    std::atomic_flag Lock = ATOMIC_FLAG_INIT;
    ~RunRecord()
    {
        while (Lock.test_and_set());
        std::map<std::string_view, std::pair<float, uint32_t>, std::less<>> nameMap;
        float maxTime = 0, totalTime = 0;
        for (const auto& [tid, runs] : RunMap) 
        {
            float time = 0;
            for (const auto& [name, t] : runs)
            {
                auto& [ntime, ncnt] = nameMap[name];
                const auto ms = static_cast<float>(t) / 1000.f;
                ntime += ms, ncnt++, time += ms;
            }
            maxTime = std::max(maxTime, time);
            totalTime += time;
        }
        printf("@@##ImplMan [%zu]Threads Total[%.2f]ms Max[%.2f]ms\n", RunMap.size(), totalTime, maxTime);
        for (const auto& [name, info] : nameMap)
        {
            const auto& [time, cnt] = info;
            auto nstr = name.data();
            if (name.size() > 7 && name.substr(0, 7) == "struct ")
                nstr += 7;
            printf("---- [%40s] : [%8.2f]ms @ [%4u]\n", nstr, time, cnt);
        }
    }
    void Put(const char* name, uint64_t us) noexcept
    {
        const auto tid = std::this_thread::get_id();
        {
            while (Lock.test_and_set());
            auto& runs = RunMap[tid];
            runs.emplace_back(name, us);
            Lock.clear();
        }
    }
    struct Rec
    {
        const std::chrono::high_resolution_clock::time_point Tbegin = std::chrono::high_resolution_clock::now();
        RunRecord& Host;
        const char* Name;
        Rec(RunRecord& host, const char* name) noexcept : Host(host), Name(name) {}
        ~Rec()
        {
            const auto tend = std::chrono::high_resolution_clock::now();
            Host.Put(Name, std::chrono::duration_cast<std::chrono::microseconds>(tend - Tbegin).count());
        }
    };
    Rec Mark(const char* name) noexcept
    {
        return Rec(*this, name);
    }
};

std::unique_ptr<primitive_impl> ImplementationManager::create(const program_node& node, const kernel_impl_params& params) const {
    static RunRecord Records;
    auto mark = Records.Mark(get_type_info().name);
    PutMarker(std::string("[impl]") + mark.Name);
    if (auto impl = create_impl(node, params)) {
        update_impl(*impl, params);
        impl->set_node_params(node);
        impl->can_share_kernels = node.get_program().get_config().get_enable_kernels_reuse();
        return impl;
    }

    return nullptr;
}

std::unique_ptr<primitive_impl> ImplementationManager::create(const kernel_impl_params& params) const {
    if (auto impl = create_impl(params)) {
        update_impl(*impl, params);
        return impl;
    }

    return nullptr;
}

void ImplementationManager::update_impl(primitive_impl& impl, const kernel_impl_params& params) const {
    impl.set_dynamic((get_shape_type() & get_shape_type(params)) == shape_types::dynamic_shape);
    impl.m_manager = this;
}

} // namespace cldnn
