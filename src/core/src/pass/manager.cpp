// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/manager.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <future>
#if defined(_WIN32)
#define NOMINMAX 1
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#include <psapi.h>
#else
#include <fstream>
#include <unistd.h>
#endif

#include "itt.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/log.hpp"
#include "perf_counters.hpp"

#ifdef ENABLE_PROFILING_ITT_FULL

namespace ov {
namespace pass {
namespace {
PerfCounters& perf_counters() {
    static PerfCounters counters;
    return counters;
}
}  // namespace
}  // namespace pass
}  // namespace ov

#endif  // ENABLE_PROFILING_ITT_FULL

#include <optional>

struct MarkerRecords
{
    struct Marker
    {
        std::chrono::high_resolution_clock::time_point Time;
        std::string Txt;
        Marker(std::string&& txt) noexcept : Time(std::chrono::high_resolution_clock::now()), Txt(std::move(txt)) {}
    };
    std::deque<Marker> Markers;
    std::atomic_flag Lock = ATOMIC_FLAG_INIT;
    std::chrono::high_resolution_clock::time_point Begin = std::chrono::high_resolution_clock::now();
    void PutMarker(std::string&& txt) noexcept
    {
        while (Lock.test_and_set());
        Markers.emplace_back(std::move(txt));
        Lock.clear();
    }
    ~MarkerRecords()
    {
        while (Lock.test_and_set());
        printf("@@##MarkerRecord: [%zu] marker\n", Markers.size());
        if (!Markers.empty())
        {
            const uint64_t tzero = Begin.time_since_epoch().count();
            auto fpm = fopen(("memmarker" + std::to_string(tzero) + ".csv").c_str(), "w+");
            fprintf(fpm, "id,time,txt\n");
            uint32_t idx = 0;
            for (const auto& marker : Markers)
                fprintf(fpm, "%u,%zu,\"%s\"\n", idx++, std::chrono::duration_cast<std::chrono::microseconds>(marker.Time - Begin).count(), marker.Txt.c_str());
            fclose(fpm);
        }
    }
};

__declspec(dllexport) void PutMarker(std::string&& txt) noexcept 
{
    static auto records = []() -> std::optional<MarkerRecords>
    { 
        const auto evar = std::getenv("marker");
        if (evar && evar == std::string("true"))
            return std::optional<MarkerRecords>(std::in_place);
        return {};
    }();
    if (records)
        records->PutMarker(std::move(txt));
}



namespace {

/**
 * @brief EnvVar gets the environment variable value by name.
 * It tries to interpret the value as boolean, if it fails then
 * the original string value is stored. This behavior helps us to reduce the number
 * of the additional env variables.
 *
 * Example of usage:
 * if OV_ENABLE_PROFILE_PASS is true, it enables console output.
 * if OV_ENABLE_PROFILE_PASS contains a path to file (string), the out logs
 * will be re-directed to the file.
 */
class EnvVar {
public:
    explicit EnvVar(const std::string& var) {
        const auto& val = ov::util::getenv_string(var.c_str());
        std::set<std::string> off = {"0", "false", "off"};
        std::set<std::string> on = {"1", "true", "on"};

        const auto& val_lower = ov::util::to_lower(val);
        if (off.count(val_lower)) {
            m_is_bool = true;
        } else if (on.count(val_lower)) {
            m_is_bool = true;
            b_value = true;
        } else {
            s_value = val;
        }
    }

    /**
     * @brief This ctor helps to activate/deactivate EnvVar from the code.
     */
    explicit EnvVar(const std::string& var, bool activate) {
        m_is_bool = true;
        b_value = activate;
    }

    bool is_enabled() const {
        return b_value || !s_value.empty();
    }

    bool is_bool() const {
        return m_is_bool;
    }

    const std::string& get_str() const {
        return s_value;
    }

private:
    bool m_is_bool = false;
    bool b_value = false;
    std::string s_value;
};

class stopwatch {
public:
    void start() {
        if (!m_active) {
            m_active = true;
            m_start_time = m_clock.now();
        }
    }

    void stop() {
        if (m_active) {
            m_end_time = m_clock.now();
            m_last_time = m_end_time - m_start_time;
            m_active = false;
        }
    }

    std::chrono::nanoseconds get_timer_value() const {
        if (m_active) {
            return (m_clock.now() - m_start_time);
        } else {
            return m_last_time;
        }
    }

    size_t get_milliseconds() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(get_timer_value()).count();
    }

    std::chrono::nanoseconds get_start_time() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(m_start_time.time_since_epoch());
    }

    std::chrono::nanoseconds get_end_time() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(m_end_time.time_since_epoch());
    }

private:
    std::chrono::high_resolution_clock m_clock;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start_time, m_end_time;
    bool m_active = false;
    std::chrono::nanoseconds m_last_time = std::chrono::high_resolution_clock::duration::zero();
};

class Profiler {
    struct PassRecord
    {
        struct MemMetric
        {
            uint64_t WorkingSet;
            uint64_t PrivateWorkingSet;
            uint64_t CommitCharge;
#if defined(_WIN32)
            MemMetric(const PROCESS_MEMORY_COUNTERS_EX2& pmc) noexcept :
                WorkingSet(pmc.WorkingSetSize),
                PrivateWorkingSet(pmc.PrivateWorkingSetSize),
                CommitCharge(pmc.PagefileUsage)
            { }
#else
            MemMetric(const std::array<uint64_t, 3>& pmc) noexcept :
                WorkingSet(pmc[0]),
                PrivateWorkingSet(pmc[1]),
                CommitCharge(pmc[2])
            { }
#endif
            MemMetric() noexcept : WorkingSet(0), PrivateWorkingSet(0), CommitCharge(0) {}
            void UpdatePeak(const MemMetric& other) noexcept
            {
                WorkingSet = std::max(WorkingSet, other.WorkingSet);
                PrivateWorkingSet = std::max(PrivateWorkingSet, other.PrivateWorkingSet);
                CommitCharge = std::max(CommitCharge, other.CommitCharge);
            }
            void Print() const noexcept
            {
                constexpr auto PrintSize = [](uint64_t size) 
                {
                    constexpr char units[] = "BKMGTPE";
                    uint32_t uidx = 0;
                    while (size >= 100000)
                        uidx++, size /= 1024;
                    printf(" [%5u]%c", static_cast<uint32_t>(size), units[uidx]);
                };
                PrintSize(WorkingSet);
                PrintSize(PrivateWorkingSet);
                PrintSize(CommitCharge);
            }
        };
        struct CallEntry
        {
            std::string Name;
            std::vector<CallEntry*> Childs;
            MemMetric MemPeak, NestedPeak;
            CallEntry* Parent = nullptr;
            std::thread::id Tid;
            float TimeMs = 0.f;
            uint32_t Depth = 0;
            bool Changed = false;
            static CallEntry*& GetCurrent() noexcept 
            {
                thread_local CallEntry* cur = nullptr;
                return cur;
            }
            CallEntry(const std::string_view& name) noexcept : Name(name), Tid(std::this_thread::get_id())
            {
                auto& cur = GetCurrent();
                Parent = cur;
                Depth = Parent ? Parent->Depth + 1 : 0;
                cur = this;
            }
            std::string GetStack(const std::unordered_map<std::string_view, std::string_view>* nameMap = nullptr) const noexcept
            {
                std::string stack;
                for (auto e = Parent; e; e = e->Parent)
                {
                    std::string_view name = e->Name;
                    if (nameMap)
                    {
                        if (const auto it = nameMap->find(name); it != nameMap->end())
                            name = it->second;
                    }
                    stack.append("<-[").append(name).append("]");
                }
                if (stack.empty())
                    stack = "(root)";
                return stack;
            }
            constexpr bool operator<(const CallEntry& rhs) const noexcept
            {
                return TimeMs == rhs.TimeMs ? (Depth == rhs.Depth ? Changed < rhs.Changed : Depth > rhs.Depth)
                                            : TimeMs < rhs.TimeMs;
            }
        };
        struct MemInfo : public MemMetric
        {
            CallEntry* Entry;
            uint64_t TimeUs; 
            template<typename... Args>
            MemInfo(CallEntry* entry, uint64_t timeUs, const Args&... args) noexcept : 
                MemMetric(args...), Entry(entry), TimeUs(timeUs) { }
        };
        std::deque<CallEntry> CallEntries;
        std::deque<MemInfo> MemInfos;
        std::atomic_flag Lock = ATOMIC_FLAG_INIT;
        std::atomic_flag ThreadRuning = ATOMIC_FLAG_INIT;
        std::atomic<CallEntry*> LatestEntry = nullptr;
        std::thread::id MainTid = std::this_thread::get_id();
        std::thread MemThread;
        static void PrintStack(std::vector<CallEntry*>& vec, const std::string& prefix, const float timeLimit, const std::unordered_map<std::string_view, std::string_view> nameMap, bool nestedMem = true) noexcept
        {
            std::sort(vec.begin(), vec.end(), [](const auto& lhs, const auto& rhs) { return *rhs < *lhs; });
            for (auto& entry : vec)
            {
                if (entry->TimeMs < timeLimit)
                    break;
                std::string_view name = entry->Name;
                if (const auto it = nameMap.find(name); it != nameMap.end())
                    name = it->second;
                printf("%s[%-*s]: [%8.3f]ms (%c)[%2zu] Mem:", prefix.c_str(), static_cast<uint32_t>(60 - prefix.size()), name.data(),
                    entry->TimeMs, entry->Changed ? '+' : '-', entry->Childs.size());
                (nestedMem ? entry->NestedPeak : entry->MemPeak).Print();
                printf("\n");
                PrintStack(entry->Childs, "|-" + prefix, timeLimit, nameMap, nestedMem);
            }
        };
        PassRecord() noexcept {}
        ~PassRecord() 
        {
            ThreadRuning.clear();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            while (Lock.test_and_set());
            for (const auto& info : MemInfos)
            {
                if (info.Entry)
                    info.Entry->MemPeak.UpdatePeak(info);
            }
            struct Info 
            {
                std::vector<const CallEntry*> Entries;
                MemMetric MemPeak;
                float TotalTime = 0.f;
                uint32_t ChangedCount = 0;
                void Sort() noexcept 
                {
                    std::sort(Entries.begin(), Entries.end(), [](const auto& lhs, const auto& rhs) { return *rhs < *lhs; });
                }
                constexpr bool operator<(const Info& rhs) const noexcept 
                {
                    return TotalTime == rhs.TotalTime ? ChangedCount < rhs.ChangedCount : TotalTime < rhs.TotalTime;
                }
            };
            std::unordered_map<std::string_view, Info> nameMap;
            std::vector<CallEntry*> root;
            std::map<std::thread::id, size_t> tidMap;  // current ST, reserved
            float totalTime = 0.f;
            for (auto& entry : CallEntries) 
            {
                tidMap.try_emplace(entry.Tid, tidMap.size());
                auto& info = nameMap[entry.Name];
                info.Entries.push_back(&entry);
                info.TotalTime += entry.TimeMs;
                info.ChangedCount += entry.Changed ? 1 : 0;
                if (!entry.Parent)
                    root.push_back(&entry), totalTime += entry.TimeMs;
                else
                    entry.Parent->Childs.push_back(&entry);
            }
            {
                std::deque<CallEntry*> updateList;
                std::set<CallEntry*> updateFilter;
                for (auto& entry : CallEntries)
                {
                    entry.NestedPeak = entry.MemPeak;
                    if (entry.Childs.empty())
                        updateList.emplace_back(&entry), updateFilter.insert(&entry);
                }
                while (!updateList.empty())
                {
                    const auto entry = updateList.front();
                    updateList.pop_front(), updateFilter.erase(entry);
                    if (entry->Parent) 
                    {
                        entry->Parent->NestedPeak.UpdatePeak(entry->MemPeak);
                        if (updateFilter.insert(entry->Parent).second)
                            updateList.emplace_back(entry->Parent);
                    }
                }
            }
            std::unordered_map<std::string_view, std::string_view> shortNames;
            std::vector<Info*> infos;
            for (auto& [name_, info] : nameMap) 
            {
                info.Sort();
                for (const auto& entry : info.Entries)
                    info.MemPeak.UpdatePeak(entry->MemPeak);
                infos.push_back(&info);
                auto name = name_;
                if (name.size() > 6 && name.substr(0, 6) == "class ")
                    name.remove_prefix(6);
                if (name.size() > 10 && name.substr(0, 10) == "ov::pass::")
                    name.remove_prefix(10);
                shortNames.insert_or_assign(name_, name);
            }
            std::sort(infos.begin(), infos.end(), [](const auto& lhs, const auto& rhs) { return *rhs < *lhs; });
            printf("@@##Passes costs [%.2f]ms:\n", totalTime);
            PrintStack(root, "", totalTime * 0.01f, shortNames, false);
            printf("@@##Top 10 Passes:\n");
            for (uint32_t infoIdx = 0; infoIdx < 10 && infoIdx < infos.size(); ++infoIdx) // Top10
            {
                const auto& info = *infos[infoIdx];
                const auto& first = *(info.Entries.front());
                const auto name = shortNames[first.Name];
                printf("--[%-60s]: [%8.3f]ms +[%2u]/[%2zu] Mem:", name.data(), info.TotalTime, info.ChangedCount, info.Entries.size());
                info.MemPeak.Print();
                printf("\n");
                if (info.Entries.size() < 2 && first.Parent)
                    printf("   %s\n", first.GetStack(&shortNames).c_str());
                else 
                {
                    for (uint32_t i = 0; i < 3 && i < info.Entries.size(); ++i)  // Top3 details
                    {
                        const auto& entry = *(info.Entries[i]);
                        printf("  --[%8.3f]ms %s Mem:", entry.TimeMs, entry.GetStack(&shortNames).c_str());
                        entry.MemPeak.Print();
                        printf("\n");
                    }
                }
            }
        }
        void Begin(const std::string& name) noexcept
        {
            PutMarker("[pass]" + name);
            while (Lock.test_and_set());
            if (!ThreadRuning.test_and_set())
            {
                ThreadRuning.clear();
                std::promise<void> pms;
                MemThread = std::thread([&]() 
                {
                    printf("Memory Info Collector Running!\n");
#if defined(_WIN32)
                    DWORD processId = GetCurrentProcessId();
                    HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, processId);
#endif
                    ThreadRuning.test_and_set();
                    pms.set_value();
                    const auto timeBegin = std::chrono::high_resolution_clock::now();
                    do
                    {
                        bool success = false;
#if defined(_WIN32)
                        PROCESS_MEMORY_COUNTERS_EX2 pmc;
                        success = GetProcessMemoryInfo(hProcess, reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc));
#else
                        std::array<uint64_t, 3> pmc = {};
                        static const uint32_t PageSize = sysconf(_SC_PAGE_SIZE);
                        try
                        {
                            std::ifstream stat("/proc/self/statm", std::ios_base::in);
                            uint64_t vms = 0, rss = 0, shared = 0;
                            stat >> vms >> rss >> shared;
                            success = true;
                            pmc[2] = vms * PageSize, pmc[0] = rss * PageSize, pmc[1] = (rss - shared) * PageSize;
                        } catch (const std::exception&) {}
#endif
                        if (success)
                        {
                            const auto timeEnd = std::chrono::high_resolution_clock::now();
                            const uint64_t timeElapse = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeBegin).count();
                            MemInfos.emplace_back(LatestEntry.load(), timeElapse, pmc);
                        }
                        std::this_thread::sleep_for(std::chrono::microseconds(500));
                    } while (ThreadRuning.test_and_set());
                    printf("Memory Info Collector Exiting!\n");
                });
                MemThread.detach();
                pms.get_future().wait();
            }
            auto& entry = CallEntries.emplace_back(name);
            if (std::this_thread::get_id() == MainTid)
                LatestEntry = &entry;
            Lock.clear();
        }
        void End(const std::string& name, uint64_t us, bool applied) noexcept
        {
            PutMarker("[pass]@" + name);
            auto& entry = CallEntry::GetCurrent();
            if (!entry)
            {
                printf("!!## current in [NULL] but ending [%s]!\n", name.c_str());
                return;
            }
            if (entry->Name == name)
            {
                entry->TimeMs = static_cast<float>(us) / 1000.f;
                entry->Changed = applied;
                entry = entry->Parent;
            }
            else
            {
                printf("!!## current in [%s] but ending [%s]!\n--- inside %s\n",
                       entry->Name.c_str(),
                       name.c_str(),
                       entry->GetStack().c_str());
            }
            if (std::this_thread::get_id() == MainTid) 
            {
                while (Lock.test_and_set());
                LatestEntry = entry;
                Lock.clear();
            }
        }
    };
    inline static PassRecord Records = {};
public:
    /**
     * @brief Profiler class helps to analyze Transformations execution times, visualize/serialize ov model after all
     * or for dedicated Transformations.
     *
     *  There are 3 environment variables which can be set for Transformations debugging:
     *
     *  1. OV_ENABLE_PROFILE_PASS - Enables profiling of transformation passes to log their execution times.
     *
     *      Usage: Set this environment variable to "true" to enable visualizations.
     *      Alternatively, specify a file path where the execution times will be saved.
     *
     *      Example:
     *      export OV_ENABLE_PROFILE_PASS=true
     *      export OV_ENABLE_PROFILE_PASS="/path/to/save/profiling/results"
     *
     *  2. OV_ENABLE_VISUALIZE_TRACING - Enables visualization of the model to .svg file after each transformation pass.
     *
     *      Usage: Set this environment variable to "true", "on" or "1" to enable visualization for all Transformations.
     *
     *      Filtering: You can specify filters to control which passes are visualized.
     *      If the variable is set to a specific filter string (e.g., "PassName", "PassName1,PassName2"),
     *      only transformations matching that filter will be visualized. Delimiter is ",".
     *
     *      Example:
     *      export OV_ENABLE_VISUALIZE_TRACING=true
     *      export OV_ENABLE_VISUALIZE_TRACING="Pass1,Pass2,Pass3"
     *
     *  3. OV_ENABLE_SERIALIZE_TRACING - Enables serialization of the model to .xml/.bin after each transformation pass.
     *
     *      Usage: Set this environment variable to "true", "on" or "1" to enable serialization for all Transformations.
     *
     *      Filtering: You can specify filters to control which passes are serialized.
     *      If the variable is set to a specific filter string (e.g., "PassName", "PassName1,PassName2"),
     *      only transformations matching that filter will be serialized. Delimiter is ",".
     *
     *      Example:
     *      export OV_ENABLE_SERIALIZE_TRACING=true
     *      export OV_ENABLE_SERIALIZE_TRACING="Pass1,Pass2,Pass3"
     *
     */
    explicit Profiler(std::string manager_name)
        : m_visualize("OV_ENABLE_VISUALIZE_TRACING"),
          m_serialize("OV_ENABLE_SERIALIZE_TRACING"),
          m_profile_pass("OV_ENABLE_PROFILE_PASS"),
          m_manager_name(std::move(manager_name)) {
        if (m_profile_pass.is_enabled() && !m_profile_pass.is_bool()) {
            m_file.open(m_profile_pass.get_str(), std::ios_base::app);
        }
    }

    ~Profiler() {
        if (m_file.is_open()) {
            m_file.close();
        }
    }

    void start_timer(const std::string& name) {
        stopwatches[name] = stopwatch();
        stopwatches[name].start();
        Records.Begin(name);

        bool is_pass_manager = name == m_manager_name;
        if (m_profile_pass.is_enabled() && is_pass_manager) {
            std::cout << std::setw(25) << std::left;
            std::cout << "PassManager started: " << m_manager_name << std::endl;
            std::cout << std::right;
        }
    }

    void stop_timer(const std::string& name, bool applied) {
        auto& stopwatch = stopwatches.at(name);
        stopwatch.stop();
        Records.End(name, std::chrono::duration_cast<std::chrono::microseconds>(stopwatch.get_timer_value()).count(), applied);

        bool is_pass_manager = name == m_manager_name;
        if (m_profile_pass.is_enabled()) {
            if (m_profile_pass.is_bool()) {
                std::cout << std::setw(25) << std::left;
                if (is_pass_manager) {
                    std::cout << "PassManager finished: ";
                } else {
                    std::cout << "  ";
                }
                std::cout << std::setw(60) << std::left << name;
                std::cout << std::setw(5) << std::right << stopwatch.get_milliseconds() << "ms "
                          << (applied ? "+" : "-") << std::endl;
            } else if (m_file.is_open()) {
                if (is_pass_manager) {
                    m_file << "m;" << name << ";" << stopwatch.get_timer_value().count() << ";" << (applied ? "1" : "0")
                           << std::endl;
                    m_file << "m_start;" << name << ";" << stopwatch.get_start_time().count() << std::endl;
                    m_file << "m_end;" << name << ";" << stopwatch.get_end_time().count() << std::endl;
                } else {
                    m_file << "t;" << name << ";" << m_manager_name << ";" << stopwatch.get_timer_value().count() << ";"
                           << (applied ? "1" : "0") << std::endl;
                }
            } else {
                OPENVINO_THROW("The output file for logging transformation statistics is closed. "
                               "Recording of statistics is not possible.");
            }
        }
    }

    void visualize(const std::shared_ptr<ov::Model>& model, const std::string& pass_name) const {
        static size_t viz_index = 0;
        if (m_visualize.is_enabled()) {
            const auto& _visualize = [&]() {
                const auto& file_name = gen_file_name(model->get_name(), pass_name, viz_index++);
                ov::pass::VisualizeTree vt(file_name + ".svg");
                vt.run_on_model(model);
            };

            if (m_visualize.is_bool()) {
                _visualize();
            } else {
                const auto& filter_tokens = ov::util::split_by_delimiter(m_visualize.get_str(), ',');
                for (const auto& token : filter_tokens) {
                    if (pass_name.find(token) != std::string::npos) {
                        _visualize();
                        return;
                    }
                }
            }
        }
    }

    void serialize(const std::shared_ptr<ov::Model>& model, const std::string& pass_name) const {
        static size_t serialize_index = 0;
        if (m_serialize.is_enabled()) {
            const auto& _serialize = [&]() {
                const auto& file_name = gen_file_name(model->get_name(), pass_name, serialize_index++);
                ov::pass::Serialize serialize(file_name + ".xml", file_name + ".bin");
                serialize.run_on_model(model);
            };

            if (m_serialize.is_bool()) {
                _serialize();
            } else {
                const auto& filter_tokens = ov::util::split_by_delimiter(m_serialize.get_str(), ',');
                for (const auto& token : filter_tokens) {
                    if (pass_name.find(token) != std::string::npos) {
                        _serialize();
                        return;
                    }
                }
            }
        }
    }

private:
    static std::string gen_file_name(const std::string& model_name, const std::string& pass_name, const size_t idx) {
        std::stringstream name;
        // visualizations and serializations will be named after the outermost function
        std::string index_str = std::to_string(idx);
        const size_t num_digits_in_pass_index = index_str.length() > 2LU ? 0LU : (3LU - index_str.length());
        index_str = std::string(num_digits_in_pass_index, '0') + index_str;

        name << model_name << std::string("_") << index_str << std::string("_") << pass_name;
        return name.str();
    }

    std::unordered_map<std::string, stopwatch> stopwatches;

    EnvVar m_visualize;
    EnvVar m_serialize;
    EnvVar m_profile_pass;

    std::string m_manager_name;
    std::fstream m_file;
};

}  // namespace

ov::pass::Manager::Manager() : m_pass_config(std::make_shared<PassConfig>()) {}

ov::pass::Manager::~Manager() = default;

ov::pass::Manager::Manager(std::string name) : m_pass_config(std::make_shared<PassConfig>()), m_name(std::move(name)) {}

ov::pass::Manager::Manager(std::shared_ptr<ov::pass::PassConfig> pass_config, std::string name)
    : m_pass_config(std::move(pass_config)),
      m_name(std::move(name)) {}

ov::pass::Manager::Manager(const PassConfig& pass_config, std::string name)
    : Manager(std::make_shared<PassConfig>(pass_config), std::move(name)) {}

void ov::pass::Manager::set_per_pass_validation(bool new_state) {
    m_per_pass_validation = new_state;
}

bool ov::pass::Manager::run_passes(const std::shared_ptr<ov::Model>& model) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::ov_core, "pass::Manager::run_passes");
    Profiler profiler(m_name);

    bool model_changed = false;
    bool pass_changed_model = false;

    profiler.start_timer(m_name);
    for (const auto& pass : m_pass_list) {
        const auto& pass_name = pass->get_name();

        profiler.start_timer(pass_name);
        pass_changed_model = run_pass(pass, model, pass_changed_model);
        profiler.stop_timer(pass_name, pass_changed_model);

        model_changed = model_changed || pass_changed_model;

        profiler.visualize(model, pass_name);
        profiler.serialize(model, pass_name);
    }
    profiler.stop_timer(m_name, model_changed);

    return model_changed;
}

bool ov::pass::Manager::run_pass(const std::shared_ptr<PassBase>& pass,
                                 const std::shared_ptr<Model>& model,
                                 bool needs_validate) {
    if (m_pass_config->is_disabled(pass->get_type_info())) {
        OPENVINO_DEBUG("Pass ", pass->get_name(), " is disabled.");
        return false;
    }

    // This checks if we need to skip the graph transformation when the graph pass relies on
    // static shape but the model state is dynamic.
    if (pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) && model->is_dynamic()) {
        OPENVINO_DEBUG("Pass ",
                       pass->get_name(),
                       " requires static shape but the ",
                       "model is dynamic. Skipping this transformation.");
        return false;
    }

    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::ov_pass, ov::pass::perf_counters()[pass->get_type_info()]);

    if (auto matcher_pass = ov::as_type_ptr<MatcherPass>(pass)) {
        // GraphRewrite is a temporary container for MatcherPass to make execution on entire ov::Model
        return GraphRewrite(matcher_pass).run_on_model(model);
    } else if (auto model_pass = ov::as_type_ptr<ModelPass>(pass)) {
        if (ov::as_type_ptr<ov::pass::Validate>(model_pass) && !needs_validate) {
            return false;
        }
        return model_pass->run_on_model(model);
    }
    return false;
}
