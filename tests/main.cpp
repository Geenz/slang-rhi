#include "testing.h"
#include "memory-report.h"
#include <slang-rhi/agility-sdk.h>

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

#include "doctest-reporter.h"

#include <charconv>
#include <cstdio>
#include <string_view>

// Due to current issues in slang we don't enable Agility SDK yet
SLANG_RHI_EXPORT_AGILITY_SDK

namespace rhi::testing {

// Helpers to get current test suite and case name.
// See https://github.com/doctest/doctest/issues/345.
// Has to be defined in the same file as DOCTEST_CONFIG_IMPLEMENT
std::string getCurrentTestSuiteName()
{
    return doctest::detail::g_cs->currentTest->m_test_suite;
}
std::string getCurrentTestCaseName()
{
    return doctest::detail::g_cs->currentTest->m_name;
}

bool checkRequiredDevices()
{
    bool allAvailable = true;
    for (DeviceType deviceType : kDeviceTypes)
    {
        if (!options().deviceRequired[size_t(deviceType)])
            continue;

        DeviceAvailabilityResult result = checkDeviceTypeAvailable(deviceType);
        if (result.available)
            continue;

        allAvailable = false;
        std::fprintf(
            stderr,
            "Required device '%s' is not available: %s\n",
            deviceTypeToString(deviceType),
            result.error.c_str()
        );
        if (!result.debugCallbackOutput.empty())
            std::fprintf(stderr, "Debug callback output: %s\n", result.debugCallbackOutput.c_str());
        if (!result.diagnostics.empty())
            std::fprintf(stderr, "Slang diagnostics: %s\n", result.diagnostics.c_str());
    }
    return allAvailable;
}

} // namespace rhi::testing

// Prints a symbolized stack for the first access violation so crashes in CI/terminal runs are attributable.
#if SLANG_WINDOWS_FAMILY
#include <windows.h>
#include <psapi.h>
#include <dbghelp.h>
#pragma comment(lib, "psapi")
#pragma comment(lib, "dbghelp")
static LONG WINAPI crashProbe(EXCEPTION_POINTERS* info)
{
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION)
        return EXCEPTION_CONTINUE_SEARCH;
    static LONG once = 0;
    if (InterlockedExchange(&once, 1))
        return EXCEPTION_CONTINUE_SEARCH;
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_LOAD_LINES | SYMOPT_DEFERRED_LOADS);
    SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    auto printAddr = [](const char* label, void* addr)
    {
        HMODULE mods[512];
        DWORD needed = 0;
        char name[MAX_PATH] = "?";
        uintptr_t base = 0;
        if (EnumProcessModules(GetCurrentProcess(), mods, sizeof(mods), &needed))
        {
            for (DWORD i = 0; i < needed / sizeof(HMODULE); i++)
            {
                MODULEINFO mi;
                if (GetModuleInformation(GetCurrentProcess(), mods[i], &mi, sizeof(mi)))
                {
                    if (addr >= mi.lpBaseOfDll && addr < (char*)mi.lpBaseOfDll + mi.SizeOfImage)
                    {
                        GetModuleFileNameA(mods[i], name, sizeof(name));
                        base = (uintptr_t)mi.lpBaseOfDll;
                        break;
                    }
                }
            }
        }
        char symBuf[sizeof(SYMBOL_INFO) + 512] = {};
        SYMBOL_INFO* sym = (SYMBOL_INFO*)symBuf;
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen = 511;
        DWORD64 disp = 0;
        const char* symName = "?";
        if (SymFromAddr(GetCurrentProcess(), (DWORD64)addr, &disp, sym))
            symName = sym->Name;
        IMAGEHLP_LINE64 line = {sizeof(IMAGEHLP_LINE64)};
        DWORD lineDisp = 0;
        char lineBuf[64] = "";
        if (SymGetLineFromAddr64(GetCurrentProcess(), (DWORD64)addr, &lineDisp, &line))
            snprintf(lineBuf, sizeof(lineBuf), " [%s:%lu]", strrchr(line.FileName, '\\') ? strrchr(line.FileName, '\\') + 1 : line.FileName, line.LineNumber);
        fprintf(
            stderr,
            "[CRASH] %s %p = %s+0x%llx %s+0x%llx%s\n",
            label,
            addr,
            name,
            (unsigned long long)((uintptr_t)addr - base),
            symName,
            (unsigned long long)disp,
            lineBuf
        );
    };
    fprintf(
        stderr,
        "[CRASH] access violation %s address %p\n",
        info->ExceptionRecord->ExceptionInformation[0] ? "writing" : "reading",
        (void*)info->ExceptionRecord->ExceptionInformation[1]
    );
    printAddr("instruction", (void*)info->ContextRecord->Rip);
    void* stack[32];
    USHORT frames = CaptureStackBackTrace(0, 32, stack, nullptr);
    for (USHORT i = 0; i < frames; i++)
        printAddr("frame", stack[i]);
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

int main(int argc, const char** argv)
{
#if SLANG_WINDOWS_FAMILY
    AddVectoredExceptionHandler(1, crashProbe);
#endif
    // Store path to the executable.
    rhi::testing::exePath() = argv[0];

    rhi::testing::cleanupTestTempDirectories();

#if SLANG_RHI_DEBUG
    rhi::DebugLayerOptions debugLayerOptions{};
    debugLayerOptions.coreValidation = true;
    rhi::getRHI()->setDebugLayerOptions(debugLayerOptions);
#endif

    // Parse extra command line options.
    {
        auto& options = rhi::testing::options();

        if (doctest::parseFlag(argc, argv, "verbose"))
        {
            options.verbose = true;
        }

        if (doctest::parseFlag(argc, argv, "check-devices"))
        {
            options.checkDevices = true;
        }

        if (doctest::parseFlag(argc, argv, "list-devices"))
        {
            options.listDevices = true;
        }

        std::vector<doctest::String> strings;
        if (doctest::parseCommaSepArgs(argc, argv, "select-devices=", strings))
        {
            options.deviceSelected.fill(false);
            for (const auto& str : strings)
            {
                if (str == "*")
                {
                    options.deviceSelected.fill(true);
                    continue;
                }
                for (rhi::DeviceType deviceType : rhi::testing::kPlatformDeviceTypes)
                {
                    doctest::String deviceTypeStr = rhi::testing::deviceTypeToString(deviceType);
                    if (str == deviceTypeStr || str.substr(0, deviceTypeStr.size()) == deviceTypeStr)
                    {
                        options.deviceSelected[size_t(deviceType)] = true;
                        if (str.size() > deviceTypeStr.size() + 1 && str[deviceTypeStr.size()] == ':')
                        {
                            int adapterIndex = atoi(str.c_str() + deviceTypeStr.size() + 1);
                            options.deviceAdapterIndex[size_t(deviceType)] = adapterIndex;
                        }
                        break;
                    }
                }
            }
        }

        strings.clear();
        if (doctest::parseCommaSepArgs(argc, argv, "require-devices=", strings))
        {
            for (const auto& str : strings)
            {
                bool matched = false;
                for (rhi::DeviceType deviceType : rhi::testing::kDeviceTypes)
                {
                    if (str == rhi::testing::deviceTypeToString(deviceType))
                    {
                        options.deviceRequired[size_t(deviceType)] = true;
                        matched = true;
                        break;
                    }
                }
                if (!matched)
                {
                    std::fprintf(stderr, "Invalid required device type '%s'.\n", str.c_str());
                    return 1;
                }
            }
        }

        doctest::String d3d12ShaderModel;
        if (doctest::parseOption(argc, argv, "d3d12-shader-model=", &d3d12ShaderModel))
        {
            std::string_view value = d3d12ShaderModel.c_str();
            size_t separator = value.find('.');
            unsigned int major = 0;
            unsigned int minor = 0;
            auto parseComponent = [](std::string_view component, unsigned int& result)
            {
                auto [end, error] = std::from_chars(component.data(), component.data() + component.size(), result);
                return !component.empty() && error == std::errc() && end == component.data() + component.size();
            };
            bool validFormat = separator != std::string_view::npos &&
                               parseComponent(value.substr(0, separator), major) &&
                               parseComponent(value.substr(separator + 1), minor);
            bool validVersion = (major == 5 && minor == 1) || (major == 6 && minor <= 10);
            if (!validFormat || !validVersion)
            {
                std::fprintf(
                    stderr,
                    "Invalid D3D12 shader model '%s'; expected 5.1 or 6.0 through 6.10 in major.minor format.\n",
                    d3d12ShaderModel.c_str()
                );
                return 1;
            }
            options.d3d12ShaderModel = (major << 4) | minor;
        }

        if (doctest::parseFlag(argc, argv, "d3d12-disable-nvapi"))
        {
            options.d3d12DisableNVAPI = true;
        }

        doctest::parseIntOption(argc, argv, "optix-version=", doctest::option_int, options.optixVersion);

        if (doctest::parseFlag(argc, argv, "memory-report"))
        {
            options.memoryReport = true;
            options.printMemoryReport = true;
        }

        doctest::String memoryReportFile;
        if (doctest::parseOption(argc, argv, "memory-report-file=", &memoryReportFile))
        {
            options.memoryReport = true;
            options.memoryReportFile = memoryReportFile.c_str();
        }
    }

    int result = 1;
    {
        doctest::Context context(argc, argv);

        context.setOption("--reporters", "custom");
        context.setOption("--order-by", "name");

        // Select specific test suite to run
        // context.setOption("-tc", "shader-cache-*");
        // Report successful tests
        // context.setOption("success", true);

        if (context.shouldExit() || rhi::testing::options().listDevices || rhi::testing::checkRequiredDevices())
            result = context.run();

        bool noSilentSkips = rhi::testing::checkNoSilentGpuSkips();
        if (result == 0 && !noSilentSkips)
            result = 1;

        rhi::testing::releaseCachedDevices();
        rhi::testing::sampleMemoryReport("after-release-cached-devices");
    }

    rhi::testing::cleanupTestTempDirectories();

    rhi::destroyRHI();

    rhi::testing::sampleMemoryReport("after-destroy-rhi");
    rhi::testing::printMemoryReport();
    rhi::testing::writeMemoryReport();

#if SLANG_RHI_ENABLE_REF_OBJECT_TRACKING
    if (!rhi::RefObjectTracker::instance().objects.empty())
    {
        std::cerr << std::to_string(rhi::RefObjectTracker::instance().objects.size()) << " leaked objects detected!"
                  << std::endl;
        std::cerr << "Leaked objects detected!" << std::endl;
        for (auto obj : rhi::RefObjectTracker::instance().objects)
        {
            std::cerr << "Leaked object: " << obj << std::endl;
        }
        return 1;
    }
#endif

#if SLANG_RHI_DEBUG
    if (rhi::RefObject::getObjectCount() > 0)
    {
        std::cerr << std::to_string(rhi::RefObject::getObjectCount()) << " leaked objects detected!" << std::endl;
        return 1;
    }
#endif

    return result;
}
