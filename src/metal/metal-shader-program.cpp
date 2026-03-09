#include "metal-shader-program.h"
#include "metal-device.h"
#include "metal-shader-object-layout.h"
#include "metal-utils.h"

#include <string>

namespace rhi::metal {

ShaderProgramImpl::ShaderProgramImpl(Device* device, const ShaderProgramDesc& desc)
    : ShaderProgram(device, desc)
{
}

ShaderProgramImpl::~ShaderProgramImpl() {}

Result ShaderProgramImpl::createShaderModule(slang::EntryPointReflection* entryPointInfo, ComPtr<ISlangBlob> kernelCode)
{
    DeviceImpl* device = getDevice<DeviceImpl>();

    Module module;
    module.stage = entryPointInfo->getStage();
    module.entryPointName = entryPointInfo->getNameOverride();
    module.code = kernelCode;

    NS::Error* error = nullptr;

#if TARGET_OS_IOS || TARGET_OS_TV || TARGET_OS_VISION
    // On iOS there is no offline Metal compiler — kernelCode is MSL source text.
    // Compile on-device via MTLDevice::newLibraryWithSource.
    // Build a null-terminated string from the blob.
    std::string mslSource(
        static_cast<const char*>(kernelCode->getBufferPointer()),
        kernelCode->getBufferSize()
    );
    NS::String* source = NS::String::string(mslSource.c_str(), NS::UTF8StringEncoding);
    MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
    module.library = NS::TransferPtr(device->m_device->newLibrary(source, options, &error));
    options->release();
#else
    // On macOS, kernelCode is pre-compiled MetalLib binary.
    dispatch_data_t data = dispatch_data_create(
        kernelCode->getBufferPointer(),
        kernelCode->getBufferSize(),
        dispatch_get_main_queue(),
        NULL
    );
    module.library = NS::TransferPtr(device->m_device->newLibrary(data, &error));
#endif

    if (!module.library)
    {
        const char* msg = error->localizedDescription()->utf8String();
        device->handleMessage(DebugMessageType::Error, DebugMessageSource::Driver, msg);
        return SLANG_E_INVALID_ARG;
    }

    m_modules.push_back(module);
    return SLANG_OK;
}

ShaderObjectLayout* ShaderProgramImpl::getRootShaderObjectLayout()
{
    return m_rootObjectLayout;
}

} // namespace rhi::metal
