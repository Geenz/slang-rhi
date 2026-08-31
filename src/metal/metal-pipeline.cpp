#include "metal-pipeline.h"
#include "metal-device.h"
#include "metal-shader-object-layout.h"
#include "metal-shader-program.h"
#include "metal-utils.h"
#include "metal-input-layout.h"

namespace rhi::metal {

RenderPipelineImpl::RenderPipelineImpl(Device* device, const RenderPipelineDesc& desc)
    : RenderPipeline(device, desc)
{
}

Result RenderPipelineImpl::getNativeHandle(NativeHandle* outHandle)
{
    outHandle->type = NativeHandleType::MTLRenderPipelineState;
    outHandle->value = (uint64_t)m_pipelineState.get();
    return SLANG_OK;
}

MTL::RenderPipelineState* RenderPipelineImpl::getOrCreateIcbPipelineState(DeviceImpl* device)
{
    // Mesh pipelines never have an ICB variant (descriptor left null).
    if (!m_icbPipelineDesc)
        return nullptr;

    // Build the ICB-enabled twin exactly once, on the first indirect draw that
    // needs it. std::call_once serializes concurrent recorders; a null result
    // (ICB-incompatible shader) is cached like the old eager path — callers then
    // fall back to a CPU draw loop, and we don't retry the failed compile.
    std::call_once(
        m_icbPipelineOnce,
        [&]()
        {
            NS::Error* icbError = nullptr;
            m_icbPipelineState =
                NS::TransferPtr(device->m_device->newRenderPipelineState(m_icbPipelineDesc.get(), &icbError));
            if (!m_icbPipelineState && icbError)
            {
                device->handleMessage(
                    DebugMessageType::Warning,
                    DebugMessageSource::Driver,
                    icbError->localizedDescription()->utf8String()
                );
            }
        }
    );

    return m_icbPipelineState.get();
}

Result DeviceImpl::createRenderPipeline2(const RenderPipelineDesc& desc, IRenderPipeline** outPipeline)
{
    AUTORELEASEPOOL

    TimePoint startTime = Timer::now();

    ShaderProgramImpl* program = checked_cast<ShaderProgramImpl*>(desc.program);
    InputLayoutImpl* inputLayout = checked_cast<InputLayoutImpl*>(desc.inputLayout);
    if (!program)
        return SLANG_FAIL;
    SLANG_RHI_ASSERT(!program->m_modules.empty());

    bool isMeshPipeline = program->isMeshShaderProgram();
    NS::SharedPtr<MTL::RenderPipelineState> pipelineState;
    // Descriptor for the ICB-enabled variant, kept for a lazy first-use build.
    NS::SharedPtr<MTL::RenderPipelineDescriptor> icbPipelineDesc;
    NS::UInteger vertexBufferOffset = 0;
    MTL::Size objectThreadgroupSize = MTL::Size::Make(1, 1, 1);
    MTL::Size meshThreadgroupSize = MTL::Size::Make(1, 1, 1);

    // Lambda to configure color/depth/stencil attachments on either pipeline descriptor type.
    auto configureRenderTargets = [&](auto* pd)
    {
        pd->setAlphaToCoverageEnabled(desc.multisample.alphaToCoverageEnable);

        for (uint32_t i = 0; i < desc.targetCount; ++i)
        {
            const ColorTargetDesc& targetState = desc.targets[i];
            MTL::RenderPipelineColorAttachmentDescriptor* colorAttachment = pd->colorAttachments()->object(i);
            colorAttachment->setPixelFormat(translatePixelFormat(targetState.format));

            colorAttachment->setBlendingEnabled(targetState.enableBlend);
            colorAttachment->setSourceRGBBlendFactor(translateBlendFactor(targetState.color.srcFactor));
            colorAttachment->setDestinationRGBBlendFactor(translateBlendFactor(targetState.color.dstFactor));
            colorAttachment->setRgbBlendOperation(translateBlendOperation(targetState.color.op));
            colorAttachment->setSourceAlphaBlendFactor(translateBlendFactor(targetState.alpha.srcFactor));
            colorAttachment->setDestinationAlphaBlendFactor(translateBlendFactor(targetState.alpha.dstFactor));
            colorAttachment->setAlphaBlendOperation(translateBlendOperation(targetState.alpha.op));
            colorAttachment->setWriteMask(translateColorWriteMask(targetState.writeMask));
        }
        if (desc.depthStencil.format != Format::Undefined)
        {
            const DepthStencilDesc& depthStencil = desc.depthStencil;
            MTL::PixelFormat pixelFormat = translatePixelFormat(depthStencil.format);
            if (isDepthFormat(pixelFormat))
            {
                pd->setDepthAttachmentPixelFormat(translatePixelFormat(depthStencil.format));
            }
            if (isStencilFormat(pixelFormat))
            {
                pd->setStencilAttachmentPixelFormat(translatePixelFormat(depthStencil.format));
            }
        }

        pd->setRasterSampleCount(desc.multisample.sampleCount);

        if (desc.label)
        {
            pd->setLabel(createString(desc.label).get());
        }
    };

    if (isMeshPipeline)
    {
        // Mesh shader pipeline path — uses MeshRenderPipelineDescriptor
        NS::SharedPtr<MTL::MeshRenderPipelineDescriptor> meshPd =
            NS::TransferPtr(MTL::MeshRenderPipelineDescriptor::alloc()->init());

        for (const ShaderProgramImpl::Module& module : program->m_modules)
        {
            auto functionName = createString(module.entryPointName.data());
            NS::SharedPtr<MTL::Function> function =
                NS::TransferPtr(module.library->newFunction(functionName.get()));
            if (!function)
                return SLANG_FAIL;

            switch (module.stage)
            {
            case SLANG_STAGE_MESH:
                meshPd->setMeshFunction(function.get());
                break;
            case SLANG_STAGE_AMPLIFICATION:
                meshPd->setObjectFunction(function.get());
                break;
            case SLANG_STAGE_FRAGMENT:
                meshPd->setFragmentFunction(function.get());
                break;
            default:
                return SLANG_FAIL;
            }
        }

        configureRenderTargets(meshPd.get());

        // Extract threadgroup sizes from Slang reflection.
        auto programReflection = program->linkedProgram->getLayout();
        for (SlangUInt i = 0; i < programReflection->getEntryPointCount(); ++i)
        {
            auto entryPoint = programReflection->getEntryPointByIndex(i);
            SlangStage stage = entryPoint->getStage();
            if (stage == SLANG_STAGE_MESH)
            {
                SlangUInt threadGroupSize[3];
                entryPoint->getComputeThreadGroupSize(3, threadGroupSize);
                meshThreadgroupSize = MTL::Size::Make(threadGroupSize[0], threadGroupSize[1], threadGroupSize[2]);
            }
            else if (stage == SLANG_STAGE_AMPLIFICATION)
            {
                SlangUInt threadGroupSize[3];
                entryPoint->getComputeThreadGroupSize(3, threadGroupSize);
                objectThreadgroupSize = MTL::Size::Make(threadGroupSize[0], threadGroupSize[1], threadGroupSize[2]);
            }
        }

        NS::Error* error = nullptr;
        pipelineState = NS::TransferPtr(
            m_device->newRenderPipelineState(meshPd.get(), MTL::PipelineOptionNone, nullptr, &error));
        if (!pipelineState)
        {
            if (error)
            {
                handleMessage(
                    DebugMessageType::Error,
                    DebugMessageSource::Driver,
                    error->localizedDescription()->utf8String()
                );
            }
            return SLANG_FAIL;
        }
        // No ICB variant for mesh pipelines
    }
    else
    {
        // Traditional vertex/fragment pipeline path
        NS::SharedPtr<MTL::RenderPipelineDescriptor> pd =
            NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());

        for (const ShaderProgramImpl::Module& module : program->m_modules)
        {
            auto functionName = createString(module.entryPointName.data());
            NS::SharedPtr<MTL::Function> function =
                NS::TransferPtr(module.library->newFunction(functionName.get()));
            if (!function)
                return SLANG_FAIL;

            switch (module.stage)
            {
            case SLANG_STAGE_VERTEX:
                pd->setVertexFunction(function.get());
                break;
            case SLANG_STAGE_FRAGMENT:
                pd->setFragmentFunction(function.get());
                break;
            default:
                return SLANG_FAIL;
            }
        }

        // Create a vertex descriptor with the vertex buffer binding indices being offset.
        // They need to be in a range not used by any buffers in the root object layout.
        // The +1 is to account for a potential constant buffer at index 0.
        vertexBufferOffset = program->m_rootObjectLayout->getTotalBufferCount() + 1;
        if (inputLayout)
        {
            NS::SharedPtr<MTL::VertexDescriptor> vertexDescriptor;
            vertexDescriptor = inputLayout->createVertexDescriptor(vertexBufferOffset);
            pd->setVertexDescriptor(vertexDescriptor.get());
        }
        pd->setInputPrimitiveTopology(translatePrimitiveTopologyClass(desc.primitiveTopology));

        configureRenderTargets(pd.get());

        // Create default pipeline WITHOUT ICB support.
        NS::Error* error;
        pipelineState = NS::TransferPtr(m_device->newRenderPipelineState(pd.get(), &error));
        if (!pipelineState)
        {
            if (error)
            {
                handleMessage(
                    DebugMessageType::Error,
                    DebugMessageSource::Driver,
                    error->localizedDescription()->utf8String()
                );
            }
            return SLANG_FAIL;
        }

        // Defer the ICB-enabled variant: retain the descriptor (with ICB support
        // flagged on) and build the second PSO lazily on the first indirect draw
        // that actually needs it (see RenderPipelineImpl::getOrCreateIcbPipelineState).
        // Traditional pipelines that never issue an indirect draw — the common case,
        // e.g. fullscreen/post passes — otherwise pay a wasted second PSO link here.
        // The descriptor is not mutated again after this, so the lazy build reads it
        // without further locking.
        pd->setSupportIndirectCommandBuffers(true);
        icbPipelineDesc = pd;
    }

    // Create depth stencil state
    auto createStencilDesc = [](const DepthStencilOpDesc& desc,
                                uint32_t readMask,
                                uint32_t writeMask) -> NS::SharedPtr<MTL::StencilDescriptor>
    {
        NS::SharedPtr<MTL::StencilDescriptor> stencilDesc = NS::TransferPtr(MTL::StencilDescriptor::alloc()->init());
        stencilDesc->setStencilCompareFunction(translateCompareFunction(desc.stencilFunc));
        stencilDesc->setStencilFailureOperation(translateStencilOperation(desc.stencilFailOp));
        stencilDesc->setDepthFailureOperation(translateStencilOperation(desc.stencilDepthFailOp));
        stencilDesc->setDepthStencilPassOperation(translateStencilOperation(desc.stencilPassOp));
        stencilDesc->setReadMask(readMask);
        stencilDesc->setWriteMask(writeMask);
        return stencilDesc;
    };

    const auto& depthStencil = desc.depthStencil;
    NS::SharedPtr<MTL::DepthStencilDescriptor> depthStencilDesc =
        NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
    if (depthStencil.depthTestEnable)
    {
        depthStencilDesc->setDepthCompareFunction(translateCompareFunction(depthStencil.depthFunc));
    }
    depthStencilDesc->setDepthWriteEnabled(depthStencil.depthWriteEnable);
    if (depthStencil.stencilEnable)
    {
        depthStencilDesc->setFrontFaceStencil(
            createStencilDesc(depthStencil.frontFace, depthStencil.stencilReadMask, depthStencil.stencilWriteMask).get()
        );
        depthStencilDesc->setBackFaceStencil(
            createStencilDesc(depthStencil.backFace, depthStencil.stencilReadMask, depthStencil.stencilWriteMask).get()
        );
    }
    NS::SharedPtr<MTL::DepthStencilState> depthStencilState =
        NS::TransferPtr(m_device->newDepthStencilState(depthStencilDesc.get()));
    if (!depthStencilState)
    {
        return SLANG_FAIL;
    }

    // Report the pipeline creation time.
    if (m_shaderCompilationReporter)
    {
        m_shaderCompilationReporter->reportCreatePipeline(
            program,
            ShaderCompilationReporter::PipelineType::Render,
            startTime,
            Timer::now(),
            false,
            0,
            nullptr
        );
    }

    RefPtr<RenderPipelineImpl> pipeline = new RenderPipelineImpl(this, desc);
    pipeline->m_program = program;
    pipeline->m_rootObjectLayout = program->m_rootObjectLayout;
    pipeline->m_pipelineState = pipelineState;
    pipeline->m_icbPipelineDesc = icbPipelineDesc; // null for mesh pipelines; ICB PSO built lazily
    pipeline->m_depthStencilState = depthStencilState;
    pipeline->m_primitiveType = translatePrimitiveType(desc.primitiveTopology);
    pipeline->m_rasterizerDesc = desc.rasterizer;
    pipeline->m_vertexBufferOffset = vertexBufferOffset;
    pipeline->m_isMeshPipeline = isMeshPipeline;
    pipeline->m_objectThreadgroupSize = objectThreadgroupSize;
    pipeline->m_meshThreadgroupSize = meshThreadgroupSize;
    returnComPtr(outPipeline, pipeline);
    return SLANG_OK;
}

ComputePipelineImpl::ComputePipelineImpl(Device* device, const ComputePipelineDesc& desc)
    : ComputePipeline(device, desc)
{
}

Result ComputePipelineImpl::getNativeHandle(NativeHandle* outHandle)
{
    outHandle->type = NativeHandleType::MTLComputePipelineState;
    outHandle->value = (uint64_t)m_pipelineState.get();
    return SLANG_OK;
}

Result DeviceImpl::createComputePipeline2(const ComputePipelineDesc& desc, IComputePipeline** outPipeline)
{
    AUTORELEASEPOOL

    TimePoint startTime = Timer::now();

    ShaderProgramImpl* program = checked_cast<ShaderProgramImpl*>(desc.program);
    SLANG_RHI_ASSERT(!program->m_modules.empty());

    const ShaderProgramImpl::Module& module = program->m_modules[0];
    auto functionName = createString(module.entryPointName.data());
    NS::SharedPtr<MTL::Function> function = NS::TransferPtr(module.library->newFunction(functionName.get()));
    if (!function)
        return SLANG_FAIL;

    NS::SharedPtr<MTL::ComputePipelineDescriptor> pd = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());

    pd->setComputeFunction(function.get());

    if (desc.label)
    {
        pd->setLabel(createString(desc.label).get());
    }

    NS::Error* error;
    NS::SharedPtr<MTL::ComputePipelineState> pipelineState =
        NS::TransferPtr(m_device->newComputePipelineState(pd.get(), MTL::PipelineOptionNone, nullptr, &error));
    if (!pipelineState)
    {
        if (error)
        {
            handleMessage(
                DebugMessageType::Error,
                DebugMessageSource::Driver,
                error->localizedDescription()->utf8String()
            );
        }
        return SLANG_FAIL;
    }

    // Query thread group size for use during dispatch.
    SlangUInt threadGroupSize[3];
    program->linkedProgram->getLayout()->getEntryPointByIndex(0)->getComputeThreadGroupSize(3, threadGroupSize);

    // Report the pipeline creation time.
    if (m_shaderCompilationReporter)
    {
        m_shaderCompilationReporter->reportCreatePipeline(
            program,
            ShaderCompilationReporter::PipelineType::Compute,
            startTime,
            Timer::now(),
            false,
            0,
            nullptr
        );
    }

    RefPtr<ComputePipelineImpl> pipeline = new ComputePipelineImpl(this, desc);
    pipeline->m_program = program;
    pipeline->m_rootObjectLayout = program->m_rootObjectLayout;
    pipeline->m_pipelineState = pipelineState;
    pipeline->m_threadGroupSize = MTL::Size(threadGroupSize[0], threadGroupSize[1], threadGroupSize[2]);
    returnComPtr(outPipeline, pipeline);
    return SLANG_OK;
}

RayTracingPipelineImpl::RayTracingPipelineImpl(Device* device, const RayTracingPipelineDesc& desc)
    : RayTracingPipeline(device, desc)
{
}

Result RayTracingPipelineImpl::getNativeHandle(NativeHandle* outHandle)
{
    *outHandle = {};
    return SLANG_E_NOT_IMPLEMENTED;
}

Result DeviceImpl::createRayTracingPipeline2(const RayTracingPipelineDesc& desc, IRayTracingPipeline** outPipeline)
{
    AUTORELEASEPOOL

    return SLANG_E_NOT_IMPLEMENTED;
}

} // namespace rhi::metal
