#pragma once

#include "metal-base.h"

#include <map>
#include <mutex>
#include <string>

namespace rhi::metal {

class RenderPipelineImpl : public RenderPipeline
{
public:
    RefPtr<RootShaderObjectLayoutImpl> m_rootObjectLayout;
    NS::SharedPtr<MTL::RenderPipelineState> m_pipelineState;
    NS::SharedPtr<MTL::RenderPipelineState> m_icbPipelineState;
    NS::SharedPtr<MTL::DepthStencilState> m_depthStencilState;
    MTL::PrimitiveType m_primitiveType;
    RasterizerDesc m_rasterizerDesc;
    NS::UInteger m_vertexBufferOffset;
    bool m_isMeshPipeline = false;
    MTL::Size m_objectThreadgroupSize = MTL::Size::Make(1, 1, 1);
    MTL::Size m_meshThreadgroupSize = MTL::Size::Make(1, 1, 1);

    // Descriptor for the ICB-enabled PSO twin, retained so the variant can be
    // built lazily on first indirect draw rather than eagerly at pipeline
    // creation (traditional pipelines that never issue an indirect draw would
    // otherwise pay a second, wasted PSO link). Null for mesh pipelines (which
    // have no ICB variant). Read-only once createRenderPipeline2 returns, so the
    // lazy build below is safe without further locking on the descriptor.
    NS::SharedPtr<MTL::RenderPipelineDescriptor> m_icbPipelineDesc;
    std::once_flag m_icbPipelineOnce;

    RenderPipelineImpl(Device* device, const RenderPipelineDesc& desc);

    // Lazily build (exactly once) and return the ICB-enabled PSO variant. Returns
    // nullptr for mesh pipelines and for traditional pipelines whose shader is
    // ICB-incompatible (the compile failed) — callers fall back to a CPU draw
    // loop. Thread-safe: command recording may reach one pipeline from several
    // worker threads; std::call_once serializes the one-time compile.
    MTL::RenderPipelineState* getOrCreateIcbPipelineState(DeviceImpl* device);

    // IRenderPipeline implementation
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(NativeHandle* outHandle) override;
};

class ComputePipelineImpl : public ComputePipeline
{
public:
    RefPtr<RootShaderObjectLayoutImpl> m_rootObjectLayout;
    NS::SharedPtr<MTL::ComputePipelineState> m_pipelineState;
    MTL::Size m_threadGroupSize;

    ComputePipelineImpl(Device* device, const ComputePipelineDesc& desc);

    // IComputePipeline implementation
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(NativeHandle* outHandle) override;
};

class RayTracingPipelineImpl : public RayTracingPipeline
{
public:
    RayTracingPipelineImpl(Device* device, const RayTracingPipelineDesc& desc);

    // IRayTracingPipeline implementation
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(NativeHandle* outHandle) override;
};

} // namespace rhi::metal
