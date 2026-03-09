#pragma once

#include <slang-rhi.h>
#include <TargetConditionals.h>

#include "metal-api.h"
#include "core/common.h"

// iOS uses StorageModeShared (UMA — no managed memory).
// macOS uses StorageModeManaged for CPU/GPU coherent staging buffers.
#if TARGET_OS_OSX
    #define RHI_MTL_STAGING_STORAGE_MODE MTL::ResourceStorageModeManaged
    #define RHI_MTL_STAGING_TEXTURE_MODE MTL::StorageModeManaged
#else
    #define RHI_MTL_STAGING_STORAGE_MODE MTL::ResourceStorageModeShared
    #define RHI_MTL_STAGING_TEXTURE_MODE MTL::StorageModeShared
#endif

namespace rhi::metal {

// Utility functions for Metal

struct ScopedAutoreleasePool
{
    ScopedAutoreleasePool() { m_pool = NS::AutoreleasePool::alloc()->init(); }
    ~ScopedAutoreleasePool() { m_pool->drain(); }
    NS::AutoreleasePool* m_pool;
};

#define AUTORELEASEPOOL ::rhi::metal::ScopedAutoreleasePool _pool_;

inline NS::SharedPtr<NS::String> createString(const char* str, NS::StringEncoding encoding = NS::UTF8StringEncoding)
{
    NS::SharedPtr<NS::String> nsString = NS::TransferPtr(NS::String::alloc()->init(str, encoding));
    return nsString;
}

inline NS::SharedPtr<NS::String> createStringView(
    void* bytes,
    size_t len,
    NS::StringEncoding encoding = NS::UTF8StringEncoding
)
{
    NS::SharedPtr<NS::String> nsString = NS::TransferPtr(NS::String::alloc()->init(bytes, len, encoding, false));
    return nsString;
}

struct FormatMapping
{
    Format format;
    MTL::PixelFormat pixelFormat;
    MTL::VertexFormat vertexFormat;
    MTL::AttributeFormat attributeFormat;
};

const FormatMapping& getFormatMapping(Format format);

MTL::PixelFormat translatePixelFormat(Format format);
MTL::VertexFormat translateVertexFormat(Format format);
MTL::AttributeFormat translateAttributeFormat(Format format);

bool isDepthFormat(MTL::PixelFormat format);
bool isStencilFormat(MTL::PixelFormat format);

MTL::TextureType translateTextureType(TextureType type);

MTL::SamplerMinMagFilter translateSamplerMinMagFilter(TextureFilteringMode mode);
MTL::SamplerMipFilter translateSamplerMipFilter(TextureFilteringMode mode);
MTL::SamplerAddressMode translateSamplerAddressMode(TextureAddressingMode mode);
MTL::CompareFunction translateCompareFunction(ComparisonFunc func);
MTL::StencilOperation translateStencilOperation(StencilOp op);

MTL::VertexStepFunction translateVertexStepFunction(InputSlotClass slotClass);

MTL::PrimitiveType translatePrimitiveType(PrimitiveTopology topology);
MTL::PrimitiveTopologyClass translatePrimitiveTopologyClass(PrimitiveTopology topology);

MTL::BlendFactor translateBlendFactor(BlendFactor factor);
MTL::BlendOperation translateBlendOperation(BlendOp op);
MTL::ColorWriteMask translateColorWriteMask(RenderTargetWriteMask mask);

MTL::Winding translateWinding(FrontFaceMode mode);
MTL::CullMode translateCullMode(CullMode mode);
MTL::TriangleFillMode translateTriangleFillMode(FillMode mode);

MTL::LoadAction translateLoadOp(LoadOp loadOp);
MTL::StoreAction translateStoreOp(StoreOp storeOp, bool resolve);

} // namespace rhi::metal
