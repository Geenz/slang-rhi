#include "metal-bindless-descriptor-set.h"
#include "metal-buffer.h"
#include "metal-texture.h"
#include "metal-sampler.h"
#include "metal-device.h"
#include "metal-utils.h"

#include <algorithm>

namespace rhi::metal {

BindlessDescriptorSet::BindlessDescriptorSet(DeviceImpl* device, const BindlessDesc& desc)
    : m_device(device)
    , m_desc(desc)
{
}

BindlessDescriptorSet::~BindlessDescriptorSet() = default;

Result BindlessDescriptorSet::initialize()
{
    m_bufferAllocator.capacity = m_desc.bufferCount;
    m_textureAllocator.capacity = m_desc.textureCount;
    m_samplerAllocator.capacity = m_desc.samplerCount;
    m_accelerationStructureAllocator.capacity = m_desc.accelerationStructureCount;
    return SLANG_OK;
}

Result BindlessDescriptorSet::allocBufferHandle(
    IBuffer* buffer,
    DescriptorHandleAccess access,
    Format format,
    BufferRange range,
    DescriptorHandle* outHandle)
{
    if (!buffer || !outHandle)
        return SLANG_E_INVALID_ARG;

    uint32_t slot;
    SLANG_RETURN_ON_FAIL(m_bufferAllocator.allocate(&slot));

    auto* metalBuffer = checked_cast<BufferImpl*>(buffer);

    outHandle->type = (access == DescriptorHandleAccess::Read)
                      ? DescriptorHandleType::Buffer
                      : DescriptorHandleType::RWBuffer;

    if (format != Format::Undefined)
    {
        // Typed buffer (Buffer<T>) — create a texture view from the buffer.
        // Metal's texture_buffer<T> requires a MTL::Texture backed by the buffer.
        MTL::PixelFormat pixelFormat = translatePixelFormat(format);
        const FormatInfo& formatInfo = getFormatInfo(format);

        NS::UInteger bufferOffset = range.offset;
        NS::UInteger bufferSize = (range.size == kEntireBuffer.size)
                                  ? metalBuffer->m_buffer->length() - bufferOffset
                                  : range.size;
        NS::UInteger elementCount = bufferSize / formatInfo.blockSizeInBytes;

        NS::SharedPtr<MTL::TextureDescriptor> texDesc = NS::TransferPtr(MTL::TextureDescriptor::alloc()->init());
        texDesc->setTextureType(MTL::TextureTypeTextureBuffer);
        texDesc->setPixelFormat(pixelFormat);
        texDesc->setWidth(elementCount);
        texDesc->setStorageMode(metalBuffer->m_buffer->storageMode());
        texDesc->setUsage(
            (access == DescriptorHandleAccess::Read)
            ? MTL::TextureUsageShaderRead
            : (MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite)
        );

        NS::SharedPtr<MTL::Texture> texView = NS::TransferPtr(
            metalBuffer->m_buffer->newTexture(texDesc.get(), bufferOffset, formatInfo.blockSizeInBytes * elementCount)
        );
        if (!texView)
        {
            m_bufferAllocator.free(slot);
            return SLANG_FAIL;
        }

        auto resourceId = texView->gpuResourceID();
        memcpy(&outHandle->value, &resourceId, sizeof(uint64_t));

        // Store the texture view pointer for flat binding and useResource
        m_handleToTexturePtr[outHandle->value] = texView.get();
        m_allocatedTextureResources.push_back(texView.get());

        // Keep the texture view alive
        m_typedBufferTextureViews.push_back(std::move(texView));
    }
    else
    {
        // Untyped buffer (StructuredBuffer, ByteAddressBuffer) — use device address directly
        DeviceAddress addr = metalBuffer->getDeviceAddress() + range.offset;
        outHandle->value = static_cast<uint64_t>(addr);

        m_handleToBufferPtr[outHandle->value] = {metalBuffer->m_buffer.get(), static_cast<NS::UInteger>(range.offset)};
        m_allocatedBufferResources.push_back(metalBuffer->m_buffer.get());
    }

    m_handleToBufferSlot[outHandle->value] = slot;
    return SLANG_OK;
}

Result BindlessDescriptorSet::allocTextureHandle(
    ITextureView* textureView,
    DescriptorHandleAccess access,
    DescriptorHandle* outHandle)
{
    if (!textureView || !outHandle)
        return SLANG_E_INVALID_ARG;

    uint32_t slot;
    SLANG_RETURN_ON_FAIL(m_textureAllocator.allocate(&slot));

    auto* metalView = checked_cast<TextureViewImpl*>(textureView);
    auto resourceId = metalView->m_textureView->gpuResourceID();

    outHandle->type = (access == DescriptorHandleAccess::Read)
                      ? DescriptorHandleType::Texture
                      : DescriptorHandleType::RWTexture;
    memcpy(&outHandle->value, &resourceId, sizeof(uint64_t));

    m_handleToTextureSlot[outHandle->value] = slot;
    m_handleToTexturePtr[outHandle->value] = metalView->m_textureView.get();
    m_allocatedTextureResources.push_back(metalView->m_textureView.get());
    return SLANG_OK;
}

Result BindlessDescriptorSet::allocSamplerHandle(ISampler* sampler, DescriptorHandle* outHandle)
{
    if (!sampler || !outHandle)
        return SLANG_E_INVALID_ARG;

    uint32_t slot;
    SLANG_RETURN_ON_FAIL(m_samplerAllocator.allocate(&slot));

    auto* metalSampler = checked_cast<SamplerImpl*>(sampler);
    auto resourceId = metalSampler->m_samplerState->gpuResourceID();

    outHandle->type = DescriptorHandleType::Sampler;
    memcpy(&outHandle->value, &resourceId, sizeof(uint64_t));

    m_handleToSamplerSlot[outHandle->value] = slot;
    m_handleToSamplerPtr[outHandle->value] = metalSampler->m_samplerState.get();
    return SLANG_OK;
}

Result BindlessDescriptorSet::allocAccelerationStructureHandle(
    IAccelerationStructure* accelerationStructure,
    DescriptorHandle* outHandle)
{
    // TODO: implement when acceleration structure support is needed
    return SLANG_E_NOT_AVAILABLE;
}

Result BindlessDescriptorSet::freeHandle(const DescriptorHandle& handle)
{
    switch (handle.type)
    {
    case DescriptorHandleType::Buffer:
    case DescriptorHandleType::RWBuffer:
    {
        auto it = m_handleToBufferSlot.find(handle.value);
        if (it == m_handleToBufferSlot.end())
            return SLANG_E_INVALID_ARG;
        uint32_t slot = it->second;
        m_handleToBufferSlot.erase(it);
        // Clean up pointer map and resource vector
        auto ptrIt = m_handleToBufferPtr.find(handle.value);
        if (ptrIt != m_handleToBufferPtr.end())
        {
            auto vecIt = std::find(
                m_allocatedBufferResources.begin(),
                m_allocatedBufferResources.end(),
                static_cast<MTL::Resource*>(ptrIt->second.buffer)
            );
            if (vecIt != m_allocatedBufferResources.end())
            {
                *vecIt = m_allocatedBufferResources.back();
                m_allocatedBufferResources.pop_back();
            }
            m_handleToBufferPtr.erase(ptrIt);
        }
        return m_bufferAllocator.free(slot);
    }
    case DescriptorHandleType::Texture:
    case DescriptorHandleType::RWTexture:
    {
        auto it = m_handleToTextureSlot.find(handle.value);
        if (it == m_handleToTextureSlot.end())
            return SLANG_E_INVALID_ARG;
        uint32_t slot = it->second;
        m_handleToTextureSlot.erase(it);
        // Clean up pointer map and resource vector
        auto ptrIt = m_handleToTexturePtr.find(handle.value);
        if (ptrIt != m_handleToTexturePtr.end())
        {
            auto vecIt = std::find(
                m_allocatedTextureResources.begin(),
                m_allocatedTextureResources.end(),
                static_cast<MTL::Resource*>(ptrIt->second)
            );
            if (vecIt != m_allocatedTextureResources.end())
            {
                *vecIt = m_allocatedTextureResources.back();
                m_allocatedTextureResources.pop_back();
            }
            m_handleToTexturePtr.erase(ptrIt);
        }
        return m_textureAllocator.free(slot);
    }
    case DescriptorHandleType::Sampler:
    {
        auto it = m_handleToSamplerSlot.find(handle.value);
        if (it == m_handleToSamplerSlot.end())
            return SLANG_E_INVALID_ARG;
        uint32_t slot = it->second;
        m_handleToSamplerSlot.erase(it);
        m_handleToSamplerPtr.erase(handle.value);
        return m_samplerAllocator.free(slot);
    }
    default:
        return SLANG_E_INVALID_ARG;
    }
}

} // namespace rhi::metal
