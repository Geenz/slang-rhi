#pragma once

#include "metal-base.h"

namespace rhi::metal {

class BindlessDescriptorSet : public RefObject
{
public:
    BindlessDescriptorSet(DeviceImpl* device, const BindlessDesc& desc);
    ~BindlessDescriptorSet();

    Result initialize();

    Result allocBufferHandle(
        IBuffer* buffer,
        DescriptorHandleAccess access,
        Format format,
        BufferRange range,
        DescriptorHandle* outHandle
    );
    Result allocTextureHandle(ITextureView* textureView, DescriptorHandleAccess access, DescriptorHandle* outHandle);
    Result allocSamplerHandle(ISampler* sampler, DescriptorHandle* outHandle);
    Result allocAccelerationStructureHandle(IAccelerationStructure* accelerationStructure, DescriptorHandle* outHandle);
    Result freeHandle(const DescriptorHandle& handle);

public:
    DeviceImpl* m_device;
    BindlessDesc m_desc;

    struct SlotAllocator
    {
        uint32_t capacity = 0;
        uint32_t count = 0;
        std::vector<uint32_t> freeSlots;

        Result allocate(uint32_t* outSlot)
        {
            if (!freeSlots.empty())
            {
                *outSlot = freeSlots.back();
                freeSlots.pop_back();
                return SLANG_OK;
            }
            if (count < capacity)
            {
                *outSlot = count++;
                return SLANG_OK;
            }
            return SLANG_E_OUT_OF_MEMORY;
        }

        Result free(uint32_t slot)
        {
            if (slot >= capacity)
            {
                return SLANG_E_INVALID_ARG;
            }
            // Double-free protection
            for (uint32_t s : freeSlots)
            {
                if (s == slot)
                    return SLANG_E_INVALID_ARG;
            }
            freeSlots.push_back(slot);
            return SLANG_OK;
        }
    };

    SlotAllocator m_bufferAllocator;
    SlotAllocator m_textureAllocator;
    SlotAllocator m_samplerAllocator;
    SlotAllocator m_accelerationStructureAllocator;

    // Metal uses gpuResourceID() directly as the handle value —
    // no GPU descriptor heap needed. The slot allocators only track
    // allocation state for freeHandle() and capacity limits.

    // Map from handle value to slot index for freeHandle() lookup
    std::unordered_map<uint64_t, uint32_t> m_handleToTextureSlot;
    std::unordered_map<uint64_t, uint32_t> m_handleToSamplerSlot;
    std::unordered_map<uint64_t, uint32_t> m_handleToBufferSlot;

    // Track allocated Metal resources for useResource calls
    std::vector<MTL::Resource*> m_allocatedTextureResources;
    std::vector<MTL::Resource*> m_allocatedBufferResources;

    // Texture views created for typed buffer handles (Buffer<T>) — kept alive here
    std::vector<NS::SharedPtr<MTL::Texture>> m_typedBufferTextureViews;

    // Map handle value → actual Metal resource pointer (for flat binding path)
    std::unordered_map<uint64_t, MTL::Texture*> m_handleToTexturePtr;
    std::unordered_map<uint64_t, MTL::SamplerState*> m_handleToSamplerPtr;
    struct BufferBinding { MTL::Buffer* buffer; NS::UInteger offset; };
    std::unordered_map<uint64_t, BufferBinding> m_handleToBufferPtr;

    /// Get all texture resources that have descriptor handles allocated.
    /// The command encoder must call useResources() with these before dispatch.
    const std::vector<MTL::Resource*>& allocatedTextureResources() const { return m_allocatedTextureResources; }
    const std::vector<MTL::Resource*>& allocatedBufferResources() const { return m_allocatedBufferResources; }
};

} // namespace rhi::metal
