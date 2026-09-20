#include "vk-shader-object.h"
#include "vk-device.h"
#include "vk-buffer.h"
#include "vk-texture.h"
#include "vk-sampler.h"
#include "vk-acceleration-structure.h"
#include "vk-shader-object-layout.h"
#include "vk-bindless-descriptor-set.h"

#include "../state-tracking.h"

#include <string>

namespace rhi::vk {

inline uint64_t hashCombine(uint64_t hash, uint64_t value)
{
    return hash ^ (value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2));
}

/// Fingerprint of everything an object contributes to a descriptor set, excluding its own ordinary
/// data. Sub-objects are folded in by identity and composite version instead.
static uint64_t hashBindingState(const ShaderObject* object)
{
    uint64_t hash = object->m_slots.size();
    for (const ResourceSlot& slot : object->m_slots)
    {
        hash = hashCombine(hash, uint64_t(slot.type));
        hash = hashCombine(hash, uint64_t(uintptr_t(slot.resource.get())));
        hash = hashCombine(hash, uint64_t(uintptr_t(slot.resource2.get())));
        hash = hashCombine(hash, uint64_t(slot.format));
        hash = hashCombine(hash, uint64_t(slot.bufferRange.offset));
        hash = hashCombine(hash, uint64_t(slot.bufferRange.size));
    }
    for (const RefPtr<ShaderObject>& subObject : object->m_objects)
    {
        hash = hashCombine(hash, uint64_t(uintptr_t(subObject.get())));
        hash = hashCombine(hash, subObject ? subObject->getCompositeVersion() : 0);
    }
    return hash;
}

/// Excludes the root's and the entry points' ordinary data: the former reaches the shader through a
/// dynamic offset and the latter through push constants, both of which are rebuilt on every draw.
static uint64_t computeRootBindingKey(RootShaderObject* rootObject)
{
    uint64_t hash = hashBindingState(rootObject);
    for (const RefPtr<ShaderObject>& entryPoint : rootObject->m_entryPoints)
    {
        hash = hashCombine(hash, uint64_t(uintptr_t(entryPoint.get())));
        if (entryPoint)
            hash = hashCombine(hash, hashBindingState(entryPoint));
    }
    return hash;
}

inline void writeDescriptor(DeviceImpl* device, const VkWriteDescriptorSet& write)
{
    device->m_api.vkUpdateDescriptorSets(device->m_device, 1, &write, 0, nullptr);
}

inline void writePlainBufferDescriptor(
    DeviceImpl* device,
    VkDescriptorSet descriptorSet,
    uint32_t binding,
    uint32_t index,
    VkDescriptorType descriptorType,
    BufferImpl* buffer,
    BufferRange range
)
{
    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.range = VK_WHOLE_SIZE;

    if (buffer)
    {
        bufferInfo.buffer = buffer->m_buffer.m_buffer;
        bufferInfo.offset = range.offset;
        bufferInfo.range = range.size;
    }

    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = binding;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = descriptorType;
    write.pBufferInfo = &bufferInfo;

    writeDescriptor(device, write);
}

/// Write a whole binding range in one call: `count` descriptors starting at `binding`, array element 0.
inline void writeImageDescriptors(
    DeviceImpl* device,
    VkDescriptorSet descriptorSet,
    uint32_t binding,
    VkDescriptorType descriptorType,
    const VkDescriptorImageInfo* imageInfos,
    uint32_t count
)
{
    if (count == 0)
        return;

    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = count;
    write.descriptorType = descriptorType;
    write.pImageInfo = imageInfos;

    writeDescriptor(device, write);
}

inline void writeBufferDescriptors(
    DeviceImpl* device,
    VkDescriptorSet descriptorSet,
    uint32_t binding,
    VkDescriptorType descriptorType,
    const VkDescriptorBufferInfo* bufferInfos,
    uint32_t count
)
{
    if (count == 0)
        return;

    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = count;
    write.descriptorType = descriptorType;
    write.pBufferInfo = bufferInfos;

    writeDescriptor(device, write);
}

inline void writeTexelBufferDescriptors(
    DeviceImpl* device,
    VkDescriptorSet descriptorSet,
    uint32_t binding,
    VkDescriptorType descriptorType,
    const VkBufferView* bufferViews,
    uint32_t count
)
{
    if (count == 0)
        return;

    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = count;
    write.descriptorType = descriptorType;
    write.pTexelBufferView = bufferViews;

    writeDescriptor(device, write);
}

inline void writeAccelerationStructureDescriptors(
    DeviceImpl* device,
    VkDescriptorSet descriptorSet,
    uint32_t binding,
    const VkAccelerationStructureKHR* accelerationStructures,
    uint32_t count
)
{
    if (count == 0)
        return;

    VkWriteDescriptorSetAccelerationStructureKHR writeAS = {};
    writeAS.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    writeAS.accelerationStructureCount = count;
    writeAS.pAccelerationStructures = accelerationStructures;

    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = count;
    write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    write.pNext = &writeAS;
    writeDescriptor(device, write);
}

inline void writeBufferState(BindingDataBuilder* builder, BufferImpl* buffer, ResourceState state)
{
    BindingDataImpl* bindingData = builder->m_bindingData;
    if (bindingData->bufferStateCount >= bindingData->bufferStateCapacity)
    {
        bindingData->bufferStateCapacity *= 2;
        BindingDataImpl::BufferState* newBufferStates =
            builder->m_allocator->allocate<BindingDataImpl::BufferState>(bindingData->bufferStateCapacity);
        std::memcpy(
            newBufferStates,
            bindingData->bufferStates,
            bindingData->bufferStateCount * sizeof(BindingDataImpl::BufferState)
        );
        bindingData->bufferStates = newBufferStates;
    }
    bindingData->bufferStates[bindingData->bufferStateCount++] = {buffer, state};
}

inline void writeTextureState(BindingDataBuilder* builder, TextureViewImpl* textureView, ResourceState state)
{
    BindingDataImpl* bindingData = builder->m_bindingData;
    if (bindingData->textureStateCount >= bindingData->textureStateCapacity)
    {
        bindingData->textureStateCapacity *= 2;
        BindingDataImpl::TextureState* newTextureStates =
            builder->m_allocator->allocate<BindingDataImpl::TextureState>(bindingData->textureStateCapacity);
        std::memcpy(
            newTextureStates,
            bindingData->textureStates,
            bindingData->textureStateCount * sizeof(BindingDataImpl::TextureState)
        );
        bindingData->textureStates = newTextureStates;
    }
    bindingData->textureStates[bindingData->textureStateCount++] = {textureView, state};
}


Result BindingDataBuilder::bindAsRoot(
    RootShaderObject* shaderObject,
    RootShaderObjectLayoutImpl* specializedLayout,
    BindingDataImpl*& outBindingData
)
{
    m_bindingData = m_allocator->allocate<BindingDataImpl>();
    m_bindingCache->bindingData.push_back(m_bindingData);

    m_bindingData->pipelineLayout = specializedLayout->m_pipelineLayout;
    m_bindingData->dynamicOffsetCount = 0;

    m_pushConstantRanges = specializedLayout->getAllPushConstantRanges();

    m_bindingData->pushConstantRanges = m_allocator->allocate<VkPushConstantRange>(m_pushConstantRanges.size());
    m_bindingData->pushConstantData = m_allocator->allocate<void*>(m_pushConstantRanges.size());
    m_bindingData->pushConstantCount = 0;
    m_pushConstantSources.clear();

    // Allocate entry point data storage for ray tracing SBT.
    size_t entryPointCount = specializedLayout->m_entryPoints.size();
    m_bindingData->entryPointCount = (uint32_t)entryPointCount;
    if (specializedLayout->findEntryPointIndex(VK_SHADER_STAGE_RAYGEN_BIT_KHR) != -1)
    {
        m_bindingData->entryPointData = m_allocator->allocate<BindingDataImpl::EntryPointData>(entryPointCount);
        for (size_t i = 0; i < entryPointCount; ++i)
        {
            m_bindingData->entryPointData[i].data = nullptr;
            m_bindingData->entryPointData[i].size = 0;
        }
    }
    else
    {
        m_bindingData->entryPointData = nullptr;
    }

    // Uploaded before the cache lookup: a dynamic descriptor leaves the rest of the set unchanged,
    // so only the offset varies between draws.
    const uint32_t ordinaryDataSize = specializedLayout->getTotalOrdinaryDataSize();
    const bool dynamicOrdinaryData = specializedLayout->m_ordinaryDataBufferIsDynamic;
    TransientBufferArena::Allocation ordinaryData = {};
    if (ordinaryDataSize != 0 && dynamicOrdinaryData)
    {
        SLANG_RETURN_ON_FAIL(m_constantBufferArena->allocate(ordinaryDataSize, &ordinaryData));
        SLANG_RETURN_ON_FAIL(
            shaderObject->writeOrdinaryData(ordinaryData.mappedData, ordinaryDataSize, specializedLayout)
        );
        m_bindingData->dynamicOffsets[0] = uint32_t(ordinaryData.offset);
        m_bindingData->dynamicOffsetCount = 1;
    }

    BufferImpl* ordinaryDataBuffer = checked_cast<BufferImpl*>(ordinaryData.buffer);
    VkBuffer ordinaryDataVkBuffer = ordinaryDataBuffer ? ordinaryDataBuffer->m_buffer.m_buffer : VK_NULL_HANDLE;

    // A static ordinary-data descriptor moves on every re-upload, and ray tracing entry point data
    // lives outside the descriptor sets; neither can be replayed.
    const bool cacheable = dynamicOrdinaryData && !m_bindingData->entryPointData;
    const uint64_t rootKey = cacheable ? computeRootBindingKey(shaderObject) : 0;
    if (cacheable && reuseRootBinding(shaderObject, specializedLayout, rootKey, ordinaryDataVkBuffer))
    {
        outBindingData = m_bindingData;
        return SLANG_OK;
    }

    // TODO(shaderobject): we should count number of buffers/textures in the layout and allocate appropriately
    // For now we use a fixed starting capacity and grow as needed.
    m_bindingData->bufferStateCapacity = 1024;
    m_bindingData->bufferStates =
        m_allocator->allocate<BindingDataImpl::BufferState>(m_bindingData->bufferStateCapacity);
    m_bindingData->bufferStateCount = 0;
    m_bindingData->textureStateCapacity = 1024;
    m_bindingData->textureStates =
        m_allocator->allocate<BindingDataImpl::TextureState>(m_bindingData->textureStateCapacity);
    m_bindingData->textureStateCount = 0;

    uint32_t totalDescriptorSetCount = specializedLayout->getTotalDescriptorSetCount();
    if (m_device->m_bindlessDescriptorSet)
    {
        // The bindless descriptor set is always the last descriptor set in the pipeline layout.
        // We need to add one more descriptor set to the count to account for it.
        totalDescriptorSetCount++;
    }
    m_bindingData->descriptorSets = m_allocator->allocate<VkDescriptorSet>(totalDescriptorSetCount);
    m_bindingData->descriptorSetCount = 0;

    BindingOffset offset = {};

    // Note: the operations here are quite similar to what `bindAsParameterBlock` does.
    // The key difference in practice is that we do *not* make use of the adjustment
    // that `bindOrdinaryDataBufferIfNeeded` applied to the offset passed into it.
    //
    // The reason for this difference in behavior is that the layout information
    // for root shader parameters is in practice *already* offset appropriately
    // (so that it ends up using absolute offsets).
    //
    // TODO: One more wrinkle here is that the `ordinaryDataBufferOffset` below
    // might not be correct if `binding=0,set=0` was already claimed via explicit
    // binding information. We should really be getting the offset information for
    // the ordinary data buffer directly from the reflection information for
    // the global scope.

    SLANG_RETURN_ON_FAIL(allocateDescriptorSets(shaderObject, offset, specializedLayout));

    if (ordinaryDataSize != 0 && dynamicOrdinaryData)
    {
        // Base offset zero: the allocation is reached by the dynamic offset instead, so the same
        // descriptor stays valid for every later allocation on this page.
        writePlainBufferDescriptor(
            m_device,
            m_bindingData->descriptorSets[offset.bindingSet],
            offset.binding,
            0,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            ordinaryDataBuffer,
            {0, ordinaryDataSize}
        );
    }
    else
    {
        BindingOffset ordinaryDataBufferOffset = offset;
        SLANG_RETURN_ON_FAIL(bindOrdinaryDataBufferIfNeeded(shaderObject, ordinaryDataBufferOffset, specializedLayout));
    }

    SLANG_RETURN_ON_FAIL(bindAsValue(shaderObject, offset, specializedLayout));

    for (size_t i = 0; i < entryPointCount; ++i)
    {
        auto entryPoint = shaderObject->m_entryPoints[i];
        const auto& entryPointInfo = specializedLayout->m_entryPoints[i];
        EntryPointLayout* entryPointLayout = entryPointInfo.layout;

        // Note: we do *not* need to add the entry point offset
        // information to the global `offset` because the
        // `RootShaderObjectLayout` has already baked any offsets
        // from the global layout into the `entryPointInfo`.

        SLANG_RETURN_ON_FAIL(bindAsEntryPoint(entryPoint, entryPointInfo.offset, entryPointLayout, (uint32_t)i));
    }

    // Assign bindless descriptor set to the last slot if available.
    if (m_device->m_bindlessDescriptorSet)
    {
        m_bindingData->descriptorSets[m_bindingData->descriptorSetCount++] =
            m_device->m_bindlessDescriptorSet->m_descriptorSet;
    }

    if (cacheable)
    {
        storeRootBinding(shaderObject, specializedLayout, rootKey, ordinaryDataVkBuffer);
    }

    outBindingData = m_bindingData;

    return SLANG_OK;
}

Result BindingDataBuilder::bindAsEntryPoint(
    ShaderObject* shaderObject,
    const BindingOffset& inOffset,
    EntryPointLayout* layout,
    uint32_t entryPointIndex
)
{
    if (layout->getSlangLayout()->getStage() != SLANG_STAGE_RAY_GENERATION)
    {
        // For non-raygen entry points, ordinary data goes into push constants.
        return bindAsPushConstantBuffer(shaderObject, inOffset, layout);
    }

    // For raygen entry points, ordinary data is stored in the SBT instead.
    if (shaderObject->m_data.size())
    {
        SLANG_RHI_ASSERT(m_bindingData->entryPointData && entryPointIndex < m_bindingData->entryPointCount);
        BindingDataImpl::EntryPointData& epData = m_bindingData->entryPointData[entryPointIndex];
        epData.size = shaderObject->m_data.size();
        epData.data = m_allocator->allocate(epData.size);
        ::memcpy(epData.data, shaderObject->m_data.data(), epData.size);
    }

    SLANG_RETURN_ON_FAIL(bindAsValue(shaderObject, inOffset, layout));

    return SLANG_OK;
}

Result BindingDataBuilder::bindAsPushConstantBuffer(
    ShaderObject* shaderObject,
    const BindingOffset& inOffset,
    ShaderObjectLayoutImpl* specializedLayout
)
{
    BindingOffset offset = inOffset;

    if (shaderObject->m_data.size())
    {
        // The offset identifies a range in the flattened pipeline layout. Increment it
        // before recursing so any push constants nested in this object's contents use
        // the following ranges.
        const auto pushConstantRangeIndex = offset.pushConstantRange++;
        SLANG_RHI_ASSERT(pushConstantRangeIndex < m_pushConstantRanges.size());
        const auto& pushConstantRange = m_pushConstantRanges[pushConstantRangeIndex];
        SLANG_RHI_ASSERT(pushConstantRange.size == shaderObject->m_data.size());

        const uint32_t index = m_bindingData->pushConstantCount++;
        SLANG_RHI_ASSERT(index < m_pushConstantRanges.size());
        m_bindingData->pushConstantRanges[index] = pushConstantRange;
        m_bindingData->pushConstantData[index] = m_allocator->allocate(pushConstantRange.size);
        SLANG_RETURN_ON_FAIL(shaderObject->writeOrdinaryData(
            m_bindingData->pushConstantData[index],
            pushConstantRange.size,
            specializedLayout
        ));
        m_pushConstantSources.push_back({pushConstantRange, shaderObject, specializedLayout});
    }

    // Resources and nested parameter blocks in the push-constant element type still
    // need their normal recursive binding treatment.
    SLANG_RETURN_ON_FAIL(bindAsValue(shaderObject, offset, specializedLayout));

    return SLANG_OK;
}


Result BindingDataBuilder::bindOrdinaryDataBufferIfNeeded(
    ShaderObject* shaderObject,
    BindingOffset& ioOffset,
    ShaderObjectLayoutImpl* specializedLayout
)
{
    uint32_t size = specializedLayout->getTotalOrdinaryDataSize();
    if (size == 0)
    {
        return SLANG_OK;
    }

    TransientBufferArena::Allocation allocation;
    SLANG_RETURN_ON_FAIL(m_constantBufferArena->allocate(size, &allocation));
    SLANG_RETURN_ON_FAIL(shaderObject->writeOrdinaryData(allocation.mappedData, size, specializedLayout));

    // If we did indeed need/create a buffer, then we must bind it into
    // the given `descriptorSet` and update the base range index for
    // subsequent binding operations to account for it.
    //
    writePlainBufferDescriptor(
        m_device,
        m_bindingData->descriptorSets[ioOffset.bindingSet],
        ioOffset.binding,
        0,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        checked_cast<BufferImpl*>(allocation.buffer),
        {allocation.offset, size}
    );
    ioOffset.binding++;

    return SLANG_OK;
}

Result BindingDataBuilder::bindAsValue(
    ShaderObject* shaderObject,
    const BindingOffset& offset,
    ShaderObjectLayoutImpl* specializedLayout
)
{
    // We start by iterating over the "simple" (non-sub-object) binding
    // ranges and writing them to the descriptor sets that are being
    // passed down.
    //
    for (auto bindingRangeInfo : specializedLayout->getBindingRanges())
    {
        BindingOffset rangeOffset = offset;
        rangeOffset.bindingSet += bindingRangeInfo.setOffset;
        rangeOffset.binding += bindingRangeInfo.bindingOffset;

        DeviceImpl* device = m_device;
        uint32_t binding = rangeOffset.binding;

        uint32_t slotIndex = bindingRangeInfo.slotIndex;
        uint32_t count = bindingRangeInfo.count;

        switch (bindingRangeInfo.bindingType)
        {
        case slang::BindingType::ConstantBuffer:
        case slang::BindingType::ParameterBlock:
        case slang::BindingType::ExistentialValue:
        case slang::BindingType::PushConstant:
            break;

        case slang::BindingType::Texture:
        case slang::BindingType::MutableTexture:
        {
            VkDescriptorSet descriptorSet = m_bindingData->descriptorSets[rangeOffset.bindingSet];
            VkDescriptorType descriptorType = bindingRangeInfo.bindingType == slang::BindingType::Texture
                                                  ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                                  : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            ResourceState requiredState = bindingRangeInfo.bindingType == slang::BindingType::Texture
                                              ? ResourceState::ShaderResource
                                              : ResourceState::UnorderedAccess;
            VkImageLayout imageLayout = descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                            ? VK_IMAGE_LAYOUT_GENERAL
                                            : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            m_imageInfos.resize(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                const ResourceSlot& slot = shaderObject->m_slots[slotIndex + i];
                TextureViewImpl* textureView = checked_cast<TextureViewImpl*>(slot.resource.get());
                VkDescriptorImageInfo& imageInfo = m_imageInfos[i];
                imageInfo = {};
                if (textureView)
                {
                    imageInfo.imageView = textureView->getView().imageView;
                    imageInfo.imageLayout = imageLayout;
                    writeTextureState(this, textureView, requiredState);
                }
            }
            writeImageDescriptors(device, descriptorSet, binding, descriptorType, m_imageInfos.data(), count);
            break;
        }
        case slang::BindingType::CombinedTextureSampler:
        {
            VkDescriptorSet descriptorSet = m_bindingData->descriptorSets[rangeOffset.bindingSet];
            ResourceState requiredState = ResourceState::ShaderResource;
            m_imageInfos.resize(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                const ResourceSlot& slot = shaderObject->m_slots[slotIndex + i];
                TextureViewImpl* textureView = checked_cast<TextureViewImpl*>(slot.resource.get());
                SamplerImpl* sampler = checked_cast<SamplerImpl*>(slot.resource2.get());
                VkDescriptorImageInfo& imageInfo = m_imageInfos[i];
                imageInfo = {};
                if (textureView && sampler)
                {
                    imageInfo.imageView = textureView->getView().imageView;
                    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    imageInfo.sampler = sampler->m_sampler;
                }
                if (textureView)
                {
                    writeTextureState(this, textureView, requiredState);
                }
            }
            writeImageDescriptors(
                device,
                descriptorSet,
                binding,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                m_imageInfos.data(),
                count
            );
            break;
        }
        case slang::BindingType::Sampler:
        {
            VkDescriptorSet descriptorSet = m_bindingData->descriptorSets[rangeOffset.bindingSet];
            m_imageInfos.resize(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                const ResourceSlot& slot = shaderObject->m_slots[slotIndex + i];
                SamplerImpl* sampler = checked_cast<SamplerImpl*>(slot.resource.get());
                VkDescriptorImageInfo& imageInfo = m_imageInfos[i];
                imageInfo = {};
                imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                imageInfo.sampler = sampler ? sampler->m_sampler : device->m_defaultSampler;
            }
            writeImageDescriptors(
                device,
                descriptorSet,
                binding,
                VK_DESCRIPTOR_TYPE_SAMPLER,
                m_imageInfos.data(),
                count
            );
            break;
        }
        case slang::BindingType::RawBuffer:
        case slang::BindingType::MutableRawBuffer:
        {
            VkDescriptorSet descriptorSet = m_bindingData->descriptorSets[rangeOffset.bindingSet];
            // TODO: should RawBuffer map to VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER?
            VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ResourceState requiredState = bindingRangeInfo.bindingType == slang::BindingType::RawBuffer
                                              ? ResourceState::ShaderResource
                                              : ResourceState::UnorderedAccess;
            m_bufferInfos.resize(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                const ResourceSlot& slot = shaderObject->m_slots[slotIndex + i];
                BufferImpl* buffer = checked_cast<BufferImpl*>(slot.resource.get());
                VkDescriptorBufferInfo& bufferInfo = m_bufferInfos[i];
                bufferInfo = {};
                bufferInfo.range = VK_WHOLE_SIZE;
                if (buffer)
                {
                    bufferInfo.buffer = buffer->m_buffer.m_buffer;
                    bufferInfo.offset = slot.bufferRange.offset;
                    bufferInfo.range = slot.bufferRange.size;
                    writeBufferState(this, buffer, requiredState);
                }
            }
            writeBufferDescriptors(device, descriptorSet, binding, descriptorType, m_bufferInfos.data(), count);
            break;
        }
        case slang::BindingType::TypedBuffer:
        case slang::BindingType::MutableTypedBuffer:
        {
            VkDescriptorSet descriptorSet = m_bindingData->descriptorSets[rangeOffset.bindingSet];
            VkDescriptorType descriptorType = bindingRangeInfo.bindingType == slang::BindingType::TypedBuffer
                                                  ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
                                                  : VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
            ResourceState requiredState = bindingRangeInfo.bindingType == slang::BindingType::TypedBuffer
                                              ? ResourceState::ShaderResource
                                              : ResourceState::UnorderedAccess;
            m_bufferViews.resize(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                const ResourceSlot& slot = shaderObject->m_slots[slotIndex + i];
                BufferImpl* buffer = checked_cast<BufferImpl*>(slot.resource.get());
                m_bufferViews[i] = buffer ? buffer->getView(slot.format, slot.bufferRange) : VK_NULL_HANDLE;
                if (buffer)
                {
                    writeBufferState(this, buffer, requiredState);
                }
            }
            writeTexelBufferDescriptors(device, descriptorSet, binding, descriptorType, m_bufferViews.data(), count);
            break;
        }
        case slang::BindingType::RayTracingAccelerationStructure:
        {
            VkDescriptorSet descriptorSet = m_bindingData->descriptorSets[rangeOffset.bindingSet];
            m_accelerationStructures.resize(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                const ResourceSlot& slot = shaderObject->m_slots[slotIndex + i];
                AccelerationStructureImpl* as = checked_cast<AccelerationStructureImpl*>(slot.resource.get());
                // The Vulkan spec states: If the nullDescriptor feature is not enabled, each
                // element of pAccelerationStructures must not be VK_NULL_HANDLE
                if (!as && !device->m_api.m_extendedFeatures.robustness2Features.nullDescriptor)
                {
                    SLANG_RHI_ASSERT_FAILURE("nullDescriptor feature is not available on the device");
                    return SLANG_FAIL;
                }
                m_accelerationStructures[i] = as ? as->m_vkHandle : VK_NULL_HANDLE;
                if (as)
                {
                    writeBufferState(this, as->m_buffer, ResourceState::AccelerationStructureRead);
                }
            }
            writeAccelerationStructureDescriptors(
                device,
                descriptorSet,
                binding,
                m_accelerationStructures.data(),
                count
            );
            break;
        }

        case slang::BindingType::VaryingInput:
        case slang::BindingType::VaryingOutput:
            break;

        default:
        {
            std::string message = "Unsupported binding type: " + std::to_string((int)bindingRangeInfo.bindingType);
            SLANG_RHI_ASSERT_FAILURE(message.c_str());
            return SLANG_FAIL;
        }
        }
    }

    // Once we've handled the simple binding ranges, we move on to the
    // sub-object ranges, which are generally more involved.
    //
    for (const auto& subObjectRange : specializedLayout->getSubObjectRanges())
    {
        const auto& bindingRangeInfo = specializedLayout->getBindingRange(subObjectRange.bindingRangeIndex);
        auto count = bindingRangeInfo.count;
        auto subObjectIndex = bindingRangeInfo.subObjectIndex;

        auto subObjectLayout = subObjectRange.layout;

        // The starting offset to use for the sub-object
        // has already been computed and stored as part
        // of the layout, so we can get to the starting
        // offset for the range easily.
        //
        BindingOffset rangeOffset = offset;
        rangeOffset += subObjectRange.offset;

        BindingOffset rangeStride = subObjectRange.stride;

        switch (bindingRangeInfo.bindingType)
        {
        case slang::BindingType::ConstantBuffer:
        {
            BindingOffset objOffset = rangeOffset;
            for (uint32_t i = 0; i < count; ++i)
            {
                // Binding a constant buffer sub-object is simple enough:
                // we just call `bindAsConstantBuffer` on it to bind
                // the ordinary data buffer (if needed) and any other
                // bindings it recursively contains.
                //
                ShaderObject* subObject = shaderObject->m_objects[subObjectIndex + i];
                SLANG_RETURN_ON_FAIL(bindAsConstantBuffer(subObject, objOffset, subObjectLayout));

                // When dealing with arrays of sub-objects, we need to make
                // sure to increment the offset for each subsequent object
                // by the appropriate stride.
                //
                objOffset += rangeStride;
            }
        }
        break;
        case slang::BindingType::ParameterBlock:
        {
            BindingOffset objOffset = rangeOffset;
            for (uint32_t i = 0; i < count; ++i)
            {
                // The case for `ParameterBlock<X>` is not that different
                // from `ConstantBuffer<X>`, except that we call `bindAsParameterBlock`
                // instead (understandably).
                //
                ShaderObject* subObject = shaderObject->m_objects[subObjectIndex + i];
                SLANG_RETURN_ON_FAIL(bindAsParameterBlock(subObject, objOffset, subObjectLayout));
            }
        }
        break;

        case slang::BindingType::PushConstant:
        {
            BindingOffset objOffset = rangeOffset;
            for (uint32_t i = 0; i < count; ++i)
            {
                ShaderObject* subObject = shaderObject->m_objects[subObjectIndex + i];
                SLANG_RETURN_ON_FAIL(bindAsPushConstantBuffer(subObject, objOffset, subObjectLayout));
                objOffset += rangeStride;
            }
        }
        break;

        case slang::BindingType::ExistentialValue:
            // Interface/existential-type sub-object ranges are the most complicated case.
            //
            // First, we can only bind things if we have static specialization information
            // to work with, which is exactly the case where `subObjectLayout` will be
            // non-null.
            //
            if (subObjectLayout)
            {
                // Interface-typed sub-object ranges are no longer supported
                // after pending data layout APIs have been removed.
            }
            break;
        case slang::BindingType::RawBuffer:
        case slang::BindingType::MutableRawBuffer:
            // No action needed for sub-objects bound though a `StructuredBuffer`.
            break;
        default:
            SLANG_RHI_ASSERT_FAILURE("Unsupported sub-object type");
            return SLANG_FAIL;
            break;
        }
    }

    return SLANG_OK;
}

Result BindingDataBuilder::allocateDescriptorSets(
    ShaderObject* shaderObject,
    const BindingOffset& offset,
    ShaderObjectLayoutImpl* specializedLayout
)
{
    SLANG_RHI_ASSERT(specializedLayout->getOwnDescriptorSets().size() <= 1);
    // The number of sets to allocate and their layouts was already pre-computed
    // as part of the shader object layout, so we use that information here.
    //
    for (auto descriptorSetInfo : specializedLayout->getOwnDescriptorSets())
    {
        auto descriptorSetHandle = m_descriptorSetAllocator->allocate(descriptorSetInfo.descriptorSetLayout).handle;

        // For each set, we need to write it into the set of descriptor sets
        // being used for binding. This is done both so that other steps
        // in binding can find the set to fill it in, but also so that
        // we can bind all the descriptor sets to the pipeline when the
        // time comes.
        //
        m_bindingData->descriptorSets[m_bindingData->descriptorSetCount++] = descriptorSetHandle;
    }

    return SLANG_OK;
}

bool BindingDataBuilder::reuseRootBinding(
    ShaderObject* shaderObject,
    ShaderObjectLayoutImpl* specializedLayout,
    uint64_t key,
    VkBuffer ordinaryDataBuffer
)
{
    auto it = m_bindingCache->rootBindings.find(shaderObject);
    if (it == m_bindingCache->rootBindings.end())
        return false;

    const RootBindingCacheEntry& entry = it->second;
    if (entry.layout != specializedLayout || entry.key != key || entry.ordinaryDataBuffer != ordinaryDataBuffer)
        return false;

    // The cached arrays are arena memory that outlives every draw in this command buffer, and a
    // replay only reads them, so they are shared rather than copied.
    m_bindingData->descriptorSets = entry.descriptorSets;
    m_bindingData->descriptorSetCount = entry.descriptorSetCount;
    m_bindingData->bufferStates = entry.bufferStates;
    m_bindingData->bufferStateCount = entry.bufferStateCount;
    m_bindingData->bufferStateCapacity = entry.bufferStateCount;
    m_bindingData->textureStates = entry.textureStates;
    m_bindingData->textureStateCount = entry.textureStateCount;
    m_bindingData->textureStateCapacity = entry.textureStateCount;

    // Push constants carry per-draw ordinary data, so they are re-uploaded from their sources.
    for (uint32_t i = 0; i < entry.pushConstantCount; ++i)
    {
        const PushConstantSource& source = entry.pushConstants[i];
        const uint32_t index = m_bindingData->pushConstantCount++;
        m_bindingData->pushConstantRanges[index] = source.range;
        m_bindingData->pushConstantData[index] = m_allocator->allocate(source.range.size);
        if (SLANG_FAILED(
                source.object->writeOrdinaryData(m_bindingData->pushConstantData[index], source.range.size, source.layout)
            ))
        {
            m_bindingData->pushConstantCount = 0;
            return false;
        }
    }

    return true;
}

void BindingDataBuilder::storeRootBinding(
    ShaderObject* shaderObject,
    ShaderObjectLayoutImpl* specializedLayout,
    uint64_t key,
    VkBuffer ordinaryDataBuffer
)
{
    RootBindingCacheEntry entry = {};
    entry.layout = specializedLayout;
    entry.key = key;
    entry.ordinaryDataBuffer = ordinaryDataBuffer;
    entry.descriptorSets = m_bindingData->descriptorSets;
    entry.descriptorSetCount = m_bindingData->descriptorSetCount;
    entry.bufferStates = m_bindingData->bufferStates;
    entry.bufferStateCount = m_bindingData->bufferStateCount;
    entry.textureStates = m_bindingData->textureStates;
    entry.textureStateCount = m_bindingData->textureStateCount;

    entry.pushConstantCount = (uint32_t)m_pushConstantSources.size();
    if (entry.pushConstantCount)
    {
        entry.pushConstants = m_allocator->allocate<PushConstantSource>(entry.pushConstantCount);
        std::memcpy(
            entry.pushConstants,
            m_pushConstantSources.data(),
            entry.pushConstantCount * sizeof(PushConstantSource)
        );
    }

    m_bindingCache->rootBindings[shaderObject] = entry;
}

bool BindingDataBuilder::reuseParameterBlock(
    ShaderObject* shaderObject,
    ShaderObjectLayoutImpl* specializedLayout,
    uint64_t version
)
{
    for (const ParameterBlockCacheEntry& entry : m_bindingCache->parameterBlocks)
    {
        if (entry.object != shaderObject || entry.layout != specializedLayout || entry.version != version)
            continue;

        for (uint32_t i = 0; i < entry.descriptorSetCount; ++i)
            m_bindingData->descriptorSets[m_bindingData->descriptorSetCount++] = entry.descriptorSets[i];

        // The bindings are reused, but the resource states they require still have to be
        // declared for this draw so the recorder emits the same barriers as a fresh build.
        for (uint32_t i = 0; i < entry.bufferStateCount; ++i)
            writeBufferState(this, entry.bufferStates[i].buffer, entry.bufferStates[i].state);
        for (uint32_t i = 0; i < entry.textureStateCount; ++i)
            writeTextureState(this, entry.textureStates[i].textureView, entry.textureStates[i].state);

        return true;
    }
    return false;
}

void BindingDataBuilder::storeParameterBlock(
    ShaderObject* shaderObject,
    ShaderObjectLayoutImpl* specializedLayout,
    uint64_t version,
    uint32_t firstDescriptorSet,
    uint32_t firstBufferState,
    uint32_t firstTextureState
)
{
    ParameterBlockCacheEntry entry = {};
    entry.object = shaderObject;
    entry.layout = specializedLayout;
    entry.version = version;

    entry.descriptorSetCount = m_bindingData->descriptorSetCount - firstDescriptorSet;
    if (entry.descriptorSetCount)
    {
        entry.descriptorSets = m_allocator->allocate<VkDescriptorSet>(entry.descriptorSetCount);
        std::memcpy(
            entry.descriptorSets,
            m_bindingData->descriptorSets + firstDescriptorSet,
            entry.descriptorSetCount * sizeof(VkDescriptorSet)
        );
    }

    entry.bufferStateCount = m_bindingData->bufferStateCount - firstBufferState;
    if (entry.bufferStateCount)
    {
        entry.bufferStates = m_allocator->allocate<BindingDataImpl::BufferState>(entry.bufferStateCount);
        std::memcpy(
            entry.bufferStates,
            m_bindingData->bufferStates + firstBufferState,
            entry.bufferStateCount * sizeof(BindingDataImpl::BufferState)
        );
    }

    entry.textureStateCount = m_bindingData->textureStateCount - firstTextureState;
    if (entry.textureStateCount)
    {
        entry.textureStates = m_allocator->allocate<BindingDataImpl::TextureState>(entry.textureStateCount);
        std::memcpy(
            entry.textureStates,
            m_bindingData->textureStates + firstTextureState,
            entry.textureStateCount * sizeof(BindingDataImpl::TextureState)
        );
    }

    // A new version supersedes the old one; the stale sets stay allocated until the
    // command buffer resets its descriptor pools, and earlier binding data still uses them.
    for (ParameterBlockCacheEntry& existing : m_bindingCache->parameterBlocks)
    {
        if (existing.object == shaderObject && existing.layout == specializedLayout)
        {
            existing = entry;
            return;
        }
    }
    m_bindingCache->parameterBlocks.push_back(entry);
}

Result BindingDataBuilder::bindAsParameterBlock(
    ShaderObject* shaderObject,
    const BindingOffset& inOffset,
    ShaderObjectLayoutImpl* specializedLayout
)
{
    // Because we are binding into a nested parameter block,
    // any texture/buffer/sampler bindings will now want to
    // write into the sets we allocate for this object and
    // not the sets for any parent object(s).
    //
    BindingOffset offset = inOffset;
    offset.bindingSet = m_bindingData->descriptorSetCount;
    offset.binding = 0;

    // A parameter block's sets depend only on its contents, so an unchanged block can reuse
    // the sets built for it earlier in this command buffer instead of rewriting every descriptor.
    const uint64_t version = shaderObject->getCompositeVersion();
    if (reuseParameterBlock(shaderObject, specializedLayout, version))
        return SLANG_OK;

    const uint32_t firstDescriptorSet = m_bindingData->descriptorSetCount;
    const uint32_t firstBufferState = m_bindingData->bufferStateCount;
    const uint32_t firstTextureState = m_bindingData->textureStateCount;
    const uint32_t pushConstantCount = m_bindingData->pushConstantCount;

    // Note: Interface-type binding handling has been simplified
    // now that pending data layout APIs have been removed.

    // Writing the bindings for a parameter block is relatively easy:
    // we just need to allocate the descriptor set(s) needed for this
    // object and then fill it in like a `ConstantBuffer<X>`.
    //
    SLANG_RETURN_ON_FAIL(allocateDescriptorSets(shaderObject, offset, specializedLayout));

    SLANG_RHI_ASSERT(offset.bindingSet < m_bindingData->descriptorSetCount);
    SLANG_RETURN_ON_FAIL(bindAsConstantBuffer(shaderObject, offset, specializedLayout));

    // Push constants live in the binding data, not in the descriptor sets, so a block
    // that emitted any cannot be replayed from the cache alone.
    if (m_bindingData->pushConstantCount == pushConstantCount)
    {
        storeParameterBlock(
            shaderObject,
            specializedLayout,
            version,
            firstDescriptorSet,
            firstBufferState,
            firstTextureState
        );
    }

    return SLANG_OK;
}

Result BindingDataBuilder::bindAsConstantBuffer(
    ShaderObject* shaderObject,
    const BindingOffset& inOffset,
    ShaderObjectLayoutImpl* specializedLayout
)
{
    // To bind an object as a constant buffer, we first
    // need to bind its ordinary data (if any) into an
    // ordinary data buffer, and then bind it as a "value"
    // which handles any of its recursively-contained bindings.
    //
    // The one detail is taht when binding the ordinary data
    // buffer we need to adjust the `binding` index used for
    // subsequent operations based on whether or not an ordinary
    // data buffer was used (and thus consumed a `binding`).
    //
    BindingOffset offset = inOffset;
    SLANG_RETURN_ON_FAIL(bindOrdinaryDataBufferIfNeeded(shaderObject, /*inout*/ offset, specializedLayout));
    SLANG_RETURN_ON_FAIL(bindAsValue(shaderObject, offset, specializedLayout));
    return SLANG_OK;
}

} // namespace rhi::vk
