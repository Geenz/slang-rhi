#include "metal-shader-object.h"
#include "metal-command.h"
#include "metal-bindless-descriptor-set.h"
#include "metal-device.h"
#include "metal-acceleration-structure.h"
#include "metal-buffer.h"
#include "metal-texture.h"
#include "metal-sampler.h"
#include "metal-utils.h"
#include <slang.h>
#include <atomic>
#include <cstdio>

namespace rhi::metal {

// [ARGCACHE PROBE — TEMPORARY, team-flow M2 confirmation; REVERT after measuring]
// Counts argument-buffer cache hits vs the two miss causes:
//   miss_newobj  = a RootShaderObject pointer never seen before (object churn)
//   miss_verbump = a reused pointer whose m_version changed (params re-set this frame)
// Throttled dump to stderr so the app log can be grepped for [ARGCACHE].
namespace {
std::atomic<uint64_t> sArgHits{0};
std::atomic<uint64_t> sArgMissNewObj{0};
std::atomic<uint64_t> sArgMissVerBump{0};
inline void argCacheDump()
{
    uint64_t h = sArgHits.load(std::memory_order_relaxed);
    uint64_t mn = sArgMissNewObj.load(std::memory_order_relaxed);
    uint64_t mv = sArgMissVerBump.load(std::memory_order_relaxed);
    uint64_t tot = h + mn + mv;
    if (tot && (tot % 4000ull) == 0)
        fprintf(stderr,
            "[ARGCACHE] total=%llu hits=%llu (%.1f%%) miss_newobj=%llu miss_verbump=%llu\n",
            (unsigned long long)tot, (unsigned long long)h,
            100.0 * double(h) / double(tot),
            (unsigned long long)mn, (unsigned long long)mv);
}
} // namespace

inline Result setBuffer(BindingDataImpl* bindingData, uint32_t index, MTL::Buffer* buffer, NS::UInteger offset = 0)
{
    if (index >= bindingData->bufferCapacity)
    {
        return SLANG_FAIL;
    }
    bindingData->bufferCount = max(bindingData->bufferCount, index + 1);
    bindingData->buffers[index] = buffer;
    bindingData->bufferOffsets[index] = offset;
    return SLANG_OK;
}

inline Result setTexture(BindingDataImpl* bindingData, uint32_t index, MTL::Texture* texture)
{
    if (index >= bindingData->textureCapacity)
    {
        return SLANG_FAIL;
    }
    bindingData->textureCount = max(bindingData->textureCount, index + 1);
    bindingData->textures[index] = texture;
    return SLANG_OK;
}

inline Result addUsedResource(BindingDataImpl* bindingData, MTL::Resource* resource)
{
    if (bindingData->usedResourceCount >= bindingData->usedResourceCapacity)
        return SLANG_OK; // Silently skip — resource is likely covered by bindless useResources
    bindingData->usedResources[bindingData->usedResourceCount++] = resource;
    return SLANG_OK;
}

inline Result addUsedRWResource(BindingDataImpl* bindingData, MTL::Resource* resource)
{
    if (bindingData->usedRWResourceCount >= bindingData->usedRWResourceCapacity)
        return SLANG_OK; // Silently skip — resource is likely covered by bindless useResources
    bindingData->usedRWResources[bindingData->usedRWResourceCount++] = resource;
    return SLANG_OK;
}

Result BindingDataBuilder::bindAsRootFlat(
    RootShaderObject* shaderObject,
    RootShaderObjectLayoutImpl* specializedLayout,
    BindingDataImpl*& outBindingData
)
{
    const FlatBindingTable& table = specializedLayout->getFlatBindingTable();

    BindingDataImpl* bindingData = m_allocator->allocate<BindingDataImpl>();
    m_bindingData = bindingData;

    uint32_t bufCount = table.maxBufferRegister;
    uint32_t texCount = table.maxTextureRegister;
    uint32_t samplerCount = specializedLayout->getTotalSamplerCount();
    uint32_t usedResCapacity = table.maxUsedResources + table.ordinaryData.size() + 1;
    uint32_t usedRWResCapacity = table.maxUsedRWResources + 1;

    m_bindingData->bufferCapacity = bufCount;
    m_bindingData->bufferCount = bufCount;
    m_bindingData->buffers = m_allocator->allocate<MTL::Buffer*>(bufCount);
    ::memset(m_bindingData->buffers, 0, sizeof(MTL::Buffer*) * bufCount);
    m_bindingData->bufferOffsets = m_allocator->allocate<NS::UInteger>(bufCount);
    ::memset(m_bindingData->bufferOffsets, 0, sizeof(NS::UInteger) * bufCount);

    m_bindingData->textureCapacity = texCount;
    m_bindingData->textureCount = texCount;
    m_bindingData->textures = m_allocator->allocate<MTL::Texture*>(texCount);
    ::memset(m_bindingData->textures, 0, sizeof(MTL::Texture*) * texCount);

    m_bindingData->samplerCount = samplerCount;
    m_bindingData->samplers = m_allocator->allocate<MTL::SamplerState*>(samplerCount);
    ::memset(m_bindingData->samplers, 0, sizeof(MTL::SamplerState*) * samplerCount);

    m_bindingData->usedResourceCount = 0;
    m_bindingData->usedResourceCapacity = usedResCapacity;
    m_bindingData->usedResources = m_allocator->allocate<MTL::Resource*>(usedResCapacity);
    m_bindingData->usedRWResourceCount = 0;
    m_bindingData->usedRWResourceCapacity = usedRWResCapacity;
    m_bindingData->usedRWResources = m_allocator->allocate<MTL::Resource*>(usedRWResCapacity);

    // Collect ShaderObject pointers using pre-computed paths.
    ShaderObject* objects[64];
    uint16_t objCount = table.objectCount;
    SLANG_RHI_ASSERT(objCount <= 64);

    for (uint16_t i = 0; i < objCount; ++i)
    {
        const auto& path = table.objectPaths[i];
        switch (path.source)
        {
        case FlatObjectPath::Source::Root:
            objects[i] = shaderObject;
            break;
        case FlatObjectPath::Source::SubObject:
            objects[i] = objects[path.parentIndex] ? objects[path.parentIndex]->m_objects[path.subObjectSlot] : nullptr;
            break;
        case FlatObjectPath::Source::EntryPoint:
            objects[i] = shaderObject->m_entryPoints[path.subObjectSlot];
            break;
        }
    }

    // Process ordinary data entries (constant buffer uploads)
    for (const auto& od : table.ordinaryData)
    {
        ShaderObject* obj = objects[od.objectTreeIndex];
        if (!obj || od.dataSize == 0)
            continue;

        ConstantBufferPool::Allocation allocation;
        SLANG_RETURN_ON_FAIL(m_constantBufferPool->allocate(od.dataSize, allocation));
        ::memcpy(allocation.mappedData, obj->m_data.data(), std::min((size_t)od.dataSize, obj->m_data.size()));

        SLANG_RETURN_ON_FAIL(setBuffer(m_bindingData, od.bufferRegister,
                                        allocation.buffer->m_buffer.get(), allocation.offset));

        if (!m_device->m_hasUnifiedMemory)
            allocation.buffer->m_buffer->didModifyRange(NS::Range(allocation.offset, od.dataSize));
    }

    // Process binding entries (single flat loop)
    for (const auto& entry : table.entries)
    {
        ShaderObject* obj = objects[entry.objectTreeIndex];
        if (!obj)
            continue;

        const ResourceSlot& slot = obj->m_slots[entry.slotIndex];
        if (!slot.resource)
            continue;

        switch (entry.type)
        {
        case FlatBindingEntry::Type::Buffer:
        {
            BufferImpl* buffer = static_cast<BufferImpl*>(slot.resource.get());
            setBuffer(m_bindingData, entry.registerIndex, buffer->m_buffer.get(), slot.bufferRange.offset);
            if (entry.isMutable)
                addUsedRWResource(m_bindingData, buffer->m_buffer.get());
            else
                addUsedResource(m_bindingData, buffer->m_buffer.get());
            break;
        }
        case FlatBindingEntry::Type::Texture:
        {
            TextureViewImpl* textureView = static_cast<TextureViewImpl*>(slot.resource.get());
            setTexture(m_bindingData, entry.registerIndex, textureView->m_textureView.get());
            if (entry.isMutable)
                addUsedRWResource(m_bindingData, textureView->m_textureView.get());
            else
                addUsedResource(m_bindingData, textureView->m_textureView.get());
            break;
        }
        case FlatBindingEntry::Type::Sampler:
        {
            SamplerImpl* sampler = static_cast<SamplerImpl*>(slot.resource.get());
            SLANG_RHI_ASSERT(entry.registerIndex < m_bindingData->samplerCount);
            m_bindingData->samplers[entry.registerIndex] = sampler->m_samplerState.get();
            break;
        }
        }
    }

    // Handle ParameterBlock sub-objects via argument buffer cache.
    // Cache lives on CommandQueueImpl for cross-frame persistence.
    // Multiple camera worker threads may call getBindingData() concurrently
    // on encoders sharing the same queue, so all cache access is guarded
    // by m_argCacheMutex.
    auto* argCache = static_cast<std::vector<CommandQueueImpl::CachedArgBuffer>*>(m_cachedArgBuffers);

    for (const auto& ab : table.argBuffers)
    {
        ShaderObject* obj = objects[ab.objectTreeIndex];
        if (!obj || ab.bufferSize == 0)
            continue;

        // Check argument buffer cache — reuse if version unchanged
        if (argCache && m_argCacheMutex)
        {
            std::lock_guard<std::mutex> lock(*m_argCacheMutex);
            bool hit = false;
            for (auto& c : *argCache)
            {
                if (c.object == obj && c.version == obj->m_version)
                {
                    SLANG_RETURN_ON_FAIL(setBuffer(m_bindingData, ab.bindingDataRegister, c.buffer.get(), 0));
                    for (auto* r : c.usedResources)
                        addUsedResource(m_bindingData, r);
                    for (auto* r : c.usedRWResources)
                        addUsedRWResource(m_bindingData, r);
                    hit = true;
                    sArgHits.fetch_add(1, std::memory_order_relaxed); // [ARGCACHE PROBE]
                    break;
                }
            }
            if (hit) continue;
        }

        // Use the layout captured per-ab at table-build time. The
        // previous code here scanned `specializedLayout->m_subObjectRanges`
        // and broke on the first `ParameterBlock` found, then used the
        // resulting `pbLayout` for ALL `argBuffers`. That misbinds when
        // a shader has multiple ParameterBlocks of different types
        // (e.g. `_bindlessTextureHeap` and `_bindlessSamplerHeap`):
        // iterating the sampler-heap argBuffer with the texture-heap's
        // layout reads the sampler sub-object's `m_slots` against
        // texture-typed binding ranges, producing
        // `checked_cast<TextureViewImpl*>(samplerImpl)` failures inside
        // `writeArgumentBuffer`.
        ShaderObjectLayoutImpl* pbLayout = ab.pbLayout;

        MTL::Buffer* argBuffer = nullptr;
        NS::UInteger argOffset = 0;

        if (pbLayout)
        {
            // Full writeArgumentBuffer — re-encodes ALL bindings
            // (uniform data, textures, buffers, samplers) from the
            // current shader object state. A prior version of this code
            // had a "fast path" that only patched uniform data via
            // pre-computed dataCopies when the version changed. That
            // left stale resource handles (GPU addresses of textures /
            // buffers from the cold miss) in the argument buffer,
            // causing wrong-texture rendering and eventual GPU page
            // faults when the original resources were freed.
            uint32_t prevUsedRes = m_bindingData->usedResourceCount;
            uint32_t prevUsedRW = m_bindingData->usedRWResourceCount;

            SLANG_RETURN_ON_FAIL(writeArgumentBuffer(obj, pbLayout, argBuffer, argOffset));

            if (argBuffer && argCache && m_argCacheMutex)
            {
                std::vector<MTL::Resource*> res, rwRes;
                for (uint32_t i = prevUsedRes; i < m_bindingData->usedResourceCount; ++i)
                    res.push_back(m_bindingData->usedResources[i]);
                for (uint32_t i = prevUsedRW; i < m_bindingData->usedRWResourceCount; ++i)
                    rwRes.push_back(m_bindingData->usedRWResources[i]);

                std::lock_guard<std::mutex> lock(*m_argCacheMutex);
                bool updated = false;
                for (auto& c : *argCache)
                {
                    if (c.object == obj) {
                        c.version = obj->m_version;
                        c.buffer = NS::RetainPtr(argBuffer);
                        c.usedResources = std::move(res);
                        c.usedRWResources = std::move(rwRes);
                        updated = true;
                        sArgMissVerBump.fetch_add(1, std::memory_order_relaxed); // [ARGCACHE PROBE]
                        break;
                    }
                }
                if (!updated) {
                    argCache->push_back({obj, obj->m_version, NS::RetainPtr(argBuffer), std::move(res), std::move(rwRes)});
                    sArgMissNewObj.fetch_add(1, std::memory_order_relaxed); // [ARGCACHE PROBE]
                }
            }
        }

        if (argBuffer)
        {
            SLANG_RETURN_ON_FAIL(setBuffer(m_bindingData, ab.bindingDataRegister, argBuffer, argOffset));
        }
    }

    argCacheDump(); // [ARGCACHE PROBE]
    outBindingData = bindingData;
    return SLANG_OK;
}

Result BindingDataBuilder::bindAsRoot(
    RootShaderObject* shaderObject,
    RootShaderObjectLayoutImpl* specializedLayout,
    BindingDataImpl*& outBindingData
)
{
    BindingDataImpl* bindingData = m_allocator->allocate<BindingDataImpl>();
    m_bindingData = bindingData;
 
    uint32_t samplerCount = specializedLayout->getTotalSamplerCount();

    if (m_previousBindingData && m_previousBindingData->bufferCapacity == 256 &&
        m_previousBindingData->samplerCount == samplerCount)
    {
        // Clone arrays from previous binding data (memcpy instead of alloc+memset).
        // The tree walk below will overwrite changed slots.
        m_bindingData->bufferCapacity = 256;
        m_bindingData->textureCapacity = 256;
        m_bindingData->usedResourceCapacity = 256;
        m_bindingData->usedRWResourceCapacity = 256;

        m_bindingData->buffers = m_allocator->allocate<MTL::Buffer*>(256);
        ::memcpy(m_bindingData->buffers, m_previousBindingData->buffers, sizeof(MTL::Buffer*) * 256);
        m_bindingData->bufferOffsets = m_allocator->allocate<NS::UInteger>(256);
        ::memcpy(m_bindingData->bufferOffsets, m_previousBindingData->bufferOffsets, sizeof(NS::UInteger) * 256);
        m_bindingData->bufferCount = m_previousBindingData->bufferCount;

        m_bindingData->textures = m_allocator->allocate<MTL::Texture*>(256);
        ::memcpy(m_bindingData->textures, m_previousBindingData->textures, sizeof(MTL::Texture*) * 256);
        m_bindingData->textureCount = m_previousBindingData->textureCount;

        m_bindingData->samplerCount = samplerCount;
        m_bindingData->samplers = m_allocator->allocate<MTL::SamplerState*>(samplerCount);
        ::memcpy(m_bindingData->samplers, m_previousBindingData->samplers, sizeof(MTL::SamplerState*) * samplerCount);

        // usedResource arrays must be rebuilt each time (addUsedResource appends)
        m_bindingData->usedResourceCount = 0;
        m_bindingData->usedResources = m_allocator->allocate<MTL::Resource*>(256);
        m_bindingData->usedRWResourceCount = 0;
        m_bindingData->usedRWResources = m_allocator->allocate<MTL::Resource*>(256);
    }
    else
    {
        // Fresh allocation path (first call or layout mismatch)
        m_bindingData->bufferCapacity = 256;
        m_bindingData->textureCapacity = 256;
        m_bindingData->usedResourceCapacity = 256;
        m_bindingData->usedRWResourceCapacity = 256;

        m_bindingData->bufferCount = 0;
        m_bindingData->buffers = m_allocator->allocate<MTL::Buffer*>(256);
        ::memset(m_bindingData->buffers, 0, sizeof(MTL::Buffer*) * 256);
        m_bindingData->bufferOffsets = m_allocator->allocate<NS::UInteger>(256);
        ::memset(m_bindingData->bufferOffsets, 0, sizeof(NS::UInteger) * 256);

        m_bindingData->textureCount = 0;
        m_bindingData->textures = m_allocator->allocate<MTL::Texture*>(256);
        ::memset(m_bindingData->textures, 0, sizeof(MTL::Texture*) * 256);

        m_bindingData->samplerCount = samplerCount;
        m_bindingData->samplers = m_allocator->allocate<MTL::SamplerState*>(samplerCount);
        ::memset(m_bindingData->samplers, 0, sizeof(MTL::SamplerState*) * samplerCount);

        m_bindingData->usedResourceCount = 0;
        m_bindingData->usedResources = m_allocator->allocate<MTL::Resource*>(256);
        m_bindingData->usedRWResourceCount = 0;
        m_bindingData->usedRWResources = m_allocator->allocate<MTL::Resource*>(256);
    }

    // Initialize binding offset for shader parameters.
    //
    BindingOffset offset;

    BindingOffset ordinaryDataBufferOffset = offset;
    SLANG_RETURN_ON_FAIL(bindOrdinaryDataBufferIfNeeded(shaderObject, ordinaryDataBufferOffset, specializedLayout));
    SLANG_RETURN_ON_FAIL(bindAsValue(shaderObject, offset, specializedLayout));

    // Once the state stored in the root shader object itself has been bound,
    // we turn our attention to the entry points and their parameters.
    //
    size_t entryPointCount = specializedLayout->m_entryPoints.size();
    for (size_t i = 0; i < entryPointCount; ++i)
    {
        auto entryPoint = shaderObject->m_entryPoints[i];
        const auto& entryPointInfo = specializedLayout->m_entryPoints[i];
        ShaderObjectLayoutImpl* entryPointLayout = entryPointInfo.layout;

        BindingOffset entryPointOffset = offset;
        entryPointOffset += entryPointInfo.offset;

        SLANG_RETURN_ON_FAIL(bindAsConstantBuffer(entryPoint, entryPointOffset, entryPointLayout));
    }

    outBindingData = bindingData;

    return SLANG_OK;
}

/// Bind this object as if it was declared as a `ConstantBuffer<T>` in Slang
Result BindingDataBuilder::bindAsConstantBuffer(
    ShaderObject* shaderObject,
    const BindingOffset& inOffset,
    ShaderObjectLayoutImpl* specializedLayout
)
{
    // When binding a `ConstantBuffer<X>` we need to first bind a constant
    // buffer for any "ordinary" data in `X`, and then bind the remaining
    // resources and sub-objects.
    //
    BindingOffset offset = inOffset;
    SLANG_RETURN_ON_FAIL(bindOrdinaryDataBufferIfNeeded(shaderObject, /*inout*/ offset, specializedLayout));

    // Once the ordinary data buffer is bound, we can move on to binding
    // the rest of the state, which can use logic shared with the case
    // for interface-type sub-object ranges.
    //
    // Note that this call will use the `inOffset` value instead of the offset
    // modified by `_bindOrindaryDataBufferIfNeeded', because the indexOffset in
    // the binding range should already take care of the offset due to the default
    // cbuffer.
    //
    SLANG_RETURN_ON_FAIL(bindAsValue(shaderObject, inOffset, specializedLayout));

    return SLANG_OK;
}

/// Bind this object as if it was declared as a `ParameterBlock<T>` in Slang
Result BindingDataBuilder::bindAsParameterBlock(
    ShaderObject* shaderObject,
    const BindingOffset& inOffset,
    ShaderObjectLayoutImpl* specializedLayout
)
{
    if (!m_device->m_hasArgumentBufferTier2)
        return SLANG_FAIL;

    MTL::Buffer* argumentBuffer = nullptr;
    NS::UInteger argumentOffset = 0;
    SLANG_RETURN_ON_FAIL(writeArgumentBuffer(shaderObject, specializedLayout, argumentBuffer, argumentOffset));

    if (argumentBuffer)
    {
        SLANG_RETURN_ON_FAIL(setBuffer(m_bindingData, inOffset.buffer, argumentBuffer, argumentOffset));
    }

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
    // Get the bindless descriptor set for descriptor handle resolution
    BindlessDescriptorSet* bindlessSet = m_device->m_bindlessDescriptorSet;

    uint32_t bindingRangeIdx = 0;
    for (const auto& bindingRangeInfo : specializedLayout->m_bindingRanges)
    {
        uint32_t slotIndex = bindingRangeInfo.slotIndex;
        uint32_t count = bindingRangeInfo.count;
        switch (bindingRangeInfo.bindingType)
        {
        case slang::BindingType::ConstantBuffer:
        case slang::BindingType::ParameterBlock:
        case slang::BindingType::ExistentialValue:
            break;

        case slang::BindingType::Texture:
        case slang::BindingType::MutableTexture:
            for (uint32_t i = 0; i < count; ++i)
            {
                const ResourceSlot& slot = shaderObject->m_slots[slotIndex + i];
                TextureViewImpl* textureView = checked_cast<TextureViewImpl*>(slot.resource.get());
                if (textureView)
                {
                    uint32_t registerIndex = bindingRangeInfo.registerOffset + offset.texture + i;
                    SLANG_RETURN_ON_FAIL(setTexture(m_bindingData, registerIndex, textureView->m_textureView.get()));
                }
                else if (bindlessSet)
                {
                    // Check for descriptor handle
                    uint64_t key = makeDescriptorHandleKey(bindingRangeIdx, i);
                    auto it = shaderObject->m_descriptorHandles.find(key);
                    if (it != shaderObject->m_descriptorHandles.end())
                    {
                        auto texIt = bindlessSet->m_handleToTexturePtr.find(it->second);
                        if (texIt != bindlessSet->m_handleToTexturePtr.end())
                        {
                            uint32_t registerIndex = bindingRangeInfo.registerOffset + offset.texture + i;
                            SLANG_RETURN_ON_FAIL(setTexture(m_bindingData, registerIndex, texIt->second));
                        }
                    }
                }
            }
            break;
        case slang::BindingType::Sampler:
            for (uint32_t i = 0; i < count; ++i)
            {
                const ResourceSlot& slot = shaderObject->m_slots[slotIndex + i];
                SamplerImpl* sampler = checked_cast<SamplerImpl*>(slot.resource.get());
                if (sampler)
                {
                    uint32_t registerIndex = bindingRangeInfo.registerOffset + offset.sampler + i;
                    SLANG_RHI_ASSERT(registerIndex < m_bindingData->samplerCount);
                    m_bindingData->samplers[registerIndex] = sampler->m_samplerState.get();
                }
                else if (bindlessSet)
                {
                    uint64_t key = makeDescriptorHandleKey(bindingRangeIdx, i);
                    auto it = shaderObject->m_descriptorHandles.find(key);
                    if (it != shaderObject->m_descriptorHandles.end())
                    {
                        auto sampIt = bindlessSet->m_handleToSamplerPtr.find(it->second);
                        if (sampIt != bindlessSet->m_handleToSamplerPtr.end())
                        {
                            uint32_t registerIndex = bindingRangeInfo.registerOffset + offset.sampler + i;
                            SLANG_RHI_ASSERT(registerIndex < m_bindingData->samplerCount);
                            m_bindingData->samplers[registerIndex] = sampIt->second;
                        }
                    }
                }
            }
            break;
        case slang::BindingType::RawBuffer:
        case slang::BindingType::MutableRawBuffer:
        case slang::BindingType::TypedBuffer:
        case slang::BindingType::MutableTypedBuffer:
            for (uint32_t i = 0; i < count; ++i)
            {
                const ResourceSlot& slot = shaderObject->m_slots[slotIndex + i];
                BufferImpl* buffer = checked_cast<BufferImpl*>(slot.resource.get());
                if (buffer)
                {
                    uint32_t registerIndex = bindingRangeInfo.registerOffset + offset.buffer + i;
                    SLANG_RETURN_ON_FAIL(
                        setBuffer(m_bindingData, registerIndex, buffer->m_buffer.get(), slot.bufferRange.offset)
                    );
                }
                else if (bindlessSet)
                {
                    uint64_t key = makeDescriptorHandleKey(bindingRangeIdx, i);
                    auto it = shaderObject->m_descriptorHandles.find(key);
                    if (it != shaderObject->m_descriptorHandles.end())
                    {
                        // Try buffer pointer first (StructuredBuffer, ByteAddressBuffer)
                        auto bufIt = bindlessSet->m_handleToBufferPtr.find(it->second);
                        if (bufIt != bindlessSet->m_handleToBufferPtr.end())
                        {
                            uint32_t registerIndex = bindingRangeInfo.registerOffset + offset.buffer + i;
                            SLANG_RETURN_ON_FAIL(
                                setBuffer(m_bindingData, registerIndex, bufIt->second.buffer, bufIt->second.offset)
                            );
                            if (bindingRangeInfo.bindingType == slang::BindingType::MutableRawBuffer ||
                                bindingRangeInfo.bindingType == slang::BindingType::MutableTypedBuffer)
                            {
                                SLANG_RETURN_ON_FAIL(addUsedRWResource(m_bindingData, bufIt->second.buffer));
                            }
                            else
                            {
                                SLANG_RETURN_ON_FAIL(addUsedResource(m_bindingData, bufIt->second.buffer));
                            }
                        }
                        else
                        {
                            // Typed buffers (Buffer<T>) are backed by texture views on Metal
                            auto texIt = bindlessSet->m_handleToTexturePtr.find(it->second);
                            if (texIt != bindlessSet->m_handleToTexturePtr.end())
                            {
                                uint32_t registerIndex = bindingRangeInfo.registerOffset + offset.texture + i;
                                SLANG_RETURN_ON_FAIL(setTexture(m_bindingData, registerIndex, texIt->second));
                            }
                        }
                    }
                }
            }
            break;
        case slang::BindingType::RayTracingAccelerationStructure:
            // Acceleration structures are bound via argument buffers, not directly
            break;
        case slang::BindingType::VaryingInput:
        case slang::BindingType::VaryingOutput:
            break;
        default:
            SLANG_RHI_ASSERT_FAILURE("Unsupported binding type");
            return SLANG_FAIL;
            break;
        }
        bindingRangeIdx++;
    }

    // Once all the simple binding ranges are dealt with, we will bind
    // all of the sub-objects in sub-object ranges.
    //
    for (const auto& subObjectRange : specializedLayout->m_subObjectRanges)
    {
        auto subObjectLayout = subObjectRange.layout;
        const auto& bindingRange = specializedLayout->m_bindingRanges[subObjectRange.bindingRangeIndex];
        uint32_t count = bindingRange.count;
        uint32_t subObjectIndex = bindingRange.subObjectIndex;

        // The starting offset for a sub-object range was computed
        // from Slang reflection information, so we can apply it here.
        //
        BindingOffset rangeOffset = offset;
        rangeOffset += subObjectRange.offset;

        // Similarly, the "stride" between consecutive objects in
        // the range was also pre-computed.
        //
        BindingOffset rangeStride = subObjectRange.stride;

        switch (bindingRange.bindingType)
        {
        case slang::BindingType::ConstantBuffer:
        {
            BindingOffset objOffset = rangeOffset;
            for (uint32_t i = 0; i < count; ++i)
            {
                auto subObject = shaderObject->m_objects[subObjectIndex + i];

                // Record current version for next call's skip detection
                if (m_currentVersions && subObject)
                    m_currentVersions->push_back({subObject, subObject->m_version});

                // Skip if sub-object version unchanged and we have cloned previous binding data
                if (m_previousBindingData && m_previousVersions && subObject)
                {
                    bool canSkip = false;
                    for (const auto& prev : *m_previousVersions)
                    {
                        if (prev.object == subObject && prev.version == subObject->m_version)
                        {
                            canSkip = true;
                            break;
                        }
                    }
                    if (canSkip)
                    {
                        objOffset += rangeStride;
                        continue;
                    }
                }

                SLANG_RETURN_ON_FAIL(bindAsConstantBuffer(subObject, objOffset, subObjectLayout));

                objOffset += rangeStride;
            }
            break;
        }
        case slang::BindingType::ParameterBlock:
        {
            BindingOffset objOffset = rangeOffset;
            for (uint32_t i = 0; i < count; ++i)
            {
                auto subObject = shaderObject->m_objects[subObjectIndex + i];
                SLANG_RETURN_ON_FAIL(bindAsParameterBlock(subObject, objOffset, subObjectLayout));
                objOffset += rangeStride;
            }
        }
        break;

#if 0
        case slang::BindingType::ExistentialValue:
            // We can only bind information for existential-typed sub-object
            // ranges if we have a static type that we are able to specialize to.
            //
            if (subObjectLayout)
            {
                SimpleBindingOffset objOffset = {};
                SimpleBindingOffset objStride = {};

                for (Index i = 0; i < count; ++i)
                {
                    auto subObject = m_objects[subObjectIndex + i];
                    subObject->bindAsValue(context, BindingOffset(objOffset), subObjectLayout);

                    objOffset += objStride;
                }
            }
            break;
#endif

        default:
            break;
        }
    }

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
        return SLANG_OK;

    ConstantBufferPool::Allocation allocation;
    SLANG_RETURN_ON_FAIL(m_constantBufferPool->allocate(size, allocation));

    SLANG_RETURN_ON_FAIL(shaderObject->writeOrdinaryData(allocation.mappedData, size, specializedLayout));

    SLANG_RETURN_ON_FAIL(
        setBuffer(m_bindingData, ioOffset.buffer, allocation.buffer->m_buffer.get(), allocation.offset)
    );
    ioOffset.buffer++;

    if (!m_device->m_hasUnifiedMemory)
        allocation.buffer->m_buffer->didModifyRange(NS::Range(allocation.offset, size));

    return SLANG_OK;
}

Result BindingDataBuilder::writeArgumentBuffer(
    ShaderObject* shaderObject,
    ShaderObjectLayoutImpl* specializedLayout,
    MTL::Buffer*& outBuffer,
    NS::UInteger& outOffset
)
{
    auto argumentBufferTypeLayout = specializedLayout->getParameterBlockTypeLayout();

    // If the argument buffer has no fields, we don't need to create one, note this is legal because there could be an
    // empty struct type in AST. We need to handle this correctly.
    if (argumentBufferTypeLayout->getFieldCount() == 0)
    {
        outBuffer = nullptr;
        outOffset = 0;
        return SLANG_OK;
    }

    ComPtr<IBuffer> argumentBuffer;
    BufferDesc argumentBufferDesc = {};
    argumentBufferDesc.size = argumentBufferTypeLayout->getSize();
    argumentBufferDesc.usage = BufferUsage::ConstantBuffer | BufferUsage::CopyDestination;
    argumentBufferDesc.defaultState = ResourceState::ConstantBuffer;
    argumentBufferDesc.memoryType = MemoryType::Upload;
    argumentBufferDesc.label = "ArgumentBuffer";
    SLANG_RETURN_ON_FAIL(m_device->createBuffer(argumentBufferDesc, nullptr, argumentBuffer.writeRef()));
    auto argumentBufferImpl = checked_cast<BufferImpl*>(argumentBuffer.get());

    memcpy(argumentBufferImpl->m_buffer->contents(), shaderObject->m_data.data(), shaderObject->m_data.size());

    // Once the buffer is allocated, we can fill it in with the uniform data
    // and resource bindings we have tracked, using `argumentBufferTypeLayout` to obtain
    // the offsets for each field.
    //
    uint8_t* argumentData = (uint8_t*)argumentBufferImpl->m_buffer->contents();

    // Write all ordinary data early to prevent overwriting gpu-addresses
    // we write below.
    SLANG_RETURN_ON_FAIL(writeOrdinaryDataIntoArgumentBuffer(
        argumentBufferTypeLayout,
        shaderObject->getElementTypeLayout(),
        argumentData,
        (uint8_t*)shaderObject->m_data.data()
    ));

    for (uint32_t bindingRangeIndex = 0; bindingRangeIndex < specializedLayout->getBindingRangeCount();
         ++bindingRangeIndex)
    {
        const auto& bindingRangeInfo = specializedLayout->m_bindingRanges[bindingRangeIndex];
        uint32_t slotIndex = bindingRangeInfo.slotIndex;
        uint32_t count = bindingRangeInfo.count;

        SlangInt setIndex = argumentBufferTypeLayout->getBindingRangeDescriptorSetIndex(bindingRangeIndex);
        SlangInt rangeIndex = argumentBufferTypeLayout->getBindingRangeFirstDescriptorRangeIndex(bindingRangeIndex);
        SlangInt argumentOffset =
            argumentBufferTypeLayout->getDescriptorSetDescriptorRangeIndexOffset(setIndex, rangeIndex);
        uint8_t* argumentPtr = argumentData + argumentOffset;

        switch (bindingRangeInfo.bindingType)
        {
        case slang::BindingType::ConstantBuffer:
        case slang::BindingType::ParameterBlock:
        case slang::BindingType::ExistentialValue:
            break;

        case slang::BindingType::Texture:
        case slang::BindingType::MutableTexture:
            for (uint32_t i = 0; i < count; ++i)
            {
                const ResourceSlot& slot = shaderObject->m_slots[slotIndex + i];
                TextureViewImpl* textureView = checked_cast<TextureViewImpl*>(slot.resource.get());
                if (textureView)
                {
                    auto resourceId = textureView->m_textureView->gpuResourceID();
                    memcpy(argumentPtr + i * sizeof(uint64_t), &resourceId, sizeof(resourceId));
                    if (bindingRangeInfo.bindingType == slang::BindingType::MutableTexture)
                    {
                        SLANG_RETURN_ON_FAIL(addUsedRWResource(m_bindingData, textureView->m_textureView.get()));
                    }
                    else
                    {
                        SLANG_RETURN_ON_FAIL(addUsedResource(m_bindingData, textureView->m_textureView.get()));
                    }
                }
                else
                {
                    // No explicit binding — check for descriptor handle set via setDescriptorHandle()
                    uint64_t key = makeDescriptorHandleKey(bindingRangeIndex, i);
                    auto it = shaderObject->m_descriptorHandles.find(key);
                    if (it != shaderObject->m_descriptorHandles.end())
                    {
                        memcpy(argumentPtr + i * sizeof(uint64_t), &it->second, sizeof(uint64_t));
                    }
                }
            }
            break;
        case slang::BindingType::Sampler:
            for (uint32_t i = 0; i < count; ++i)
            {
                const ResourceSlot& slot = shaderObject->m_slots[slotIndex + i];
                SamplerImpl* sampler = checked_cast<SamplerImpl*>(slot.resource.get());
                if (sampler)
                {
                    auto resourceId = sampler->m_samplerState->gpuResourceID();
                    memcpy(argumentPtr + i * sizeof(uint64_t), &resourceId, sizeof(resourceId));
                }
                else
                {
                    // Check for descriptor handle set via setDescriptorHandle()
                    uint64_t key = makeDescriptorHandleKey(bindingRangeIndex, i);
                    auto it = shaderObject->m_descriptorHandles.find(key);
                    if (it != shaderObject->m_descriptorHandles.end())
                    {
                        memcpy(argumentPtr + i * sizeof(uint64_t), &it->second, sizeof(uint64_t));
                    }
                }
            }
            break;
        case slang::BindingType::RawBuffer:
        case slang::BindingType::MutableRawBuffer:
        case slang::BindingType::TypedBuffer:
        case slang::BindingType::MutableTypedBuffer:
        {
            for (uint32_t i = 0; i < count; ++i)
            {
                const ResourceSlot& slot = shaderObject->m_slots[slotIndex + i];
                BufferImpl* buffer = checked_cast<BufferImpl*>(slot.resource.get());
                if (buffer)
                {
                    DeviceAddress bufferPtr = buffer->getDeviceAddress() + slot.bufferRange.offset;
                    memcpy(argumentPtr + i * sizeof(uint64_t), &bufferPtr, sizeof(bufferPtr));
                    if (bindingRangeInfo.bindingType == slang::BindingType::MutableRawBuffer ||
                        bindingRangeInfo.bindingType == slang::BindingType::MutableTypedBuffer)
                    {
                        SLANG_RETURN_ON_FAIL(addUsedRWResource(m_bindingData, buffer->m_buffer.get()));
                    }
                    else
                    {
                        SLANG_RETURN_ON_FAIL(addUsedResource(m_bindingData, buffer->m_buffer.get()));
                    }
                }
                else
                {
                    // Check for descriptor handle set via setDescriptorHandle()
                    uint64_t key = makeDescriptorHandleKey(bindingRangeIndex, i);
                    auto it = shaderObject->m_descriptorHandles.find(key);
                    if (it != shaderObject->m_descriptorHandles.end())
                    {
                        memcpy(argumentPtr + i * sizeof(uint64_t), &it->second, sizeof(uint64_t));
                    }
                }
            }
        }
        break;
        case slang::BindingType::RayTracingAccelerationStructure:
            for (uint32_t i = 0; i < count; ++i)
            {
                const ResourceSlot& slot = shaderObject->m_slots[slotIndex + i];
                AccelerationStructureImpl* accelerationStructure =
                    checked_cast<AccelerationStructureImpl*>(slot.resource.get());
                if (accelerationStructure)
                {
                    auto resourceId = accelerationStructure->m_accelerationStructure->gpuResourceID();
                    memcpy(argumentPtr + i * sizeof(uint64_t), &resourceId, sizeof(resourceId));
                    SLANG_RETURN_ON_FAIL(
                        addUsedResource(m_bindingData, accelerationStructure->m_accelerationStructure.get())
                    );
                }
            }
            break;
        case slang::BindingType::VaryingInput:
        case slang::BindingType::VaryingOutput:
            break;

        default:
            SLANG_RHI_ASSERT_FAILURE("Unsupported binding type");
            return SLANG_FAIL;
            break;
        }
    }

    for (const auto& subObjectRange : specializedLayout->m_subObjectRanges)
    {
        auto subObjectLayout = subObjectRange.layout;
        const auto& bindingRange = specializedLayout->m_bindingRanges[subObjectRange.bindingRangeIndex];
        uint32_t count = bindingRange.count;
        uint32_t subObjectIndex = bindingRange.subObjectIndex;

        switch (bindingRange.bindingType)
        {
        case slang::BindingType::ParameterBlock:
        case slang::BindingType::ConstantBuffer:
        {
            SlangInt setIndex =
                argumentBufferTypeLayout->getBindingRangeDescriptorSetIndex(subObjectRange.bindingRangeIndex);
            SlangInt rangeIndex =
                argumentBufferTypeLayout->getBindingRangeFirstDescriptorRangeIndex(subObjectRange.bindingRangeIndex);
            SlangInt argumentOffset =
                argumentBufferTypeLayout->getDescriptorSetDescriptorRangeIndexOffset(setIndex, rangeIndex);
            uint8_t* argumentPtr = argumentData + argumentOffset;

            for (uint32_t i = 0; i < count; ++i)
            {
                auto subObject = shaderObject->m_objects[subObjectIndex + i];
                MTL::Buffer* subArgBuffer = nullptr;
                NS::UInteger subArgOffset = 0;
                SLANG_RETURN_ON_FAIL(writeArgumentBuffer(subObject, subObjectLayout, subArgBuffer, subArgOffset));
                if (!subArgBuffer)
                    continue;
                DeviceAddress bufferPtr = subArgBuffer->gpuAddress() + subArgOffset;
                memcpy(argumentPtr + i * sizeof(uint64_t), &bufferPtr, sizeof(bufferPtr));
                SLANG_RETURN_ON_FAIL(addUsedResource(m_bindingData, subArgBuffer));
            }
            break;
        }
        default:
            break;
        }
    }

    if (!m_device->m_hasUnifiedMemory)
        argumentBufferImpl->m_buffer->didModifyRange(NS::Range(0, argumentBufferDesc.size));

    outBuffer = argumentBufferImpl->m_buffer.get();
    outOffset = 0;

    // Keep BufferImpl alive for the frame via binding cache.
    // The queue's arg buffer cache will separately retain the MTL::Buffer for cross-frame persistence.
    m_bindingCache->buffers.push_back(argumentBufferImpl);

    return SLANG_OK;
}

Result BindingDataBuilder::writeOrdinaryDataIntoArgumentBuffer(
    slang::TypeLayoutReflection* argumentBufferTypeLayout,
    slang::TypeLayoutReflection* defaultTypeLayout,
    uint8_t* argumentBuffer,
    uint8_t* srcData
)
{
    // If we are pure data, just copy it over from srcData.
    if (defaultTypeLayout->getCategoryCount() == 1)
    {
        if (defaultTypeLayout->getCategoryByIndex(0) == slang::ParameterCategory::Uniform)
        {
            // Just write the uniform data.
            memcpy(argumentBuffer, srcData, defaultTypeLayout->getSize());
        }
        return SLANG_OK;
    }

    for (unsigned int i = 0; i < argumentBufferTypeLayout->getFieldCount(); i++)
    {
        auto argumentBufferField = argumentBufferTypeLayout->getFieldByIndex(i);
        auto defaultLayoutField = defaultTypeLayout->getFieldByIndex(i);
        // If the field is mixed type, recurse.
        SLANG_RETURN_ON_FAIL(writeOrdinaryDataIntoArgumentBuffer(
            argumentBufferField->getTypeLayout(),
            defaultLayoutField->getTypeLayout(),
            argumentBuffer + argumentBufferField->getOffset(),
            srcData + defaultLayoutField->getOffset()
        ));
    }
    return SLANG_OK;
}

} // namespace rhi::metal
