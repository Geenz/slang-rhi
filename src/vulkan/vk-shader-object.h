#pragma once

#include "vk-base.h"
#include "vk-shader-object-layout.h"
#include "../transient-buffer-heap.h"

#include "core/short_vector.h"

#include <set>
#include <unordered_map>
#include <vector>

namespace rhi::vk {

/// A push constant range plus the object whose ordinary data fills it, so a replayed root binding
/// can re-upload the per-draw contents without walking the object tree again.
struct PushConstantSource
{
    VkPushConstantRange range;
    ShaderObject* object;
    ShaderObjectLayoutImpl* layout;
};

struct BindingDataBuilder
{
    DeviceImpl* m_device;
    ArenaAllocator* m_allocator;
    BindingCache* m_bindingCache;
    BindingDataImpl* m_bindingData;
    TransientBufferArena* m_constantBufferArena;
    DescriptorSetAllocator* m_descriptorSetAllocator;
    CommandQueueImpl* m_queue;
    /// The command buffer's retain set. bindAsValue pins every resource it writes into a descriptor.
    std::set<RefPtr<RefObject>>* m_trackedObjects;

    // TODO remove
    std::span<const VkPushConstantRange> m_pushConstantRanges;

    /// Scratch storage for writing a whole binding range in one vkUpdateDescriptorSets call.
    short_vector<VkDescriptorImageInfo, 16> m_imageInfos;
    short_vector<VkDescriptorBufferInfo, 16> m_bufferInfos;
    short_vector<VkBufferView, 16> m_bufferViews;
    short_vector<VkAccelerationStructureKHR, 16> m_accelerationStructures;

    /// Push constants emitted by the root build in progress.
    short_vector<PushConstantSource, 8> m_pushConstantSources;


    /// Bind this object as a root shader object
    Result bindAsRoot(
        RootShaderObject* shaderObject,
        RootShaderObjectLayoutImpl* specializedLayout,
        BindingDataImpl*& outBindingData
    );

    /// Bind this shader object as an entry point
    Result bindAsEntryPoint(
        ShaderObject* shaderObject,
        const BindingOffset& inOffset,
        EntryPointLayout* specializedLayout,
        uint32_t entryPointIndex
    );

    /// Bind this object as a `PushConstantBuffer<X>`.
    Result bindAsPushConstantBuffer(
        ShaderObject* shaderObject,
        const BindingOffset& inOffset,
        ShaderObjectLayoutImpl* specializedLayout
    );

    /// Bind the ordinary data buffer if needed.
    Result bindOrdinaryDataBufferIfNeeded(
        ShaderObject* shaderObject,
        BindingOffset& ioOffset,
        ShaderObjectLayoutImpl* specializedLayout
    );

    /// Bind this shader object as a "value"
    ///
    /// This is the mode used for binding sub-objects for existential-type
    /// fields, and is also used as part of the implementation of the
    /// parameter-block and constant-buffer cases.
    ///
    Result bindAsValue(
        ShaderObject* shaderObject,
        const BindingOffset& offset,
        ShaderObjectLayoutImpl* specializedLayout
    );

    /// Allocate the descriptor sets needed for binding this object (but not nested parameter
    /// blocks)
    Result allocateDescriptorSets(
        ShaderObject* shaderObject,
        const BindingOffset& offset,
        ShaderObjectLayoutImpl* specializedLayout
    );

    /// Bind this object as a `ParameterBlock<X>`.
    Result bindAsParameterBlock(
        ShaderObject* shaderObject,
        const BindingOffset& inOffset,
        ShaderObjectLayoutImpl* specializedLayout
    );

    /// Bind an eligible parameter block through the queue's shared set cache.
    /// Sets outHandled to false when the caller must build the block itself.
    Result bindSharedParameterBlock(
        ShaderObject* shaderObject,
        const BindingOffset& offset,
        ShaderObjectLayoutImpl* specializedLayout,
        bool& outHandled
    );

    /// Replay a cached parameter block, if one was built for this object/layout/version
    /// earlier in the current command buffer. Returns false if the caller must build it.
    bool reuseParameterBlock(ShaderObject* shaderObject, ShaderObjectLayoutImpl* specializedLayout, uint64_t version);

    /// Record the descriptor sets and resource states a parameter block just produced.
    void storeParameterBlock(
        ShaderObject* shaderObject,
        ShaderObjectLayoutImpl* specializedLayout,
        uint64_t version,
        uint32_t firstDescriptorSet,
        uint32_t firstBufferState,
        uint32_t firstTextureState,
        uint32_t firstCachedBlock
    );

    /// Replay a root shader object's descriptor sets, resource states and push constants from an
    /// earlier draw in the current command buffer. Returns false if the caller must build them.
    bool reuseRootBinding(
        ShaderObject* shaderObject,
        ShaderObjectLayoutImpl* specializedLayout,
        uint64_t key,
        VkBuffer ordinaryDataBuffer
    );

    /// Record what a root build just produced so a later draw can replay it.
    void storeRootBinding(
        ShaderObject* shaderObject,
        ShaderObjectLayoutImpl* specializedLayout,
        uint64_t key,
        VkBuffer ordinaryDataBuffer
    );

    /// Bind this object as a `ConstantBuffer<X>`.
    Result bindAsConstantBuffer(
        ShaderObject* shaderObject,
        const BindingOffset& inOffset,
        ShaderObjectLayoutImpl* specializedLayout
    );
};

struct BindingDataImpl : BindingData
{
public:
    struct BufferState
    {
        BufferImpl* buffer;
        ResourceState state;
    };
    struct TextureState
    {
        TextureViewImpl* textureView;
        ResourceState state;
    };
    /// Entry point data for copying to shader binding table (ray tracing)
    struct EntryPointData
    {
        // Host memory pointer to entry point uniform data
        void* data;
        // Size of the data in bytess
        size_t size;
    };

    /// Required buffer states.
    BufferState* bufferStates;
    uint32_t bufferStateCapacity;
    uint32_t bufferStateCount;
    /// Required texture states.
    TextureState* textureStates;
    uint32_t textureStateCapacity;
    uint32_t textureStateCount;
    /// Queue-cached parameter blocks this binding uses; their states are required from the block itself.
    CachedParameterBlock** cachedBlocks;
    uint32_t cachedBlockCapacity;
    uint32_t cachedBlockCount;

    /// Pipeline layout.
    VkPipelineLayout pipelineLayout;

    /// Descriptor sets.
    VkDescriptorSet* descriptorSets;
    uint32_t descriptorSetCount;

    /// Dynamic offsets for the bound sets, in set then binding order. Only the root's
    /// ordinary-data buffer is dynamic, so there is at most one.
    uint32_t dynamicOffsets[1];
    uint32_t dynamicOffsetCount;

    /// Push constants.
    VkPushConstantRange* pushConstantRanges;
    void** pushConstantData;
    uint32_t pushConstantCount;

    /// Entry point data (for ray tracing SBT).
    EntryPointData* entryPointData;
    uint32_t entryPointCount;
};

/// Descriptor sets a `ParameterBlock<X>` sub-object produced, keyed by its composite version.
/// Valid only for the command buffer that built them: they, the arena they reference and this
/// cache are all reset together in `CommandBufferImpl::reset()`.
struct ParameterBlockCacheEntry
{
    ShaderObject* object;
    ShaderObjectLayoutImpl* layout;
    uint64_t version;
    VkDescriptorSet* descriptorSets;
    uint32_t descriptorSetCount;
    BindingDataImpl::BufferState* bufferStates;
    uint32_t bufferStateCount;
    BindingDataImpl::TextureState* textureStates;
    uint32_t textureStateCount;
    CachedParameterBlock** cachedBlocks;
    uint32_t cachedBlockCount;
};

/// Everything a root shader object contributed to its binding data, replayable while nothing but
/// its ordinary data has changed. Same command-buffer lifetime as ParameterBlockCacheEntry.
struct RootBindingCacheEntry
{
    ShaderObjectLayoutImpl* layout;
    uint64_t key;
    /// Arena page backing the dynamic ordinary-data descriptor; a new page invalidates the entry.
    VkBuffer ordinaryDataBuffer;
    VkDescriptorSet* descriptorSets;
    uint32_t descriptorSetCount;
    BindingDataImpl::BufferState* bufferStates;
    uint32_t bufferStateCount;
    BindingDataImpl::TextureState* textureStates;
    uint32_t textureStateCount;
    CachedParameterBlock** cachedBlocks;
    uint32_t cachedBlockCount;
    PushConstantSource* pushConstants;
    uint32_t pushConstantCount;
};

/// A parameter block's descriptor set shared across command buffers and frames.
/// Each command buffer that binds it holds a reference until that command buffer retires.
class CachedParameterBlock : public RefObject
{
public:
    CommandQueueImpl* m_queue = nullptr;
    RefPtr<ShaderObjectLayoutImpl> m_layout;
    VulkanDescriptorSet m_descriptorSet = {};
    std::vector<RefPtr<Resource>> m_resources;
    std::vector<BindingDataImpl::BufferState> m_bufferStates;
    std::vector<BindingDataImpl::TextureState> m_textureStates;
    uint64_t m_lastUsedID = 0;

    ~CachedParameterBlock();
};

/// A shader object's slot-content hash at one version, so an unchanged block is not rehashed.
struct ParameterBlockContent
{
    uint32_t uid;
    uint32_t version;
    uint64_t contentHash;
    uint64_t lastUsedID;
};

struct BindingCache
{
    std::vector<BindingDataImpl*> bindingData;
    std::vector<ParameterBlockCacheEntry> parameterBlocks;
    std::unordered_map<ShaderObject*, RootBindingCacheEntry> rootBindings;

    void reset()
    {
        bindingData.clear();
        parameterBlocks.clear();
        rootBindings.clear();
    }
};

} // namespace rhi::vk
