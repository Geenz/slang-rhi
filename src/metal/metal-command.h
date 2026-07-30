#pragma once

#include "metal-base.h"
#include "metal-shader-object.h"
#include "metal-constant-buffer-pool.h"

#include "core/ring-queue.h"

namespace rhi::metal {

class CommandQueueImpl : public CommandQueue
{
public:
    NS::SharedPtr<MTL::CommandQueue> m_commandQueue;
    NS::SharedPtr<MTL::SharedEvent> m_trackingEvent;
    NS::SharedPtr<MTL::SharedEventListener> m_trackingEventListener;
    uint64_t m_lastSubmittedID;
    uint64_t m_lastFinishedID;
    std::list<RefPtr<CommandBufferImpl>> m_commandBuffersInFlight;

    // Deferred delete queue for GPU resources.
    // Resources are held here until the GPU has finished using them.
    struct DeferredDelete
    {
        uint64_t submissionID;
        Resource* resource;
    };
    std::mutex m_deferredDeleteQueueMutex;
    RingQueue<DeferredDelete> m_deferredDeleteQueue;

    // Persistent argument buffer cache — survives across CommandEncoder lifetimes.
    // Keyed by ShaderObject pointer + version; holds ownership of the MTL::Buffer.
    // Guarded by m_argCacheMutex: multiple camera worker threads create
    // CommandEncoders on the same queue and call getBindingData() concurrently
    // during recording, which reads/writes this vector.
    struct CachedArgBuffer
    {
        ShaderObject* object = nullptr;
        uint32_t version = 0;
        NS::SharedPtr<MTL::Buffer> buffer;  // Owns the buffer
        std::vector<MTL::Resource*> usedResources;    // Cached for residency replay
        std::vector<MTL::Resource*> usedRWResources;
    };
    std::mutex m_argCacheMutex;
    std::vector<CachedArgBuffer> m_cachedArgBuffers;

    CommandQueueImpl(Device* device, QueueType type);
    ~CommandQueueImpl();

    void init(NS::SharedPtr<MTL::CommandQueue> commandQueue);
    void shutdown();

    void retireCommandBuffers();
    uint64_t updateLastFinishedID();

    /// Queue a resource for deferred deletion. The resource will be deleted
    /// once the GPU has finished all work submitted up to this point.
    void deferDelete(Resource* resource);

    /// Delete deferred resources that are no longer in use by the GPU.
    void executeDeferredDeletes();

    // ICommandQueue implementation
    virtual SLANG_NO_THROW Result SLANG_MCALL createCommandEncoder(
        const CommandEncoderDesc& desc,
        ICommandEncoder** outEncoder
    ) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL submit(const SubmitDesc& desc) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL waitOnHost() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(NativeHandle* outHandle) override;
};

class CommandEncoderImpl : public CommandEncoder
{
public:
    CommandQueueImpl* m_queue;
    RefPtr<CommandBufferImpl> m_commandBuffer;

    // Incremental binding state
    RootShaderObject* m_lastTrackedRootObject = nullptr;
    BindingDataImpl* m_previousBindingData = nullptr;
    std::vector<SubObjectVersionEntry> m_versionSnapA;
    std::vector<SubObjectVersionEntry> m_versionSnapB;
    bool m_versionSnapFlip = false;

    // Argument buffer cache lives on CommandQueueImpl for cross-frame persistence.

    CommandEncoderImpl(Device* device, CommandQueueImpl* queue, const CommandEncoderDesc& desc);
    ~CommandEncoderImpl();

    Result init();

    virtual Result getBindingData(RootShaderObject* rootObject, BindingData*& outBindingData) override;

    // ICommandEncoder implementation
    virtual SLANG_NO_THROW Result SLANG_MCALL finish(
        const CommandBufferDesc& desc,
        ICommandBuffer** outCommandBuffer
    ) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(NativeHandle* outHandle) override;
};

class CommandBufferImpl : public CommandBuffer
{
public:
    CommandQueueImpl* m_queue;
    NS::SharedPtr<MTL::CommandBuffer> m_commandBuffer;
    BindingCache m_bindingCache;
    ConstantBufferPool m_constantBufferPool;
    uint64_t m_submissionID;

    /// Strong refs to argument buffers resolved into this CB's binding data.
    /// Parallels the base CommandBuffer::m_trackedObjects, but pins the argument
    /// buffer OBJECT itself (m_trackedObjects only pins its slot contents). The
    /// cached arg buffer's only other owner is CommandQueueImpl's cross-frame
    /// cache, which drops its ref at submit() (m_cachedArgBuffers.clear()) and on
    /// version bumps — both BEFORE GPU completion, while this in-flight CB still
    /// references the raw MTL::Buffer* in its (deferred-encoded) binding data.
    /// Held until reset() releases them, which only runs after the CB completes.
    std::vector<NS::SharedPtr<MTL::Buffer>> m_trackedArgBuffers;

    CommandBufferImpl(Device* device, CommandQueueImpl* queue);
    ~CommandBufferImpl();

    Result init();
    virtual Result reset() override;

    // ICommandBuffer implementation
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(NativeHandle* outHandle) override;
};

} // namespace rhi::metal
