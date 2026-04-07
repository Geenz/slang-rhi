#include "metal-buffer.h"
#include "metal-bindless-descriptor-set.h"
#include "metal-device.h"
#include "metal-utils.h"

namespace rhi::metal {

BufferImpl::BufferImpl(Device* device, const BufferDesc& desc)
    : Buffer(device, desc)
{
}

BufferImpl::~BufferImpl() {}

void BufferImpl::deleteThis()
{
    getDevice<DeviceImpl>()->deferDelete(this);
}

Result BufferImpl::getNativeHandle(NativeHandle* outHandle)
{
    outHandle->type = NativeHandleType::MTLBuffer;
    outHandle->value = (uint64_t)m_buffer.get();
    return SLANG_OK;
}

Result BufferImpl::getSharedHandle(NativeHandle* outHandle)
{
    *outHandle = {};
    return SLANG_E_NOT_AVAILABLE;
}

DeviceAddress BufferImpl::getDeviceAddress()
{
    return m_buffer->gpuAddress();
}

Result BufferImpl::getDescriptorHandle(
    DescriptorHandleAccess access,
    Format format,
    BufferRange range,
    DescriptorHandle* outHandle)
{
    if (!outHandle)
        return SLANG_E_INVALID_ARG;

    DeviceImpl* device = getDevice<DeviceImpl>();
    if (!device->m_bindlessDescriptorSet)
        return SLANG_E_NOT_AVAILABLE;

    // Simple cache for whole-buffer handles (most common case)
    if (range.offset == 0 && range.size == kEntireBuffer.size)
    {
        uint32_t cacheIndex = (access == DescriptorHandleAccess::Read) ? 0 : 1;
        DescriptorHandle& cached = m_descriptorHandle[cacheIndex];
        if (cached)
        {
            *outHandle = cached;
            return SLANG_OK;
        }
        SLANG_RETURN_ON_FAIL(device->m_bindlessDescriptorSet->allocBufferHandle(this, access, format, range, outHandle));
        cached = *outHandle;
        return SLANG_OK;
    }

    // Non-cacheable (sub-range) -- allocate without caching
    return device->m_bindlessDescriptorSet->allocBufferHandle(this, access, format, range, outHandle);
}

Result DeviceImpl::createBuffer(const BufferDesc& desc_, const void* initData, IBuffer** outBuffer)
{
    AUTORELEASEPOOL

    BufferDesc desc = fixupBufferDesc(desc_);

    const Size bufferSize = desc.size;

    MTL::ResourceOptions resourceOptions = MTL::ResourceOptions(0);
    switch (desc.memoryType)
    {
    case MemoryType::DeviceLocal:
        resourceOptions = MTL::ResourceStorageModePrivate;
        break;
    case MemoryType::Upload:
    case MemoryType::ReadBack:
        resourceOptions = RHI_MTL_STAGING_STORAGE_MODE;
        break;
    }

    RefPtr<BufferImpl> buffer(new BufferImpl(this, desc));
    buffer->m_buffer = NS::TransferPtr(m_device->newBuffer(bufferSize, resourceOptions));
    if (!buffer->m_buffer)
    {
        return SLANG_FAIL;
    }

    if (desc.label)
        buffer->m_buffer->addDebugMarker(createString(desc.label).get(), NS::Range(0, desc.size));

    if (initData)
    {
        NS::SharedPtr<MTL::Buffer> stagingBuffer =
            NS::TransferPtr(m_device->newBuffer(initData, bufferSize, RHI_MTL_STAGING_STORAGE_MODE));
        MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
        MTL::BlitCommandEncoder* encoder = commandBuffer->blitCommandEncoder();
        if (!stagingBuffer || !commandBuffer || !encoder)
        {
            return SLANG_FAIL;
        }
        encoder->copyFromBuffer(stagingBuffer.get(), 0, buffer->m_buffer.get(), 0, bufferSize);
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
    }

    returnComPtr(outBuffer, buffer);
    return SLANG_OK;
}

Result DeviceImpl::createBufferFromNativeHandle(NativeHandle handle, const BufferDesc& desc, IBuffer** outBuffer)
{
    AUTORELEASEPOOL

    return SLANG_E_NOT_IMPLEMENTED;
}

Result DeviceImpl::mapBuffer(IBuffer* buffer, CpuAccessMode mode, void** outData)
{
    AUTORELEASEPOOL

    BufferImpl* bufferImpl = checked_cast<BufferImpl*>(buffer);
    bufferImpl->m_lastCpuAccessMode = mode;
    if (mode == CpuAccessMode::Read)
    {
#if TARGET_OS_OSX
        MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
        MTL::BlitCommandEncoder* encoder = commandBuffer->blitCommandEncoder();
        encoder->synchronizeResource(bufferImpl->m_buffer.get());
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
#endif
    }
    *outData = bufferImpl->m_buffer->contents();
    return SLANG_OK;
}

Result DeviceImpl::unmapBuffer(IBuffer* buffer)
{
    AUTORELEASEPOOL

    BufferImpl* bufferImpl = checked_cast<BufferImpl*>(buffer);
    if (bufferImpl->m_lastCpuAccessMode == CpuAccessMode::Write)
    {
#if TARGET_OS_OSX
        bufferImpl->m_buffer->didModifyRange(NS::Range(0, bufferImpl->m_desc.size));
#endif
#if TARGET_OS_OSX
        MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
        MTL::BlitCommandEncoder* encoder = commandBuffer->blitCommandEncoder();
        encoder->synchronizeResource(bufferImpl->m_buffer.get());
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
#endif
    }
    return SLANG_OK;
}

} // namespace rhi::metal
