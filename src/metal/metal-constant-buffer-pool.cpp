#include "metal-constant-buffer-pool.h"
#include "metal-device.h"
#include "metal-buffer.h"

#include <cstdio>

namespace rhi::metal {

inline size_t alignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

void ConstantBufferPool::init(DeviceImpl* device)
{
    m_device = device;
}

void ConstantBufferPool::finish()
{
    m_pages.clear();
}

void ConstantBufferPool::reset()
{
    if (m_allocationCount > 0)
    {
        fprintf(stderr, "[slang-rhi] ConstantBufferPool: %u allocations across %d pages\n",
                m_allocationCount, m_currentPage + 1);
    }
    m_allocationCount = 0;
    m_currentPage = -1;
    m_currentOffset = 0;
}

Result ConstantBufferPool::allocate(size_t size, Allocation& outAllocation)
{
    if (size > kPageSize)
    {
        return SLANG_FAIL;
    }

    if (m_currentPage == -1 || m_currentOffset + size > kPageSize)
    {
        m_currentPage += 1;
        if (m_currentPage >= int(m_pages.size()))
        {
            m_pages.push_back(Page());
            SLANG_RETURN_ON_FAIL(createPage(kPageSize, m_pages.back()));
        }
        m_currentOffset = 0;
    }

    const Page& page = m_pages[m_currentPage];
    outAllocation.buffer = page.buffer;
    outAllocation.offset = m_currentOffset;
    outAllocation.mappedData = page.mappedData + m_currentOffset;
    m_currentOffset = alignUp(m_currentOffset + size, kAlignment);
    m_allocationCount++;
    return SLANG_OK;
}

Result ConstantBufferPool::createPage(size_t size, Page& outPage)
{
    ComPtr<IBuffer> buffer;
    BufferDesc bufferDesc;
    bufferDesc.usage = BufferUsage::ConstantBuffer | BufferUsage::CopyDestination;
    bufferDesc.defaultState = ResourceState::ConstantBuffer;
    bufferDesc.memoryType = MemoryType::Upload;
    bufferDesc.size = size;
    bufferDesc.label = "ConstantBufferPool_Page";
    SLANG_RETURN_ON_FAIL(m_device->createBuffer(bufferDesc, nullptr, buffer.writeRef()));

    outPage.size = size;
    outPage.buffer = checked_cast<BufferImpl*>(buffer.get());
    outPage.buffer->breakStrongReferenceToDevice();
    outPage.mappedData = static_cast<uint8_t*>(outPage.buffer->m_buffer->contents());
    return SLANG_OK;
}

} // namespace rhi::metal
