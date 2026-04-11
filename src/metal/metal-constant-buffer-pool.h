#pragma once

#include "metal-base.h"

#include <vector>

namespace rhi::metal {

class ConstantBufferPool
{
public:
    struct Allocation
    {
        BufferImpl* buffer;
        size_t offset;
        void* mappedData;
    };

    void init(DeviceImpl* device);
    void finish();
    void reset();

    Result allocate(size_t size, Allocation& outAllocation);

private:
    static constexpr size_t kAlignment = 256;
    static constexpr size_t kPageSize = 4 * 1024 * 1024;

    struct Page
    {
        RefPtr<BufferImpl> buffer;
        size_t size = 0;
        uint8_t* mappedData = nullptr;
    };

    DeviceImpl* m_device = nullptr;

    std::vector<Page> m_pages;

    int m_currentPage = -1;
    size_t m_currentOffset = 0;
    uint32_t m_allocationCount = 0;

    Result createPage(size_t size, Page& outPage);
};

} // namespace rhi::metal
