#include "testing.h"

#include <array>
#include <thread>

using namespace rhi;
using namespace rhi::testing;

static constexpr uint32_t kQueueThreadCount = 4;
static constexpr uint32_t kQueueIterationCount = 32;
static constexpr uint32_t kQueueElementCount = 64;

using QueueBufferData = std::array<uint32_t, kQueueElementCount>;

static QueueBufferData makeQueueBufferData(uint32_t threadIndex, uint32_t iteration, uint32_t salt)
{
    QueueBufferData data;
    for (uint32_t element = 0; element < kQueueElementCount; ++element)
    {
        data[element] = (salt << 28) | (threadIndex << 20) | (iteration << 8) | element;
    }
    return data;
}

static bool runQueueWorker(
    IDevice* device,
    ICommandQueue* queue,
    uint32_t threadIndex,
    ComPtr<IBuffer>& outInitBuffer,
    ComPtr<IBuffer>& outUploadBuffer
)
{
    BufferDesc uploadDesc = {};
    uploadDesc.size = sizeof(QueueBufferData);
    uploadDesc.usage = BufferUsage::CopyDestination | BufferUsage::CopySource;
    if (SLANG_FAILED(device->createBuffer(uploadDesc, nullptr, outUploadBuffer.writeRef())))
    {
        return false;
    }

    for (uint32_t iteration = 0; iteration < kQueueIterationCount; ++iteration)
    {
        TextureDesc textureDesc = {};
        textureDesc.type = TextureType::Texture2D;
        textureDesc.size = {4, 4, 1};
        textureDesc.format = Format::RGBA8Unorm;
        textureDesc.usage = TextureUsage::ShaderResource | TextureUsage::CopyDestination;
        textureDesc.defaultState = ResourceState::ShaderResource;
        std::array<uint32_t, 16> texels;
        texels.fill(threadIndex * kQueueIterationCount + iteration);
        SubresourceData texelData = {texels.data(), 4 * sizeof(uint32_t), 0};
        ComPtr<ITexture> texture;
        if (SLANG_FAILED(device->createTexture(textureDesc, &texelData, texture.writeRef())))
        {
            return false;
        }

        QueueBufferData initData = makeQueueBufferData(threadIndex, iteration, 1);
        BufferDesc initDesc = {};
        initDesc.size = sizeof(QueueBufferData);
        initDesc.memoryType = MemoryType::DeviceLocal;
        initDesc.usage = BufferUsage::ShaderResource | BufferUsage::CopySource;
        initDesc.defaultState = ResourceState::ShaderResource;
        if (SLANG_FAILED(device->createBuffer(initDesc, initData.data(), outInitBuffer.writeRef())))
        {
            return false;
        }

        QueueBufferData uploadData = makeQueueBufferData(threadIndex, iteration, 2);
        ComPtr<ICommandEncoder> encoder;
        if (SLANG_FAILED(queue->createCommandEncoder(encoder.writeRef())))
        {
            return false;
        }

        if (SLANG_FAILED(encoder->uploadBufferData(outUploadBuffer, 0, sizeof(QueueBufferData), uploadData.data())))
        {
            return false;
        }

        ComPtr<ICommandBuffer> commandBuffer;
        if (SLANG_FAILED(encoder->finish(commandBuffer.writeRef())))
        {
            return false;
        }

        if (SLANG_FAILED(queue->submit(commandBuffer)))
        {
            return false;
        }

        if (iteration % 8 == 7)
        {
            if (SLANG_FAILED(queue->waitOnHost()))
            {
                return false;
            }
        }
    }

    return true;
}

GPU_TEST_CASE("queue-concurrency", Vulkan | DontCacheDevice)
{
    ComPtr<ICommandQueue> queue = device->getQueue(QueueType::Graphics);

    std::array<ComPtr<IBuffer>, kQueueThreadCount> initBuffers;
    std::array<ComPtr<IBuffer>, kQueueThreadCount> uploadBuffers;
    std::array<bool, kQueueThreadCount> succeeded = {};
    std::array<std::thread, kQueueThreadCount> threads;
    for (uint32_t threadIndex = 0; threadIndex < kQueueThreadCount; ++threadIndex)
    {
        threads[threadIndex] = std::thread(
            [&, threadIndex]()
            {
                succeeded[threadIndex] = runQueueWorker(
                    device.get(),
                    queue.get(),
                    threadIndex,
                    initBuffers[threadIndex],
                    uploadBuffers[threadIndex]
                );
            }
        );
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    REQUIRE_CALL(queue->waitOnHost());

    for (uint32_t threadIndex = 0; threadIndex < kQueueThreadCount; ++threadIndex)
    {
        REQUIRE(succeeded[threadIndex]);
        compareComputeResult(
            device,
            initBuffers[threadIndex],
            makeQueueBufferData(threadIndex, kQueueIterationCount - 1, 1)
        );
        compareComputeResult(
            device,
            uploadBuffers[threadIndex],
            makeQueueBufferData(threadIndex, kQueueIterationCount - 1, 2)
        );
    }
}
