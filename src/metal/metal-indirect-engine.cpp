#include "metal-indirect-engine.h"
#include "metal-utils.h"

#include <cmrc/cmrc.hpp>
CMRC_DECLARE(resources);

namespace rhi::metal {

Result IndirectEngine::initialize(MTL::Device* device)
{
    m_device = device;

    auto fs = cmrc::resources::get_filesystem();
    auto shader = fs.open("src/metal/shaders/encode-indirect.metal");

    auto source = createStringView((void*)shader.begin(), shader.size());

    NS::Error* error = nullptr;
    m_library = NS::TransferPtr(device->newLibrary(source.get(), nullptr, &error));
    if (error)
    {
        fprintf(stderr, "Metal error: %s\n", error->localizedDescription()->utf8String());
        return SLANG_FAIL;
    }

    // Create compute pipelines and argument encoders for each kernel.
    // Must use descriptor with supportIndirectCommandBuffers=true for GPU ICB encoding.
    auto createPipeline = [&](const char* name,
                              NS::SharedPtr<MTL::ComputePipelineState>& pipeline,
                              NS::SharedPtr<MTL::ArgumentEncoder>& argEncoder) -> Result
    {
        auto functionName = createString(name);
        auto function = NS::TransferPtr(m_library->newFunction(functionName.get()));
        if (!function)
        {
            fprintf(stderr, "Metal error: failed to find function '%s'\n", name);
            return SLANG_FAIL;
        }

        auto desc = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
        desc->setComputeFunction(function.get());
        desc->setSupportIndirectCommandBuffers(true);

        MTL::AutoreleasedComputePipelineReflection reflection = nullptr;
        pipeline = NS::TransferPtr(
            device->newComputePipelineState(desc.get(), MTL::PipelineOptionNone, &reflection, &error)
        );
        if (error)
        {
            fprintf(stderr, "Metal error: %s\n", error->localizedDescription()->utf8String());
            return SLANG_FAIL;
        }
        // buffer(0) is the ICBContainer argument buffer
        argEncoder = NS::TransferPtr(function->newArgumentEncoder(0));
        return SLANG_OK;
    };

    SLANG_RETURN_ON_FAIL(createPipeline("encode_draw_indirect", m_drawPipeline, m_drawArgEncoder));
    SLANG_RETURN_ON_FAIL(
        createPipeline("encode_draw_indexed_indirect_uint16", m_drawIndexedU16Pipeline, m_drawIndexedU16ArgEncoder)
    );
    SLANG_RETURN_ON_FAIL(
        createPipeline("encode_draw_indexed_indirect_uint32", m_drawIndexedU32Pipeline, m_drawIndexedU32ArgEncoder)
    );

    // Create shared range buffer (8 bytes: {uint32_t location, uint32_t length})
    m_rangeBuffer = NS::TransferPtr(device->newBuffer(sizeof(uint32_t) * 2, MTL::ResourceStorageModePrivate));

    return SLANG_OK;
}

void IndirectEngine::release()
{
    m_icbCache.clear();
    m_rangeBuffer.reset();
    m_drawPipeline.reset();
    m_drawIndexedU16Pipeline.reset();
    m_drawIndexedU32Pipeline.reset();
    m_drawArgEncoder.reset();
    m_drawIndexedU16ArgEncoder.reset();
    m_drawIndexedU32ArgEncoder.reset();
    m_library.reset();
    m_device = nullptr;
}

IndirectEngine::ICBCacheEntry IndirectEngine::createICBEntry(
    MTL::IndirectCommandType commandType,
    uint32_t maxDrawCount,
    MTL::ArgumentEncoder* argEncoder
)
{
    auto desc = NS::TransferPtr(MTL::IndirectCommandBufferDescriptor::alloc()->init());
    desc->setCommandTypes(commandType);
    desc->setInheritPipelineState(true);
    desc->setInheritBuffers(true);
    desc->setMaxVertexBufferBindCount(0);
    desc->setMaxFragmentBufferBindCount(0);

    auto icb = NS::TransferPtr(m_device->newIndirectCommandBuffer(desc.get(), maxDrawCount, MTL::ResourceStorageModePrivate));

    // Create argument buffer encoding the ICB for compute access
    auto argBuf = NS::TransferPtr(m_device->newBuffer(argEncoder->encodedLength(), MTL::ResourceStorageModeShared));
    argEncoder->setArgumentBuffer(argBuf.get(), 0);
    argEncoder->setIndirectCommandBuffer(icb.get(), 0); // id(0) in ICBContainer

    ICBCacheEntry entry;
    entry.commandType = commandType;
    entry.maxDrawCount = maxDrawCount;
    entry.icb = icb;
    entry.argBuffer = argBuf;
    return entry;
}

IndirectEngine::ICBResult IndirectEngine::getOrCreateICB(MTL::IndirectCommandType commandType, uint32_t maxDrawCount)
{
    // Look for existing entry
    for (auto& entry : m_icbCache)
    {
        if (entry.commandType == commandType && entry.maxDrawCount >= maxDrawCount)
        {
            return {entry.icb.get(), entry.argBuffer.get()};
        }
    }

    // Determine which argument encoder to use
    MTL::ArgumentEncoder* argEncoder;
    if (commandType == MTL::IndirectCommandTypeDraw)
    {
        argEncoder = m_drawArgEncoder.get();
    }
    else
    {
        // For DrawIndexed, use uint32 arg encoder (they have the same layout for the ICB container)
        argEncoder = m_drawIndexedU32ArgEncoder.get();
    }

    m_icbCache.push_back(createICBEntry(commandType, maxDrawCount, argEncoder));
    auto& entry = m_icbCache.back();
    return {entry.icb.get(), entry.argBuffer.get()};
}

void IndirectEngine::encodeDraw(
    MTL::ComputeCommandEncoder* encoder,
    MTL::Buffer* argBuffer,
    NS::UInteger argOffset,
    MTL::Buffer* countBuffer,
    NS::UInteger countOffset,
    uint32_t maxDrawCount,
    MTL::PrimitiveType primitiveType,
    MTL::Buffer* icbArgBuffer,
    MTL::Buffer* rangeBuffer
)
{
    encoder->setComputePipelineState(m_drawPipeline.get());

    // buffer(0): ICB argument buffer
    encoder->setBuffer(icbArgBuffer, 0, 0);
    // buffer(1): draw arguments (with offset)
    encoder->setBuffer(argBuffer, argOffset, 1);
    // buffer(2): count buffer (with offset)
    encoder->setBuffer(countBuffer, countOffset, 2);
    // buffer(3): range output buffer
    encoder->setBuffer(rangeBuffer, 0, 3);
    // buffer(4): params (inline via setBytes)
    struct
    {
        uint32_t maxDrawCount;
        uint32_t primitiveType;
    } params = {maxDrawCount, static_cast<uint32_t>(primitiveType)};
    encoder->setBytes(&params, sizeof(params), 4);

    // Dispatch one thread per potential draw
    NS::UInteger threadGroupSize = m_drawPipeline->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > maxDrawCount)
        threadGroupSize = maxDrawCount;
    encoder->dispatchThreads(MTL::Size{maxDrawCount, 1, 1}, MTL::Size{threadGroupSize, 1, 1});
}

void IndirectEngine::encodeDrawIndexed(
    MTL::ComputeCommandEncoder* encoder,
    MTL::Buffer* argBuffer,
    NS::UInteger argOffset,
    MTL::Buffer* countBuffer,
    NS::UInteger countOffset,
    uint32_t maxDrawCount,
    MTL::PrimitiveType primitiveType,
    MTL::Buffer* indexBuffer,
    NS::UInteger indexBufferOffset,
    MTL::IndexType indexType,
    MTL::Buffer* icbArgBuffer,
    MTL::Buffer* rangeBuffer
)
{
    MTL::ComputePipelineState* pipeline =
        (indexType == MTL::IndexTypeUInt16) ? m_drawIndexedU16Pipeline.get() : m_drawIndexedU32Pipeline.get();

    encoder->setComputePipelineState(pipeline);

    // buffer(0): ICB argument buffer
    encoder->setBuffer(icbArgBuffer, 0, 0);
    // buffer(1): draw indexed arguments (with offset)
    encoder->setBuffer(argBuffer, argOffset, 1);
    // buffer(2): count buffer (with offset)
    encoder->setBuffer(countBuffer, countOffset, 2);
    // buffer(3): range output buffer
    encoder->setBuffer(rangeBuffer, 0, 3);
    // buffer(4): index buffer (with offset)
    encoder->setBuffer(indexBuffer, indexBufferOffset, 4);
    // buffer(5): params (inline via setBytes)
    struct
    {
        uint32_t maxDrawCount;
        uint32_t primitiveType;
    } params = {maxDrawCount, static_cast<uint32_t>(primitiveType)};
    encoder->setBytes(&params, sizeof(params), 5);

    // Dispatch one thread per potential draw
    NS::UInteger threadGroupSize = pipeline->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > maxDrawCount)
        threadGroupSize = maxDrawCount;
    encoder->dispatchThreads(MTL::Size{maxDrawCount, 1, 1}, MTL::Size{threadGroupSize, 1, 1});
}

} // namespace rhi::metal
