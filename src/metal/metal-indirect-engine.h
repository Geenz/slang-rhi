#pragma once

#include "metal-base.h"

#include <vector>

namespace rhi::metal {

/// Metal doesn't have a drawIndirectCount equivalent.
/// This class uses Metal Indirect Command Buffers (ICBs) to support GPU-driven draw counts.
/// A compute pre-pass populates the ICB commands from the arg buffer and writes
/// the execution range from the count buffer.
class IndirectEngine
{
public:
    Result initialize(MTL::Device* device);
    void release();

    struct ICBResult
    {
        MTL::IndirectCommandBuffer* icb;
        MTL::Buffer* argBuffer;
    };

    /// Get or create an ICB for the given command type and max draw count.
    ICBResult getOrCreateICB(MTL::IndirectCommandType commandType, uint32_t maxDrawCount);

    /// Get the shared range buffer (8 bytes: {uint32_t location, uint32_t length}).
    MTL::Buffer* getRangeBuffer() { return m_rangeBuffer.get(); }

    /// Populate ICB from arg buffer, write execution range from count buffer.
    void encodeDraw(
        MTL::ComputeCommandEncoder* encoder,
        MTL::Buffer* argBuffer,
        NS::UInteger argOffset,
        MTL::Buffer* countBuffer,
        NS::UInteger countOffset,
        uint32_t maxDrawCount,
        MTL::PrimitiveType primitiveType,
        MTL::Buffer* icbArgBuffer,
        MTL::Buffer* rangeBuffer
    );

    void encodeDrawIndexed(
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
    );

private:
    MTL::Device* m_device = nullptr;
    NS::SharedPtr<MTL::Library> m_library;

    // 3 compute pipelines: draw, draw_indexed_uint16, draw_indexed_uint32
    NS::SharedPtr<MTL::ComputePipelineState> m_drawPipeline;
    NS::SharedPtr<MTL::ComputePipelineState> m_drawIndexedU16Pipeline;
    NS::SharedPtr<MTL::ComputePipelineState> m_drawIndexedU32Pipeline;

    // Argument encoders for each kernel's ICB argument buffer parameter.
    NS::SharedPtr<MTL::ArgumentEncoder> m_drawArgEncoder;
    NS::SharedPtr<MTL::ArgumentEncoder> m_drawIndexedU16ArgEncoder;
    NS::SharedPtr<MTL::ArgumentEncoder> m_drawIndexedU32ArgEncoder;

    // ICB cache: keyed by (commandType, maxDrawCount).
    struct ICBCacheEntry
    {
        MTL::IndirectCommandType commandType;
        uint32_t maxDrawCount;
        NS::SharedPtr<MTL::IndirectCommandBuffer> icb;
        NS::SharedPtr<MTL::Buffer> argBuffer; // argument buffer encoding the ICB for compute
    };
    std::vector<ICBCacheEntry> m_icbCache;

    // Shared range buffer (8 bytes: {uint32_t location, uint32_t length})
    NS::SharedPtr<MTL::Buffer> m_rangeBuffer;

    // Helper to create ICB + its argument buffer
    ICBCacheEntry createICBEntry(
        MTL::IndirectCommandType commandType,
        uint32_t maxDrawCount,
        MTL::ArgumentEncoder* argEncoder
    );
};

} // namespace rhi::metal
