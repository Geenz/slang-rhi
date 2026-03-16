#include <metal_stdlib>
using namespace metal;

// Argument buffer containing the ICB — required pattern for GPU ICB access.
// MSL command_buffer type can only appear inside an argument buffer struct with [[ id(N) ]].
struct ICBContainer {
    command_buffer commandBuffer [[ id(0) ]];
};

struct DrawArgs {
    uint vertexCountPerInstance;
    uint instanceCount;
    uint startVertexLocation;
    uint startInstanceLocation;
};

struct DrawIndexedArgs {
    uint indexCountPerInstance;
    uint instanceCount;
    uint startIndexLocation;
    int  baseVertexLocation;
    uint startInstanceLocation;
};

struct Params {
    uint maxDrawCount;
    uint primitiveType;
};

kernel void encode_draw_indirect(
    device ICBContainer* icb_container  [[buffer(0)]],
    const device DrawArgs* args         [[buffer(1)]],
    const device uint* countPtr         [[buffer(2)]],
    device uint2* rangeOut              [[buffer(3)]],
    constant Params& params             [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint count = min(*countPtr, params.maxDrawCount);
    if (tid == 0) {
        *rangeOut = uint2(0, count);
    }
    if (tid < count) {
        render_command cmd(icb_container->commandBuffer, tid);
        cmd.draw_primitives(
            (primitive_type)params.primitiveType,
            args[tid].startVertexLocation,
            args[tid].vertexCountPerInstance,
            args[tid].instanceCount,
            args[tid].startInstanceLocation
        );
    }
}

kernel void encode_draw_indexed_indirect_uint32(
    device ICBContainer* icb_container   [[buffer(0)]],
    const device DrawIndexedArgs* args   [[buffer(1)]],
    const device uint* countPtr          [[buffer(2)]],
    device uint2* rangeOut               [[buffer(3)]],
    const device uint* indexBuffer       [[buffer(4)]],
    constant Params& params              [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint count = min(*countPtr, params.maxDrawCount);
    if (tid == 0) {
        *rangeOut = uint2(0, count);
    }
    if (tid < count) {
        render_command cmd(icb_container->commandBuffer, tid);
        cmd.draw_indexed_primitives(
            (primitive_type)params.primitiveType,
            args[tid].indexCountPerInstance,
            indexBuffer + args[tid].startIndexLocation,
            args[tid].instanceCount,
            args[tid].baseVertexLocation,
            args[tid].startInstanceLocation
        );
    }
}

kernel void encode_draw_indexed_indirect_uint16(
    device ICBContainer* icb_container   [[buffer(0)]],
    const device DrawIndexedArgs* args   [[buffer(1)]],
    const device uint* countPtr          [[buffer(2)]],
    device uint2* rangeOut               [[buffer(3)]],
    const device ushort* indexBuffer     [[buffer(4)]],
    constant Params& params              [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint count = min(*countPtr, params.maxDrawCount);
    if (tid == 0) {
        *rangeOut = uint2(0, count);
    }
    if (tid < count) {
        render_command cmd(icb_container->commandBuffer, tid);
        cmd.draw_indexed_primitives(
            (primitive_type)params.primitiveType,
            args[tid].indexCountPerInstance,
            indexBuffer + args[tid].startIndexLocation,
            args[tid].instanceCount,
            args[tid].baseVertexLocation,
            args[tid].startInstanceLocation
        );
    }
}
