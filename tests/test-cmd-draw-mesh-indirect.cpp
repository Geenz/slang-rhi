#include "testing.h"

using namespace rhi;
using namespace rhi::testing;

const int kWidth = 256;
const int kHeight = 256;
const Format kFormat = Format::RGBA32Float;

static ComPtr<ITexture> createColorBuffer(IDevice* device)
{
    TextureDesc colorBufferDesc;
    colorBufferDesc.type = TextureType::Texture2D;
    colorBufferDesc.size.width = kWidth;
    colorBufferDesc.size.height = kHeight;
    colorBufferDesc.size.depth = 1;
    colorBufferDesc.mipCount = 1;
    colorBufferDesc.format = kFormat;
    colorBufferDesc.usage = TextureUsage::RenderTarget | TextureUsage::CopySource;
    colorBufferDesc.defaultState = ResourceState::RenderTarget;
    ComPtr<ITexture> colorBuffer = device->createTexture(colorBufferDesc, nullptr);
    REQUIRE(colorBuffer != nullptr);
    return colorBuffer;
}

GPU_TEST_CASE("cmd-draw-mesh-tasks-indirect", D3D12 | Metal)
{
    if (!device->hasFeature(Feature::MeshShader))
        SKIP("mesh shaders not supported");

    // Reuse the same mesh shader as the direct test
    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(device, "test-cmd-draw-mesh", {"meshMain", "fragmentMain"}, program.writeRef()));

    ColorTargetDesc colorTarget;
    colorTarget.format = kFormat;
    RenderPipelineDesc pipelineDesc = {};
    pipelineDesc.program = program.get();
    pipelineDesc.targets = &colorTarget;
    pipelineDesc.targetCount = 1;
    pipelineDesc.depthStencil.depthTestEnable = false;
    pipelineDesc.depthStencil.depthWriteEnable = false;
    ComPtr<IRenderPipeline> pipeline;
    REQUIRE_CALL(device->createRenderPipeline(pipelineDesc, pipeline.writeRef()));

    auto colorBuffer = createColorBuffer(device);
    ComPtr<ITextureView> colorBufferView;
    TextureViewDesc colorBufferViewDesc = {};
    colorBufferViewDesc.format = kFormat;
    REQUIRE_CALL(device->createTextureView(colorBuffer, colorBufferViewDesc, colorBufferView.writeRef()));

    // Create indirect argument buffer with dispatch args (1, 1, 1)
    uint32_t dispatchArgs[3] = {1, 1, 1};
    BufferDesc argBufferDesc = {};
    argBufferDesc.size = sizeof(dispatchArgs);
    argBufferDesc.usage = BufferUsage::IndirectArgument | BufferUsage::CopyDestination;
    argBufferDesc.defaultState = ResourceState::IndirectArgument;
    ComPtr<IBuffer> argBuffer = device->createBuffer(argBufferDesc, &dispatchArgs);
    REQUIRE(argBuffer != nullptr);

    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();

    RenderPassColorAttachment colorAttachment;
    colorAttachment.view = colorBufferView;
    colorAttachment.loadOp = LoadOp::Clear;
    colorAttachment.storeOp = StoreOp::Store;
    RenderPassDesc renderPass;
    renderPass.colorAttachments = &colorAttachment;
    renderPass.colorAttachmentCount = 1;
    auto pass = encoder->beginRenderPass(renderPass);

    pass->bindPipeline(pipeline);

    RenderState state;
    state.viewports[0] = Viewport::fromSize(kWidth, kHeight);
    state.viewportCount = 1;
    state.scissorRects[0] = ScissorRect::fromSize(kWidth, kHeight);
    state.scissorRectCount = 1;
    pass->setRenderState(state);
    pass->drawMeshTasksIndirect({argBuffer.get(), 0});
    pass->end();

    queue->submit(encoder->finish());
    queue->waitOnHost();

    // Verify center pixel was drawn (same result as direct drawMeshTasks)
    ComPtr<ISlangBlob> resultBlob;
    SubresourceLayout layout;
    REQUIRE_CALL(device->readTexture(colorBuffer, 0, 0, resultBlob.writeRef(), &layout));
    auto result = (float*)resultBlob->getBufferPointer();
    auto pixel = result + 128 * 4 + 128 * layout.rowPitch / sizeof(float);
    CHECK(pixel[3] > 0.0f);
}

GPU_TEST_CASE("cmd-draw-mesh-tasks-indirect-offset", D3D12 | Metal)
{
    if (!device->hasFeature(Feature::MeshShader))
        SKIP("mesh shaders not supported");

    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(device, "test-cmd-draw-mesh", {"meshMain", "fragmentMain"}, program.writeRef()));

    ColorTargetDesc colorTarget;
    colorTarget.format = kFormat;
    RenderPipelineDesc pipelineDesc = {};
    pipelineDesc.program = program.get();
    pipelineDesc.targets = &colorTarget;
    pipelineDesc.targetCount = 1;
    pipelineDesc.depthStencil.depthTestEnable = false;
    pipelineDesc.depthStencil.depthWriteEnable = false;
    ComPtr<IRenderPipeline> pipeline;
    REQUIRE_CALL(device->createRenderPipeline(pipelineDesc, pipeline.writeRef()));

    auto colorBuffer = createColorBuffer(device);
    ComPtr<ITextureView> colorBufferView;
    TextureViewDesc colorBufferViewDesc = {};
    colorBufferViewDesc.format = kFormat;
    REQUIRE_CALL(device->createTextureView(colorBuffer, colorBufferViewDesc, colorBufferView.writeRef()));

    // Create buffer with padding before the actual dispatch args
    // This tests the offset parameter (like reading from clusterCount[1..3])
    uint32_t bufferData[4] = {42, 1, 1, 1};  // [0]=padding, [1..3]=dispatch args
    BufferDesc argBufferDesc = {};
    argBufferDesc.size = sizeof(bufferData);
    argBufferDesc.usage = BufferUsage::IndirectArgument | BufferUsage::CopyDestination;
    argBufferDesc.defaultState = ResourceState::IndirectArgument;
    ComPtr<IBuffer> argBuffer = device->createBuffer(argBufferDesc, &bufferData);
    REQUIRE(argBuffer != nullptr);

    auto queue = device->getQueue(QueueType::Graphics);
    auto encoder = queue->createCommandEncoder();

    RenderPassColorAttachment colorAttachment;
    colorAttachment.view = colorBufferView;
    colorAttachment.loadOp = LoadOp::Clear;
    colorAttachment.storeOp = StoreOp::Store;
    RenderPassDesc renderPass;
    renderPass.colorAttachments = &colorAttachment;
    renderPass.colorAttachmentCount = 1;
    auto pass = encoder->beginRenderPass(renderPass);

    pass->bindPipeline(pipeline);

    RenderState state;
    state.viewports[0] = Viewport::fromSize(kWidth, kHeight);
    state.viewportCount = 1;
    state.scissorRects[0] = ScissorRect::fromSize(kWidth, kHeight);
    state.scissorRectCount = 1;
    // Offset by 4 bytes to skip the padding value
    pass->drawMeshTasksIndirect({argBuffer.get(), 4});
    pass->end();

    queue->submit(encoder->finish());
    queue->waitOnHost();

    // Verify center pixel was drawn
    ComPtr<ISlangBlob> resultBlob;
    SubresourceLayout layout;
    REQUIRE_CALL(device->readTexture(colorBuffer, 0, 0, resultBlob.writeRef(), &layout));
    auto result = (float*)resultBlob->getBufferPointer();
    auto pixel = result + 128 * 4 + 128 * layout.rowPitch / sizeof(float);
    CHECK(pixel[3] > 0.0f);
}
