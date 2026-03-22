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

GPU_TEST_CASE("cmd-draw-mesh-tasks", D3D12 | Metal)
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
    pass->drawMeshTasks(1, 1, 1);
    pass->end();

    queue->submit(encoder->finish());
    queue->waitOnHost();

    // Verify center pixel was drawn (non-zero alpha)
    ComPtr<ISlangBlob> resultBlob;
    SubresourceLayout layout;
    REQUIRE_CALL(device->readTexture(colorBuffer, 0, 0, resultBlob.writeRef(), &layout));
    auto result = (float*)resultBlob->getBufferPointer();
    // Center pixel (128, 128) should have been covered by the full-screen triangle
    auto pixel = result + 128 * 4 + 128 * layout.rowPitch / sizeof(float);
    CHECK(pixel[3] > 0.0f);
}

GPU_TEST_CASE("cmd-draw-mesh-tasks-with-object", D3D12 | Metal)
{
    if (!device->hasFeature(Feature::MeshShader))
        SKIP("mesh shaders not supported");

    ComPtr<IShaderProgram> program;
    REQUIRE_CALL(loadProgram(
        device,
        "test-cmd-draw-mesh-with-object",
        {"objectMain", "meshMain", "fragmentMain"},
        program.writeRef()
    ));

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
    pass->drawMeshTasks(1, 1, 1);
    pass->end();

    queue->submit(encoder->finish());
    queue->waitOnHost();

    // Verify center pixel is red (from the object shader's payload)
    ComPtr<ISlangBlob> resultBlob;
    SubresourceLayout layout;
    REQUIRE_CALL(device->readTexture(colorBuffer, 0, 0, resultBlob.writeRef(), &layout));
    auto result = (float*)resultBlob->getBufferPointer();
    auto pixel = result + 128 * 4 + 128 * layout.rowPitch / sizeof(float);
    // Should be red (1, 0, 0, 1) from the payload
    CHECK(pixel[0] > 0.9f);
    CHECK(pixel[1] < 0.1f);
    CHECK(pixel[2] < 0.1f);
    CHECK(pixel[3] > 0.9f);
}
