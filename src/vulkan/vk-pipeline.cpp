#include "vk-pipeline.h"
#include "vk-device.h"
#include "vk-shader-object-layout.h"
#include "vk-shader-program.h"
#include "vk-input-layout.h"
#include "vk-utils.h"

#include "core/static_vector.h"
#include "core/sha1.h"

#include <map>
#include <string>
#include <vector>

namespace rhi::vk {

// For pipeline caching, we use the VK_KHR_pipeline_binary extension.
// We serialize the pipeline binaries into a custom format that stores a number of pipeline binaries,
// each with a key and data size, along with the binary data itself.
// The format is laid out as follows:
// Header [PipelineCacheHeader] (12 bytes):
// - Magic number (4 bytes)
// - Version (4 bytes)
// - Number of binaries (4 bytes)
// Binary headers [PipelineCacheBinaryHeader] (44 bytes each):
// - Key size (4 bytes)
// - Key (32 bytes (VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR))
// - Data size (4 bytes)
// - Data offset (4 bytes, relative to the start of the blob)
// Binary data (variable size)

struct PipelineCacheHeader
{
    static constexpr uint32_t kMagic = 0x12345678;
    static constexpr uint32_t kVersion = 1;

    uint32_t magic;
    uint32_t version;
    uint32_t binaryCount;
};

struct PipelineCacheBinaryHeader
{
    uint32_t keySize;
    uint8_t key[VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR];
    uint32_t dataSize;
    uint32_t dataOffset;
};

// Both structs are written to and read from disk directly, so the layout documented above is part of
// the cache format that kVersion identifies.
static_assert(sizeof(PipelineCacheHeader) == 12);
static_assert(sizeof(PipelineCacheBinaryHeader) == 44);

// Hash the identity of a shader program (SPIR-V modules, entry point names, stages).
static void hashProgramIdentity(SHA1& sha1, ShaderProgramImpl* program)
{
    for (const auto& module : program->m_modules)
    {
        sha1.update(module.code->getBufferPointer(), module.code->getBufferSize());
        sha1.update(module.entryPointName.data(), module.entryPointName.size());
    }
    for (const auto& stage : program->m_stageCreateInfos)
    {
        sha1.update(&stage.stage, sizeof(stage.stage));
    }
}

// Keys are derived app-side from device + pipeline identity: NVIDIA drivers (observed on 610.47)
// corrupt the caller's stack in vkGetPipelineKeyKHR, so that entry point must not be used.
Result getPipelineCacheKey(DeviceImpl* device, const SHA1& pipelineIdentity, ISlangBlob** outBlob)
{
    SHA1 sha1(pipelineIdentity);
    {
        const AdapterLUID& luid = device->getInfo().adapterLUID;
        sha1.update(luid.luid, sizeof(luid.luid));
    }
    {
        const VkPhysicalDeviceProperties& props = device->m_api.m_deviceProperties;
        sha1.update(&props.vendorID, sizeof(props.vendorID));
        sha1.update(&props.deviceID, sizeof(props.deviceID));
        sha1.update(&props.driverVersion, sizeof(props.driverVersion));
        sha1.update(props.pipelineCacheUUID, sizeof(props.pipelineCacheUUID));
    }
    SHA1::Digest digest = sha1.getDigest();
    ComPtr<ISlangBlob> blob = OwnedBlob::create(digest.data(), digest.size());
    returnComPtr(outBlob, blob);
    return SLANG_OK;
}

// Serialize a vulkan pipeline into a blob containing the pipeline binaries.
Result serializePipelineBinaries(DeviceImpl* device, VkPipeline pipeline, ISlangBlob** outBlob)
{
    auto& api = device->m_api;

    VkPipelineBinaryCreateInfoKHR binaryCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR};
    binaryCreateInfo.pipeline = pipeline;

    VkPipelineBinaryHandlesInfoKHR binaryHandlesInfo = {VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};

    SLANG_VK_RETURN_ON_FAIL_REPORT(
        api.vkCreatePipelineBinariesKHR(device->m_device, &binaryCreateInfo, nullptr, &binaryHandlesInfo),
        device
    );

    short_vector<VkPipelineBinaryKHR> pipelineBinaries(binaryHandlesInfo.pipelineBinaryCount, VK_NULL_HANDLE);
    binaryHandlesInfo.pPipelineBinaries = pipelineBinaries.data();
    SLANG_VK_RETURN_ON_FAIL_REPORT(
        api.vkCreatePipelineBinariesKHR(device->m_device, &binaryCreateInfo, nullptr, &binaryHandlesInfo),
        device
    );

    // Compute total size of the cache data blob.
    size_t dataSize = sizeof(PipelineCacheHeader);
    dataSize += binaryHandlesInfo.pipelineBinaryCount * sizeof(PipelineCacheBinaryHeader);
    for (uint32_t i = 0; i < binaryHandlesInfo.pipelineBinaryCount; ++i)
    {
        VkPipelineBinaryDataInfoKHR binaryInfo = {VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR};
        binaryInfo.pipelineBinary = pipelineBinaries[i];
        VkPipelineBinaryKeyKHR binaryKey = {VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
        size_t binaryDataSize = 0;
        SLANG_VK_RETURN_ON_FAIL_REPORT(
            api.vkGetPipelineBinaryDataKHR(device->m_device, &binaryInfo, &binaryKey, &binaryDataSize, nullptr),
            device
        );
        dataSize += binaryDataSize;
    }

    ComPtr<ISlangBlob> blob = OwnedBlob::create(dataSize);
    uint8_t* data = (uint8_t*)blob->getBufferPointer();
    uint8_t* dataPtr = data;

    // Write cache data header.
    PipelineCacheHeader* header = (PipelineCacheHeader*)dataPtr;
    header->magic = PipelineCacheHeader::kMagic;
    header->version = PipelineCacheHeader::kVersion;
    header->binaryCount = binaryHandlesInfo.pipelineBinaryCount;
    dataPtr += sizeof(PipelineCacheHeader);

    // Write binary data.
    uint32_t binaryDataOffset =
        sizeof(PipelineCacheHeader) + binaryHandlesInfo.pipelineBinaryCount * sizeof(PipelineCacheBinaryHeader);
    for (uint32_t i = 0; i < binaryHandlesInfo.pipelineBinaryCount; ++i)
    {
        VkPipelineBinaryDataInfoKHR binaryInfo = {VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR};
        binaryInfo.pipelineBinary = pipelineBinaries[i];

        VkPipelineBinaryKeyKHR binaryKey = {VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
        size_t binaryDataSize = 0;
        SLANG_VK_RETURN_ON_FAIL_REPORT(
            api.vkGetPipelineBinaryDataKHR(device->m_device, &binaryInfo, &binaryKey, &binaryDataSize, nullptr),
            device
        );

        SLANG_VK_RETURN_ON_FAIL_REPORT(
            api.vkGetPipelineBinaryDataKHR(
                device->m_device,
                &binaryInfo,
                &binaryKey,
                &binaryDataSize,
                data + binaryDataOffset
            ),
            device
        );

        if (binaryKey.keySize > sizeof(PipelineCacheBinaryHeader::key))
            return SLANG_FAIL;

        PipelineCacheBinaryHeader* binaryHeader = (PipelineCacheBinaryHeader*)dataPtr;
        std::memset(binaryHeader->key, 0, sizeof(PipelineCacheBinaryHeader::key));
        std::memcpy(binaryHeader->key, binaryKey.key, binaryKey.keySize);
        binaryHeader->keySize = binaryKey.keySize;
        binaryHeader->dataSize = (uint32_t)binaryDataSize;
        binaryHeader->dataOffset = binaryDataOffset;
        dataPtr += sizeof(PipelineCacheBinaryHeader);

        binaryDataOffset += binaryDataSize;

        api.vkDestroyPipelineBinaryKHR(device->m_device, pipelineBinaries[i], nullptr);
    }

    returnComPtr(outBlob, blob);
    return SLANG_OK;
}

// Parse the pipeline binary keys and data ranges out of a serialized cache blob.
//
// The blob comes from an IPersistentCache implementation, so a truncated, stale or tampered entry
// must be rejected rather than trusted: every length and offset is validated before it is used to
// index, to size a copy, or to form a pointer. Fields are copied out of the blob rather than read
// through a struct pointer, as getBufferPointer() carries no alignment guarantee.
//
// Returns SLANG_FAIL for a malformed blob, which the caller handles by creating the pipeline without
// the cache. Kept independent of the device so it can be validated without one, since reaching it
// through the cache requires VK_KHR_pipeline_binary.
Result parsePipelineCacheBlob(
    const void* blobData,
    size_t blobSize,
    short_vector<VkPipelineBinaryKeyKHR>& outKeys,
    short_vector<VkPipelineBinaryDataKHR>& outData
)
{
    const uint8_t* data = (const uint8_t*)blobData;
    const uint8_t* dataPtr = data;
    if (blobSize < sizeof(PipelineCacheHeader))
    {
        return SLANG_FAIL;
    }

    PipelineCacheHeader header;
    std::memcpy(&header, dataPtr, sizeof(header));
    if (header.magic != PipelineCacheHeader::kMagic || header.version != PipelineCacheHeader::kVersion ||
        header.binaryCount == 0)
    {
        return SLANG_FAIL;
    }
    dataPtr += sizeof(PipelineCacheHeader);

    // Expressed as a division so the bound itself cannot overflow.
    if (header.binaryCount > (blobSize - sizeof(PipelineCacheHeader)) / sizeof(PipelineCacheBinaryHeader))
    {
        return SLANG_FAIL;
    }

    // Binary data must start past the record table, or a record could point back into the header or
    // table and that metadata would be handed to the driver as pipeline binary data.
    const size_t tableEnd = sizeof(PipelineCacheHeader) + header.binaryCount * sizeof(PipelineCacheBinaryHeader);

    short_vector<VkPipelineBinaryKeyKHR> binaryKeys(header.binaryCount, {VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR});
    short_vector<VkPipelineBinaryDataKHR> pipelineData(header.binaryCount, {});

    for (uint32_t i = 0; i < header.binaryCount; ++i)
    {
        PipelineCacheBinaryHeader binaryHeader;
        std::memcpy(&binaryHeader, dataPtr, sizeof(binaryHeader));
        dataPtr += sizeof(PipelineCacheBinaryHeader);

        if (binaryHeader.keySize > VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR)
        {
            return SLANG_FAIL;
        }
        if (binaryHeader.dataOffset < tableEnd || binaryHeader.dataOffset > blobSize ||
            binaryHeader.dataSize > blobSize - binaryHeader.dataOffset)
        {
            return SLANG_FAIL;
        }

        binaryKeys[i].keySize = binaryHeader.keySize;
        std::memcpy(binaryKeys[i].key, binaryHeader.key, binaryHeader.keySize);

        pipelineData[i].dataSize = binaryHeader.dataSize;
        pipelineData[i].pData = (void*)(data + binaryHeader.dataOffset);
    }

    outKeys = binaryKeys;
    outData = pipelineData;
    return SLANG_OK;
}

// Deserialize a blob containing pipeline binaries into a vector of VkPipelineBinaryKHR handles.
// The caller is responsible for destroying the VkPipelineBinaryKHR handles after use.
Result deserializePipelineBinaries(DeviceImpl* device, ISlangBlob* blob, short_vector<VkPipelineBinaryKHR>& outBinaries)
{
    auto& api = device->m_api;

    short_vector<VkPipelineBinaryKeyKHR> binaryKeys;
    short_vector<VkPipelineBinaryDataKHR> pipelineData;
    SLANG_RETURN_ON_FAIL(
        parsePipelineCacheBlob(blob->getBufferPointer(), blob->getBufferSize(), binaryKeys, pipelineData)
    );

    VkPipelineBinaryKeysAndDataKHR binaryKeysAndData;
    binaryKeysAndData.binaryCount = (uint32_t)binaryKeys.size();
    binaryKeysAndData.pPipelineBinaryKeys = binaryKeys.data();
    binaryKeysAndData.pPipelineBinaryData = pipelineData.data();

    VkPipelineBinaryCreateInfoKHR createInfo = {VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR};
    createInfo.pKeysAndDataInfo = &binaryKeysAndData;

    short_vector<VkPipelineBinaryKHR> binaries(binaryKeys.size(), VK_NULL_HANDLE);

    VkPipelineBinaryHandlesInfoKHR handlesInfo = {VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
    handlesInfo.pipelineBinaryCount = binaries.size();
    handlesInfo.pPipelineBinaries = binaries.data();

    SLANG_VK_RETURN_ON_FAIL_REPORT(
        api.vkCreatePipelineBinariesKHR(device->m_device, &createInfo, nullptr, &handlesInfo),
        device
    );

    outBinaries = binaries;
    return SLANG_OK;
}

template<typename VkPipelineCreateInfo>
Result createPipelineWithCache(
    DeviceImpl* device,
    VkPipelineCreateInfo* createInfo,
    const SHA1& pipelineIdentity,
    VkResult (*createPipelineFunc)(DeviceImpl* device, VkPipelineCreateInfo* createInfo, VkPipeline* outPipeline),
    VkPipeline* outPipeline,
    bool& outCached,
    size_t& outCacheSize,
    ComPtr<ISlangBlob>& outCacheKey
)
{
    auto& api = device->m_api;

    outCached = false;
    outCacheSize = 0;
    outCacheKey = nullptr;

    // Early out if cache is not enabled or the feature is not supported.
    if (!device->m_persistentPipelineCache || !api.m_extendedFeatures.pipelineBinaryFeatures.pipelineBinaries)
    {
        return createPipelineFunc(device, createInfo, outPipeline);
    }

    bool writeCache = true;
    ComPtr<ISlangBlob> pipelineCacheKey;
    ComPtr<ISlangBlob> pipelineCacheData;
    VkPipeline pipeline = VK_NULL_HANDLE;

    // Create pipeline cache key.
    if (SLANG_FAILED(getPipelineCacheKey(device, pipelineIdentity, pipelineCacheKey.writeRef())))
    {
        device->printWarning("Failed to get pipeline cache key, disabling pipeline cache.");
        return createPipelineFunc(device, createInfo, outPipeline);
    }

    // Query pipeline cache.
    if (SLANG_FAILED(device->m_persistentPipelineCache->queryCache(pipelineCacheKey, pipelineCacheData.writeRef())))
    {
        pipelineCacheData = nullptr;
    }

    // Try create pipeline from cache.
    if (pipelineCacheData)
    {
        short_vector<VkPipelineBinaryKHR> pipelineBinaries;
        if (SLANG_SUCCEEDED(deserializePipelineBinaries(device, pipelineCacheData, pipelineBinaries)))
        {
            VkPipelineBinaryInfoKHR binaryInfo = {VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR};
            binaryInfo.binaryCount = (uint32_t)pipelineBinaries.size();
            binaryInfo.pPipelineBinaries = pipelineBinaries.data();
            binaryInfo.pNext = createInfo->pNext;
            createInfo->pNext = &binaryInfo;
            if (createPipelineFunc(device, createInfo, &pipeline) == VK_SUCCESS)
            {
                writeCache = false;
                outCached = true;
                outCacheSize = pipelineCacheData->getBufferSize();
                outCacheKey = pipelineCacheKey;
            }
            else
            {
                createInfo->pNext = binaryInfo.pNext;
                pipeline = VK_NULL_HANDLE;
                device->printWarning("Failed to create pipeline from cache, creating new pipeline.");
            }
            for (auto& binary : pipelineBinaries)
            {
                api.vkDestroyPipelineBinaryKHR(device->m_device, binary, nullptr);
            }
        }
        else
        {
            device->printWarning("Failed to deserialize pipeline binaries from cache, creating new pipeline.");
        }
    }

    // Create pipeline if not found in cache.
    if (!pipeline)
    {
        // To capture the pipeline data, we need to set the VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR flag
        // in VkPipelineCreateFlags2CreateInfoKHR. In some cases, the passed in createInfo already has a
        // VkPipelineCreateFlags2CreateInfoKHR in the chain, so we use that, otherwise create a new one on the stack.
        VkPipelineCreateFlags2CreateInfoKHR createFlags = {VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR};
        if (writeCache)
        {
            // Check createInfo chain for existing VkPipelineCreateFlags2CreateInfoKHR
            bool foundExistingCreateFlags = false;
            VkBaseInStructure* inStruct = (VkBaseInStructure*)createInfo->pNext;
            while (inStruct)
            {
                if (inStruct->sType == VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR)
                {
                    ((VkPipelineCreateFlags2CreateInfoKHR*)inStruct)->flags |=
                        VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR;
                    foundExistingCreateFlags = true;
                    break;
                }
                inStruct = (VkBaseInStructure*)inStruct->pNext;
            }
            // If not found, append VkPipelineCreateFlags2CreateInfoKHR on stack
            if (!foundExistingCreateFlags)
            {
                createFlags.flags = VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR;
                createFlags.pNext = createInfo->pNext;
                createInfo->pNext = &createFlags;
            }
        }
        SLANG_VK_RETURN_ON_FAIL_REPORT(createPipelineFunc(device, createInfo, &pipeline), device);
    }

    // Write to the cache.
    if (writeCache)
    {
        if (SLANG_SUCCEEDED(serializePipelineBinaries(device, pipeline, pipelineCacheData.writeRef())))
        {
            device->m_persistentPipelineCache->writeCache(pipelineCacheKey, pipelineCacheData);
            outCacheSize = pipelineCacheData->getBufferSize();
            outCacheKey = pipelineCacheKey;
        }
        else
        {
            device->printWarning("Failed to serialize pipeline binaries, cache write skipped.");
        }
    }

    // Release captured pipeline data.
    if (writeCache)
    {
        VkReleaseCapturedPipelineDataInfoKHR releaseInfo = {VK_STRUCTURE_TYPE_RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR};
        releaseInfo.pipeline = pipeline;
        SLANG_VK_RETURN_ON_FAIL_REPORT(
            api.vkReleaseCapturedPipelineDataKHR(device->m_device, &releaseInfo, nullptr),
            device
        );
    }

    *outPipeline = pipeline;
    return SLANG_OK;
}

RenderPipelineImpl::RenderPipelineImpl(Device* device, const RenderPipelineDesc& desc)
    : RenderPipeline(device, desc)
{
}

RenderPipelineImpl::~RenderPipelineImpl()
{
    DeviceImpl* device = getDevice<DeviceImpl>();

    if (m_pipeline != VK_NULL_HANDLE)
    {
        device->m_api.vkDestroyPipeline(device->m_api.m_device, m_pipeline, nullptr);
    }
}

Result RenderPipelineImpl::getNativeHandle(NativeHandle* outHandle)
{
    outHandle->type = NativeHandleType::VkPipeline;
    outHandle->value = (uint64_t)m_pipeline;
    return SLANG_OK;
}

Result DeviceImpl::createRenderPipeline2(const RenderPipelineDesc& desc, IRenderPipeline** outPipeline)
{
    TimePoint startTime = Timer::now();

    ShaderProgramImpl* program = checked_cast<ShaderProgramImpl*>(desc.program);
    SLANG_RHI_ASSERT(!program->m_modules.empty());
    InputLayoutImpl* inputLayout = checked_cast<InputLayoutImpl*>(desc.inputLayout);

    // VertexBuffer/s
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;

    if (inputLayout)
    {
        const auto& srcAttributeDescs = inputLayout->m_attributeDescs;
        const auto& srcStreamDescs = inputLayout->m_streamDescs;

        vertexInputInfo.vertexBindingDescriptionCount = (uint32_t)srcStreamDescs.size();
        vertexInputInfo.pVertexBindingDescriptions = srcStreamDescs.data();

        vertexInputInfo.vertexAttributeDescriptionCount = (uint32_t)srcAttributeDescs.size();
        vertexInputInfo.pVertexAttributeDescriptions = srcAttributeDescs.data();
    }

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    // All other forms of primitive toplogies are specified via dynamic state.
    inputAssembly.topology = translatePrimitiveListTopology(desc.primitiveTopology);
    inputAssembly.primitiveRestartEnable = VK_FALSE; // TODO: Currently unsupported

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    // We are using dynamic viewport and scissor state.
    // Here we specify an arbitrary size, actual viewport will be set at `beginRenderPass`
    // time.
    viewport.width = 16.0f;
    viewport.height = 16.0f;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {uint32_t(16), uint32_t(16)};

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    auto rasterizerDesc = desc.rasterizer;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_TRUE; // TODO: Depth clipping and clamping are different between Vk and D3D12
    rasterizer.rasterizerDiscardEnable = VK_FALSE; // TODO: Currently unsupported
    rasterizer.polygonMode = translateFillMode(rasterizerDesc.fillMode);
    rasterizer.cullMode = translateCullMode(rasterizerDesc.cullMode);
    rasterizer.frontFace = translateFrontFaceMode(rasterizerDesc.frontFace);
    rasterizer.depthBiasEnable = (rasterizerDesc.depthBias == 0) ? VK_FALSE : VK_TRUE;
    rasterizer.depthBiasConstantFactor = (float)rasterizerDesc.depthBias;
    rasterizer.depthBiasClamp = rasterizerDesc.depthBiasClamp;
    rasterizer.depthBiasSlopeFactor = rasterizerDesc.slopeScaledDepthBias;
    rasterizer.lineWidth = 1.0f; // TODO: Currently unsupported

    VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeRasterInfo = {};
    conservativeRasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
    conservativeRasterInfo.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
    if (desc.rasterizer.enableConservativeRasterization)
    {
        rasterizer.pNext = &conservativeRasterInfo;
    }

    auto forcedSampleCount = rasterizerDesc.forcedSampleCount;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = (forcedSampleCount == 0) ? VkSampleCountFlagBits(desc.multisample.sampleCount)
                                                                  : translateSampleCount(forcedSampleCount);
    multisampling.sampleShadingEnable = VK_FALSE; // TODO: Should check if fragment shader needs this
    // TODO: Sample mask is dynamic in D3D12 but PSO state in Vulkan
    multisampling.alphaToCoverageEnable = desc.multisample.alphaToCoverageEnable;
    multisampling.alphaToOneEnable = desc.multisample.alphaToOneEnable;

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendTargets;

    // Regardless of whether blending is enabled, Vulkan always applies the color write mask
    // operation, so if there is no blending then we need to add an attachment that defines
    // the color write mask to ensure colors are actually written.
    if (desc.targetCount == 0)
    {
        colorBlendTargets.resize(1);
        auto& vkBlendDesc = colorBlendTargets[0];
        memset(&vkBlendDesc, 0, sizeof(vkBlendDesc));
        vkBlendDesc.blendEnable = VK_FALSE;
        vkBlendDesc.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        vkBlendDesc.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        vkBlendDesc.colorBlendOp = VK_BLEND_OP_ADD;
        vkBlendDesc.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        vkBlendDesc.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        vkBlendDesc.alphaBlendOp = VK_BLEND_OP_ADD;
        vkBlendDesc.colorWriteMask = (VkColorComponentFlags)RenderTargetWriteMask::All;
    }
    else
    {
        colorBlendTargets.resize(desc.targetCount);
        for (uint32_t i = 0; i < desc.targetCount; ++i)
        {
            auto& target = desc.targets[i];
            auto& vkBlendDesc = colorBlendTargets[i];

            vkBlendDesc.blendEnable = target.enableBlend;
            vkBlendDesc.srcColorBlendFactor = translateBlendFactor(target.color.srcFactor);
            vkBlendDesc.dstColorBlendFactor = translateBlendFactor(target.color.dstFactor);
            vkBlendDesc.colorBlendOp = translateBlendOp(target.color.op);
            vkBlendDesc.srcAlphaBlendFactor = translateBlendFactor(target.alpha.srcFactor);
            vkBlendDesc.dstAlphaBlendFactor = translateBlendFactor(target.alpha.dstFactor);
            vkBlendDesc.alphaBlendOp = translateBlendOp(target.alpha.op);
            vkBlendDesc.colorWriteMask = (VkColorComponentFlags)target.writeMask;
        }
    }

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE; // TODO: D3D12 has per attachment logic op (and
                                            // both have way more than one op)
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = (uint32_t)colorBlendTargets.size();
    colorBlending.pAttachments = colorBlendTargets.data();
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    static_vector<VkDynamicState, 8> dynamicStates;
    dynamicStates.push_back(VK_DYNAMIC_STATE_VIEWPORT);
    dynamicStates.push_back(VK_DYNAMIC_STATE_SCISSOR);
    dynamicStates.push_back(VK_DYNAMIC_STATE_STENCIL_REFERENCE);
    dynamicStates.push_back(VK_DYNAMIC_STATE_BLEND_CONSTANTS);
    VkPipelineDynamicStateCreateInfo dynamicStateInfo = {};
    dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.dynamicStateCount = (uint32_t)dynamicStates.size();
    dynamicStateInfo.pDynamicStates = dynamicStates.data();

    VkPipelineDepthStencilStateCreateInfo depthStencilStateInfo = {};
    depthStencilStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilStateInfo.depthTestEnable = desc.depthStencil.depthTestEnable ? 1 : 0;
    depthStencilStateInfo.back = translateStencilState(desc.depthStencil.backFace);
    depthStencilStateInfo.front = translateStencilState(desc.depthStencil.frontFace);
    depthStencilStateInfo.back.compareMask = desc.depthStencil.stencilReadMask;
    depthStencilStateInfo.back.writeMask = desc.depthStencil.stencilWriteMask;
    depthStencilStateInfo.front.compareMask = desc.depthStencil.stencilReadMask;
    depthStencilStateInfo.front.writeMask = desc.depthStencil.stencilWriteMask;
    depthStencilStateInfo.depthBoundsTestEnable = 0; // TODO: Currently unsupported
    depthStencilStateInfo.depthCompareOp = translateComparisonFunc(desc.depthStencil.depthFunc);
    depthStencilStateInfo.depthWriteEnable = desc.depthStencil.depthWriteEnable ? 1 : 0;
    depthStencilStateInfo.stencilTestEnable = desc.depthStencil.stencilEnable ? 1 : 0;

    VkPipelineRenderingCreateInfoKHR renderingInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    short_vector<VkFormat> colorAttachmentFormats;
    for (uint32_t i = 0; i < desc.targetCount; ++i)
    {
        colorAttachmentFormats.push_back(getVkFormat(desc.targets[i].format));
    }
    renderingInfo.colorAttachmentCount = colorAttachmentFormats.size();
    renderingInfo.pColorAttachmentFormats = colorAttachmentFormats.data();
    renderingInfo.depthAttachmentFormat = getVkFormat(desc.depthStencil.format);
    if (isStencilFormat(renderingInfo.depthAttachmentFormat))
    {
        renderingInfo.stencilAttachmentFormat = renderingInfo.depthAttachmentFormat;
    }

    VkGraphicsPipelineCreateInfo createInfo = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    createInfo.pNext = &renderingInfo;
    createInfo.stageCount = (uint32_t)program->m_stageCreateInfos.size();
    createInfo.pStages = program->m_stageCreateInfos.data();
    createInfo.pVertexInputState = &vertexInputInfo;
    createInfo.pInputAssemblyState = &inputAssembly;
    createInfo.pViewportState = &viewportState;
    createInfo.pRasterizationState = &rasterizer;
    createInfo.pMultisampleState = &multisampling;
    createInfo.pColorBlendState = &colorBlending;
    createInfo.pDepthStencilState = &depthStencilStateInfo;
    createInfo.layout = program->m_rootShaderObjectLayout->m_pipelineLayout;
    createInfo.subpass = 0;
    createInfo.basePipelineHandle = VK_NULL_HANDLE;
    createInfo.pDynamicState = &dynamicStateInfo;

    SHA1 pipelineIdentity;
    hashProgramIdentity(pipelineIdentity, program);
    // Field-wise: struct padding bytes are indeterminate and must not reach the hash.
    auto hashValue = [&](const auto& value) { pipelineIdentity.update(&value, sizeof(value)); };
    hashValue(desc.primitiveTopology);
    for (uint32_t i = 0; i < desc.targetCount; i++)
    {
        const ColorTargetDesc& target = desc.targets[i];
        hashValue(target.format);
        hashValue(target.color);
        hashValue(target.alpha);
        hashValue(target.enableBlend);
        hashValue(target.logicOp);
        hashValue(target.writeMask);
    }
    const DepthStencilDesc& ds = desc.depthStencil;
    hashValue(ds.format);
    hashValue(ds.depthTestEnable);
    hashValue(ds.depthWriteEnable);
    hashValue(ds.depthFunc);
    hashValue(ds.stencilEnable);
    hashValue(ds.stencilReadMask);
    hashValue(ds.stencilWriteMask);
    hashValue(ds.frontFace);
    hashValue(ds.backFace);
    hashValue(ds.stencilRef);
    const RasterizerDesc& rs = desc.rasterizer;
    hashValue(rs.fillMode);
    hashValue(rs.cullMode);
    hashValue(rs.frontFace);
    hashValue(rs.depthBias);
    hashValue(rs.depthBiasClamp);
    hashValue(rs.slopeScaledDepthBias);
    hashValue(rs.depthClipEnable);
    hashValue(rs.scissorEnable);
    hashValue(rs.multisampleEnable);
    hashValue(rs.antialiasedLineEnable);
    hashValue(rs.enableConservativeRasterization);
    hashValue(rs.forcedSampleCount);
    const MultisampleDesc& ms = desc.multisample;
    hashValue(ms.sampleCount);
    hashValue(ms.sampleMask);
    hashValue(ms.alphaToCoverageEnable);
    hashValue(ms.alphaToOneEnable);
    if (inputLayout)
    {
        const auto& attributeDescs = inputLayout->m_attributeDescs;
        const auto& streamDescs = inputLayout->m_streamDescs;
        pipelineIdentity.update(attributeDescs.data(), attributeDescs.size() * sizeof(attributeDescs[0]));
        pipelineIdentity.update(streamDescs.data(), streamDescs.size() * sizeof(streamDescs[0]));
    }

    VkPipeline vkPipeline = VK_NULL_HANDLE;
    ComPtr<ISlangBlob> cacheKey;
    bool cached = false;
    size_t cacheSize = 0;
    SLANG_RETURN_ON_FAIL(
        createPipelineWithCache<VkGraphicsPipelineCreateInfo>(
            this,
            &createInfo,
            pipelineIdentity,
            [](DeviceImpl* device, VkGraphicsPipelineCreateInfo* createInfo2, VkPipeline* pipeline) -> VkResult
            {
                return device->m_api
                    .vkCreateGraphicsPipelines(device->m_device, VK_NULL_HANDLE, 1, createInfo2, nullptr, pipeline);
            },
            &vkPipeline,
            cached,
            cacheSize,
            cacheKey
        )
    );

    _labelObject((uint64_t)vkPipeline, VK_OBJECT_TYPE_PIPELINE, desc.label);

    // Report the pipeline creation time.
    if (m_shaderCompilationReporter)
    {
        m_shaderCompilationReporter->reportCreatePipeline(
            program,
            ShaderCompilationReporter::PipelineType::Render,
            startTime,
            Timer::now(),
            cached,
            cacheSize,
            cacheKey
        );
    }

    RefPtr<RenderPipelineImpl> pipeline = new RenderPipelineImpl(this, desc);
    pipeline->m_program = program;
    pipeline->m_rootObjectLayout = program->m_rootShaderObjectLayout;
    pipeline->m_pipeline = vkPipeline;
    returnComPtr(outPipeline, pipeline);
    return SLANG_OK;
}

ComputePipelineImpl::ComputePipelineImpl(Device* device, const ComputePipelineDesc& desc)
    : ComputePipeline(device, desc)
{
}

ComputePipelineImpl::~ComputePipelineImpl()
{
    DeviceImpl* device = getDevice<DeviceImpl>();

    if (m_pipeline != VK_NULL_HANDLE)
    {
        device->m_api.vkDestroyPipeline(device->m_api.m_device, m_pipeline, nullptr);
    }
}

Result ComputePipelineImpl::getNativeHandle(NativeHandle* outHandle)
{
    outHandle->type = NativeHandleType::VkPipeline;
    outHandle->value = (uint64_t)m_pipeline;
    return SLANG_OK;
}

Result DeviceImpl::createComputePipeline2(const ComputePipelineDesc& desc, IComputePipeline** outPipeline)
{
    TimePoint startTime = Timer::now();

    ShaderProgramImpl* program = checked_cast<ShaderProgramImpl*>(desc.program);
    SLANG_RHI_ASSERT(!program->m_modules.empty());

    VkComputePipelineCreateInfo createInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    createInfo.stage = program->m_stageCreateInfos[0];
    createInfo.layout = program->m_rootShaderObjectLayout->m_pipelineLayout;

    SHA1 pipelineIdentity;
    hashProgramIdentity(pipelineIdentity, program);

    VkPipeline vkPipeline = VK_NULL_HANDLE;
    ComPtr<ISlangBlob> cacheKey;
    bool cached = false;
    size_t cacheSize = 0;
    SLANG_RETURN_ON_FAIL(
        createPipelineWithCache<VkComputePipelineCreateInfo>(
            this,
            &createInfo,
            pipelineIdentity,
            [](DeviceImpl* device, VkComputePipelineCreateInfo* createInfo2, VkPipeline* pipeline) -> VkResult
            {
                return device->m_api
                    .vkCreateComputePipelines(device->m_device, VK_NULL_HANDLE, 1, createInfo2, nullptr, pipeline);
            },
            &vkPipeline,
            cached,
            cacheSize,
            cacheKey
        )
    );

    _labelObject((uint64_t)vkPipeline, VK_OBJECT_TYPE_PIPELINE, desc.label);

    // Report the pipeline creation time.
    if (m_shaderCompilationReporter)
    {
        m_shaderCompilationReporter->reportCreatePipeline(
            program,
            ShaderCompilationReporter::PipelineType::Compute,
            startTime,
            Timer::now(),
            cached,
            cacheSize,
            cacheKey
        );
    }

    RefPtr<ComputePipelineImpl> pipeline = new ComputePipelineImpl(this, desc);
    pipeline->m_program = program;
    pipeline->m_rootObjectLayout = program->m_rootShaderObjectLayout;
    pipeline->m_pipeline = vkPipeline;
    returnComPtr(outPipeline, pipeline);
    return SLANG_OK;
}

RayTracingPipelineImpl::RayTracingPipelineImpl(Device* device, const RayTracingPipelineDesc& desc)
    : RayTracingPipeline(device, desc)
{
}

RayTracingPipelineImpl::~RayTracingPipelineImpl()
{
    DeviceImpl* device = getDevice<DeviceImpl>();

    if (m_pipeline != VK_NULL_HANDLE)
    {
        device->m_api.vkDestroyPipeline(device->m_api.m_device, m_pipeline, nullptr);
    }
}

Result RayTracingPipelineImpl::getNativeHandle(NativeHandle* outHandle)
{
    outHandle->type = NativeHandleType::VkPipeline;
    outHandle->value = (uint64_t)m_pipeline;
    return SLANG_OK;
}

inline uint32_t findEntryPointIndexByName(
    const std::map<std::string, uint32_t>& entryPointIndexByName,
    const char* name
)
{
    if (!name)
        return VK_SHADER_UNUSED_KHR;

    auto it = entryPointIndexByName.find(name);
    if (it != entryPointIndexByName.end())
        return it->second;
    // TODO: Error reporting?
    return VK_SHADER_UNUSED_KHR;
}


Result DeviceImpl::createRayTracingPipeline2(const RayTracingPipelineDesc& desc, IRayTracingPipeline** outPipeline)
{
    TimePoint startTime = Timer::now();

    ShaderProgramImpl* program = checked_cast<ShaderProgramImpl*>(desc.program);
    SLANG_RHI_ASSERT(!program->m_modules.empty());

    VkRayTracingPipelineCreateInfoKHR createInfo = {VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    createInfo.flags = translateRayTracingPipelineFlags(desc.flags);

    VkPipelineCreateFlags2CreateInfoKHR createFlags2Info = {VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR};
    createFlags2Info.flags = translateRayTracingPipelineFlags2(desc.flags);
    if (createFlags2Info.flags != createInfo.flags)
    {
        createInfo.flags = 0; // Unused
        createInfo.pNext = &createFlags2Info;
    }

    VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV clusterCreateInfo = {
        VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV
    };
    if (is_set(desc.flags, RayTracingPipelineFlags::EnableClusters))
    {
        clusterCreateInfo.allowClusterAccelerationStructure = VK_TRUE;
        clusterCreateInfo.pNext = (void*)createInfo.pNext;
        createInfo.pNext = &clusterCreateInfo;
    }

    createInfo.stageCount = (uint32_t)program->m_stageCreateInfos.size();
    createInfo.pStages = program->m_stageCreateInfos.data();

    // Build a map from entry point name to entry point index
    // (which matches stageCreateInfos index) for all entry points.
    std::map<std::string, uint32_t> entryPointIndexByName;
    for (uint32_t i = 0; i < createInfo.stageCount; ++i)
    {
        entryPointIndexByName.emplace(program->m_modules[i].entryPointName, i);
    }

    std::map<std::string, uint32_t> shaderGroupIndexByName;
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroupInfos;

    // Create shader groups for ray generation, miss and callable shaders.
    for (uint32_t i = 0; i < createInfo.stageCount; ++i)
    {
        auto entryPointName = program->m_modules[i].entryPointName;
        auto stageCreateInfo = program->m_stageCreateInfos[i];
        if (!(stageCreateInfo.stage &
              (VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CALLABLE_BIT_KHR)))
            continue;

        VkRayTracingShaderGroupCreateInfoKHR shaderGroupInfo = {
            VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR
        };
        shaderGroupInfo.pNext = nullptr;
        shaderGroupInfo.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        shaderGroupInfo.generalShader = i;
        shaderGroupInfo.closestHitShader = VK_SHADER_UNUSED_KHR;
        shaderGroupInfo.anyHitShader = VK_SHADER_UNUSED_KHR;
        shaderGroupInfo.intersectionShader = VK_SHADER_UNUSED_KHR;
        shaderGroupInfo.pShaderGroupCaptureReplayHandle = nullptr;

        // For groups with a single entry point, the group name is the entry point name.
        auto shaderGroupName = entryPointName;
        uint32_t shaderGroupIndex = (uint32_t)shaderGroupInfos.size();
        shaderGroupInfos.push_back(shaderGroupInfo);
        shaderGroupIndexByName.emplace(shaderGroupName, shaderGroupIndex);
    }

    // Create shader groups for hit groups.
    for (uint32_t i = 0; i < desc.hitGroupCount; ++i)
    {
        VkRayTracingShaderGroupCreateInfoKHR shaderGroupInfo = {
            VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR
        };
        auto& groupDesc = desc.hitGroups[i];

        shaderGroupInfo.pNext = nullptr;
        shaderGroupInfo.type = groupDesc.intersectionEntryPoint
                                   ? VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
                                   : VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        shaderGroupInfo.generalShader = VK_SHADER_UNUSED_KHR;
        shaderGroupInfo.closestHitShader =
            findEntryPointIndexByName(entryPointIndexByName, groupDesc.closestHitEntryPoint);
        shaderGroupInfo.anyHitShader = findEntryPointIndexByName(entryPointIndexByName, groupDesc.anyHitEntryPoint);
        shaderGroupInfo.intersectionShader =
            findEntryPointIndexByName(entryPointIndexByName, groupDesc.intersectionEntryPoint);
        shaderGroupInfo.pShaderGroupCaptureReplayHandle = nullptr;

        uint32_t shaderGroupIndex = (uint32_t)shaderGroupInfos.size();
        shaderGroupInfos.push_back(shaderGroupInfo);
        shaderGroupIndexByName.emplace(groupDesc.hitGroupName, shaderGroupIndex);
    }

    createInfo.groupCount = (uint32_t)shaderGroupInfos.size();
    createInfo.pGroups = shaderGroupInfos.data();

    createInfo.maxPipelineRayRecursionDepth = desc.maxRecursion;

    createInfo.pLibraryInfo = nullptr;
    createInfo.pLibraryInterface = nullptr;

    createInfo.pDynamicState = nullptr;

    createInfo.layout = program->m_rootShaderObjectLayout->m_pipelineLayout;
    createInfo.basePipelineHandle = VK_NULL_HANDLE;
    createInfo.basePipelineIndex = 0;

    SHA1 pipelineIdentity;
    hashProgramIdentity(pipelineIdentity, program);
    auto hashString = [&](const char* str)
    {
        if (str)
            pipelineIdentity.update(str, strlen(str));
        pipelineIdentity.update("|", 1);
    };
    for (uint32_t i = 0; i < desc.hitGroupCount; i++)
    {
        hashString(desc.hitGroups[i].hitGroupName);
        hashString(desc.hitGroups[i].closestHitEntryPoint);
        hashString(desc.hitGroups[i].anyHitEntryPoint);
        hashString(desc.hitGroups[i].intersectionEntryPoint);
    }
    pipelineIdentity.update(&desc.maxRecursion, sizeof(desc.maxRecursion));
    pipelineIdentity.update(&desc.maxRayPayloadSize, sizeof(desc.maxRayPayloadSize));
    pipelineIdentity.update(&desc.maxAttributeSizeInBytes, sizeof(desc.maxAttributeSizeInBytes));
    pipelineIdentity.update(&desc.flags, sizeof(desc.flags));

    VkPipeline vkPipeline = VK_NULL_HANDLE;
    ComPtr<ISlangBlob> cacheKey;
    bool cached = false;
    size_t cacheSize = 0;
    SLANG_RETURN_ON_FAIL(
        createPipelineWithCache<VkRayTracingPipelineCreateInfoKHR>(
            this,
            &createInfo,
            pipelineIdentity,
            [](DeviceImpl* device, VkRayTracingPipelineCreateInfoKHR* createInfo2, VkPipeline* pipeline) -> VkResult
            {
                return device->m_api.vkCreateRayTracingPipelinesKHR(
                    device->m_device,
                    VK_NULL_HANDLE,
                    VK_NULL_HANDLE,
                    1,
                    createInfo2,
                    nullptr,
                    pipeline
                );
            },
            &vkPipeline,
            cached,
            cacheSize,
            cacheKey
        )
    );

    _labelObject((uint64_t)vkPipeline, VK_OBJECT_TYPE_PIPELINE, desc.label);

    // Report the pipeline creation time.
    if (m_shaderCompilationReporter)
    {
        m_shaderCompilationReporter->reportCreatePipeline(
            program,
            ShaderCompilationReporter::PipelineType::RayTracing,
            startTime,
            Timer::now(),
            cached,
            cacheSize,
            cacheKey
        );
    }

    RefPtr<RayTracingPipelineImpl> pipeline = new RayTracingPipelineImpl(this, desc);
    pipeline->m_program = program;
    pipeline->m_rootObjectLayout = program->m_rootShaderObjectLayout;
    pipeline->m_pipeline = vkPipeline;
    pipeline->m_entryPointIndexByName = std::move(entryPointIndexByName);
    pipeline->m_shaderGroupIndexByName = std::move(shaderGroupIndexByName);
    pipeline->m_shaderGroupCount = shaderGroupInfos.size();
    returnComPtr(outPipeline, pipeline);
    return SLANG_OK;
}

} // namespace rhi::vk
