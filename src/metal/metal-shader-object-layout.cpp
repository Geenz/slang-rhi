#include "metal-shader-object-layout.h"
#include <cstdio>

namespace rhi::metal {

/// Flatten the recursive writeOrdinaryDataIntoArgumentBuffer walk into pre-computed copy commands.
static void flattenOrdinaryDataCopies(
    slang::TypeLayoutReflection* argLayout,
    slang::TypeLayoutReflection* srcLayout,
    uint32_t argBase,
    uint32_t srcBase,
    std::vector<FlatArgBufferDataCopy>& outCopies
)
{
    if (srcLayout->getCategoryCount() == 1)
    {
        if (srcLayout->getCategoryByIndex(0) == slang::ParameterCategory::Uniform)
        {
            uint32_t size = (uint32_t)srcLayout->getSize();
            if (size > 0)
                outCopies.push_back({srcBase, argBase, size});
        }
        return;
    }

    for (unsigned int i = 0; i < argLayout->getFieldCount(); i++)
    {
        auto argField = argLayout->getFieldByIndex(i);
        auto srcField = srcLayout->getFieldByIndex(i);
        flattenOrdinaryDataCopies(
            argField->getTypeLayout(),
            srcField->getTypeLayout(),
            argBase + (uint32_t)argField->getOffset(),
            srcBase + (uint32_t)srcField->getOffset(),
            outCopies
        );
    }
}

static slang::TypeLayoutReflection* _getParameterBlockTypeLayout(
    slang::ISession* slangSession,
    slang::TypeLayoutReflection* elementTypeLayout
)
{
    return slangSession->getTypeLayout(elementTypeLayout->getType(), 0, slang::LayoutRules::MetalArgumentBufferTier2);
}

ShaderObjectLayoutImpl::SubObjectRangeOffset::SubObjectRangeOffset(slang::VariableLayoutReflection* varLayout)
    : BindingOffset(varLayout)
{
}

ShaderObjectLayoutImpl::SubObjectRangeStride::SubObjectRangeStride(slang::TypeLayoutReflection* typeLayout)
    : BindingOffset(typeLayout)
{
}

Result ShaderObjectLayoutImpl::Builder::setElementTypeLayout(slang::TypeLayoutReflection* typeLayout)
{
    typeLayout = _unwrapParameterGroups(typeLayout, m_containerType);

    m_elementTypeLayout = typeLayout;

    if (m_containerType == ShaderObjectContainerType::ParameterBlock)
    {
        m_parameterBlockTypeLayout = _getParameterBlockTypeLayout(m_session, m_elementTypeLayout);

        // If we have a parameter-block, we should be working on the `ParameterBlockTypeLayout`
        // since this layout will format data for an arg-buffer-tier2 if available.
        typeLayout = m_parameterBlockTypeLayout;
    }
    m_totalOrdinaryDataSize = (uint32_t)typeLayout->getSize();
    if (m_totalOrdinaryDataSize > 0)
    {
        m_resourceCount.buffer++;
    }

    // Compute the binding ranges that are used to store
    // the logical contents of the object in memory.

    SlangInt bindingRangeCount = typeLayout->getBindingRangeCount();
    for (SlangInt r = 0; r < bindingRangeCount; ++r)
    {
        slang::BindingType slangBindingType = typeLayout->getBindingRangeType(r);
        SlangInt count = typeLayout->getBindingRangeBindingCount(r);
        slang::TypeLayoutReflection* slangLeafTypeLayout = typeLayout->getBindingRangeLeafTypeLayout(r);

        uint32_t slotIndex = 0;
        uint32_t subObjectIndex = 0;

        switch (slangBindingType)
        {
        case slang::BindingType::ConstantBuffer:
        case slang::BindingType::ParameterBlock:
        case slang::BindingType::ExistentialValue:
            subObjectIndex = m_subObjectCount;
            m_subObjectCount += count;
            break;
        case slang::BindingType::RawBuffer:
        case slang::BindingType::MutableRawBuffer:
            slotIndex = m_slotCount;
            if (slangLeafTypeLayout->getType()->getElementType() != nullptr)
            {
                // A structured buffer occupies both a resource slot and
                // a sub-object slot.
                subObjectIndex = m_subObjectCount;
                m_subObjectCount += count;
            }
            m_slotCount += count;
            m_resourceCount.buffer += count;
            break;
        case slang::BindingType::Sampler:
            slotIndex = m_slotCount;
            m_slotCount += count;
            m_resourceCount.sampler += count;
            break;
        case slang::BindingType::Texture:
        case slang::BindingType::MutableTexture:
            slotIndex = m_slotCount;
            m_slotCount += count;
            m_resourceCount.texture += count;
            break;
        case slang::BindingType::TypedBuffer:
        case slang::BindingType::MutableTypedBuffer:
            slotIndex = m_slotCount;
            m_slotCount += count;
            m_resourceCount.buffer += count;
            break;
        case slang::BindingType::RayTracingAccelerationStructure:
            slotIndex = m_slotCount;
            m_slotCount += count;
            break;
        default:
            break;
        }

        BindingRangeInfo bindingRangeInfo;
        bindingRangeInfo.bindingType = slangBindingType;
        bindingRangeInfo.count = count;
        bindingRangeInfo.slotIndex = slotIndex;
        bindingRangeInfo.subObjectIndex = subObjectIndex;

        // We'd like to extract the information on the Metal resource
        // index that this range should bind into.
        //
        // A binding range represents a logical member of the shader
        // object type, and it may encompass zero or more *descriptor
        // ranges* that describe how it is physically bound to pipeline
        // state.
        //
        // If the current binding range is backed by at least one descriptor
        // range then we can query the register offset of that descriptor
        // range. We expect that in the common case there will be exactly
        // one descriptor range, and we can extract the information easily.
        //
        // TODO: we might eventually need to special-case our handling
        // of combined texture-sampler ranges since they will need to
        // store two different offsets.
        //
        if (typeLayout->getBindingRangeDescriptorRangeCount(r) != 0)
        {
            // The Slang reflection information organizes the descriptor ranges
            // into "descriptor sets" but Metal has no notion like that so we
            // expect all ranges belong to a single set.
            //
            SlangInt descriptorSetIndex = typeLayout->getBindingRangeDescriptorSetIndex(r);
            SLANG_RHI_ASSERT(descriptorSetIndex == 0);

            SlangInt descriptorRangeIndex = typeLayout->getBindingRangeFirstDescriptorRangeIndex(r);
            auto registerOffset =
                typeLayout->getDescriptorSetDescriptorRangeIndexOffset(descriptorSetIndex, descriptorRangeIndex);

            bindingRangeInfo.registerOffset = (uint32_t)registerOffset;
        }

        m_bindingRanges.push_back(bindingRangeInfo);
    }

    m_totalResourceCount = m_resourceCount;

    SlangInt subObjectRangeCount = typeLayout->getSubObjectRangeCount();
    for (SlangInt r = 0; r < subObjectRangeCount; ++r)
    {
        SlangInt bindingRangeIndex = typeLayout->getSubObjectRangeBindingRangeIndex(r);

        auto slangBindingType = typeLayout->getBindingRangeType(bindingRangeIndex);
        slang::TypeLayoutReflection* slangLeafTypeLayout = typeLayout->getBindingRangeLeafTypeLayout(bindingRangeIndex);

        SubObjectRangeInfo subObjectRange;
        subObjectRange.bindingRangeIndex = bindingRangeIndex;

        // We will use Slang reflection information to extract the offset and stride
        // information for each sub-object range.
        //
        subObjectRange.offset = SubObjectRangeOffset(typeLayout->getSubObjectRangeOffset(r));
        subObjectRange.stride = SubObjectRangeStride(slangLeafTypeLayout);

        // A sub-object range can either represent a sub-object of a known
        // type, like a `ConstantBuffer<Foo>` or `ParameterBlock<Foo>`
        // *or* it can represent a sub-object of some existential type (e.g., `IBar`).
        //
        RefPtr<ShaderObjectLayoutImpl> subObjectLayout;
        switch (slangBindingType)
        {
        default:
        {
            auto elementTypeLayout = slangLeafTypeLayout->getElementTypeLayout();
            createForElementType(m_device, m_session, elementTypeLayout, subObjectLayout.writeRef());
        }
        break;
        case slang::BindingType::ConstantBuffer:
        {
            // In the case of `ConstantBuffer<X>` or `cbuffer`
            // we can construct a layout from the element type directly.
            auto elementTypeLayout = slangLeafTypeLayout->getElementTypeLayout();
            createForElementType(m_device, m_session, elementTypeLayout, subObjectLayout.writeRef());
            break;
        }
        case slang::BindingType::ParameterBlock:
            // On metal, ParameterBlock is represented as a single argument buffer.
            // We will let _unwrapParameterGroups to handle the dereference logic.
            createForElementType(m_device, m_session, slangLeafTypeLayout, subObjectLayout.writeRef());
            break;
        case slang::BindingType::ExistentialValue:
            // In the case of an interface-type sub-object range, we can only
            // construct a layout if we have static specialization information
            // that tells us what type we expect to find in that range.
            //
            // Pending data layout APIs have been removed.
            // Interface-type ranges now have no additional layout information.
            // Sub-object layout remains nullptr for interface types.
            break;
        }
        subObjectRange.layout = subObjectLayout;

        m_subObjectRanges.push_back(subObjectRange);

        if (subObjectLayout && slangBindingType != slang::BindingType::ParameterBlock)
        {
            m_totalResourceCount += subObjectLayout->m_totalResourceCount;
        }
    }
    return SLANG_OK;
}

Result ShaderObjectLayoutImpl::Builder::build(ShaderObjectLayoutImpl** outLayout)
{
    auto layout = RefPtr<ShaderObjectLayoutImpl>(new ShaderObjectLayoutImpl());
    SLANG_RETURN_ON_FAIL(layout->_init(this));

    returnRefPtrMove(outLayout, layout);
    return SLANG_OK;
}

slang::TypeLayoutReflection* ShaderObjectLayoutImpl::getParameterBlockTypeLayout()
{
    if (!m_parameterBlockTypeLayout)
        m_parameterBlockTypeLayout = _getParameterBlockTypeLayout(m_slangSession.get(), m_elementTypeLayout);
    return m_parameterBlockTypeLayout;
}

Result ShaderObjectLayoutImpl::createForElementType(
    Device* device,
    slang::ISession* session,
    slang::TypeLayoutReflection* elementType,
    ShaderObjectLayoutImpl** outLayout
)
{
    Builder builder(device, session);
    builder.setElementTypeLayout(elementType);
    return builder.build(outLayout);
}

Result ShaderObjectLayoutImpl::_init(const Builder* builder)
{
    auto device = builder->m_device;

    initBase(device, builder->m_session, builder->m_elementTypeLayout);

    m_parameterBlockTypeLayout = builder->m_parameterBlockTypeLayout;
    m_slotCount = builder->m_slotCount;
    m_subObjectCount = builder->m_subObjectCount;
    m_resourceCount = builder->m_resourceCount;
    m_totalResourceCount = builder->m_totalResourceCount;

    m_bindingRanges = builder->m_bindingRanges;
    m_subObjectRanges = builder->m_subObjectRanges;

    m_totalOrdinaryDataSize = builder->m_totalOrdinaryDataSize;

    m_containerType = builder->m_containerType;
    return SLANG_OK;
}

Result RootShaderObjectLayoutImpl::Builder::build(RootShaderObjectLayoutImpl** outLayout)
{
    RefPtr<RootShaderObjectLayoutImpl> layout = new RootShaderObjectLayoutImpl();
    SLANG_RETURN_ON_FAIL(layout->_init(this));

    returnRefPtrMove(outLayout, layout);
    return SLANG_OK;
}

void RootShaderObjectLayoutImpl::Builder::addGlobalParams(slang::VariableLayoutReflection* globalsLayout)
{
    setElementTypeLayout(globalsLayout->getTypeLayout());
}

void RootShaderObjectLayoutImpl::Builder::addEntryPoint(
    SlangStage stage,
    ShaderObjectLayoutImpl* entryPointLayout,
    slang::EntryPointLayout* slangEntryPoint
)
{
    EntryPointInfo info;
    info.layout = entryPointLayout;
    info.offset = BindingOffset(slangEntryPoint->getVarLayout());
    m_entryPoints.push_back(info);
    m_totalResourceCount += entryPointLayout->m_totalResourceCount;
}

Result RootShaderObjectLayoutImpl::create(
    Device* device,
    slang::IComponentType* program,
    slang::ProgramLayout* programLayout,
    RootShaderObjectLayoutImpl** outLayout
)
{
    RootShaderObjectLayoutImpl::Builder builder(device, program, programLayout);
    builder.addGlobalParams(programLayout->getGlobalParamsVarLayout());

    SlangInt entryPointCount = programLayout->getEntryPointCount();
    for (SlangInt e = 0; e < entryPointCount; ++e)
    {
        auto slangEntryPoint = programLayout->getEntryPointByIndex(e);
        RefPtr<ShaderObjectLayoutImpl> entryPointLayout;
        SLANG_RETURN_ON_FAIL(
            ShaderObjectLayoutImpl::createForElementType(
                device,
                program->getSession(),
                slangEntryPoint->getTypeLayout(),
                entryPointLayout.writeRef()
            )
        );
        builder.addEntryPoint(slangEntryPoint->getStage(), entryPointLayout, slangEntryPoint);
    }

    SLANG_RETURN_ON_FAIL(builder.build(outLayout));

    return SLANG_OK;
}

Result RootShaderObjectLayoutImpl::_init(const Builder* builder)
{
    SLANG_RETURN_ON_FAIL(Super::_init(builder));

    m_program = builder->m_program;
    m_programLayout = builder->m_programLayout;
    m_entryPoints = builder->m_entryPoints;
    m_slangSession = m_program->getSession();

    return SLANG_OK;
}

// ============================================================================
// Flat binding table builder
// ============================================================================

static void buildFlatEntries(
    FlatBindingTable& table,
    ShaderObjectLayoutImpl* layout,
    BindingOffset offset,
    uint16_t objectTreeIndex,
    bool isRoot
)
{
    // Record ordinary data for this object level (root handled externally).
    // NOTE: Do NOT increment offset.buffer here. In the recursive path,
    // bindAsConstantBuffer uses the ORIGINAL inOffset for bindAsValue,
    // not the offset modified by bindOrdinaryDataBufferIfNeeded.
    uint32_t ordinaryDataSize = layout->getTotalOrdinaryDataSize();
    if (ordinaryDataSize > 0 && !isRoot)
    {
        FlatOrdinaryDataEntry entry;
        entry.objectTreeIndex = objectTreeIndex;
        entry.bufferRegister = (uint16_t)offset.buffer;
        entry.dataSize = ordinaryDataSize;
        table.ordinaryData.push_back(entry);
    }

    // Record binding range entries (textures, buffers, samplers)
    for (const auto& bindingRangeInfo : layout->m_bindingRanges)
    {
        uint32_t count = bindingRangeInfo.count;
        switch (bindingRangeInfo.bindingType)
        {
        case slang::BindingType::Texture:
        case slang::BindingType::MutableTexture:
            for (uint32_t i = 0; i < count; ++i)
            {
                FlatBindingEntry entry;
                entry.objectTreeIndex = objectTreeIndex;
                entry.slotIndex = (uint16_t)(bindingRangeInfo.slotIndex + i);
                entry.registerIndex = (uint16_t)(bindingRangeInfo.registerOffset + offset.texture + i);
                entry.type = FlatBindingEntry::Type::Texture;
                entry.isMutable = (bindingRangeInfo.bindingType == slang::BindingType::MutableTexture);
                table.entries.push_back(entry);
            }
            break;

        case slang::BindingType::Sampler:
            for (uint32_t i = 0; i < count; ++i)
            {
                FlatBindingEntry entry;
                entry.objectTreeIndex = objectTreeIndex;
                entry.slotIndex = (uint16_t)(bindingRangeInfo.slotIndex + i);
                entry.registerIndex = (uint16_t)(bindingRangeInfo.registerOffset + offset.sampler + i);
                entry.type = FlatBindingEntry::Type::Sampler;
                entry.isMutable = false;
                table.entries.push_back(entry);
            }
            break;

        case slang::BindingType::RawBuffer:
        case slang::BindingType::MutableRawBuffer:
        case slang::BindingType::TypedBuffer:
        case slang::BindingType::MutableTypedBuffer:
            for (uint32_t i = 0; i < count; ++i)
            {
                FlatBindingEntry entry;
                entry.objectTreeIndex = objectTreeIndex;
                entry.slotIndex = (uint16_t)(bindingRangeInfo.slotIndex + i);
                entry.registerIndex = (uint16_t)(bindingRangeInfo.registerOffset + offset.buffer + i);
                entry.type = FlatBindingEntry::Type::Buffer;
                entry.isMutable = (bindingRangeInfo.bindingType == slang::BindingType::MutableRawBuffer ||
                                   bindingRangeInfo.bindingType == slang::BindingType::MutableTypedBuffer);
                table.entries.push_back(entry);
            }
            break;

        default:
            break;
        }
    }

    // Recurse into sub-object ranges
    for (const auto& subObjectRange : layout->m_subObjectRanges)
    {
        auto subObjectLayout = subObjectRange.layout;
        if (!subObjectLayout)
            continue;

        const auto& bindingRange = layout->m_bindingRanges[subObjectRange.bindingRangeIndex];
        if (bindingRange.bindingType == slang::BindingType::ParameterBlock)
        {
            // Pre-compute argument buffer layout for ParameterBlock sub-objects.
            auto pbLayout = subObjectRange.layout;
            if (!pbLayout) continue;

            auto argTypeLayout = pbLayout->getParameterBlockTypeLayout();
            if (!argTypeLayout || argTypeLayout->getFieldCount() == 0) continue;

            BindingOffset rangeOffset = offset;
            rangeOffset += subObjectRange.offset;

            for (uint32_t i = 0; i < bindingRange.count; ++i)
            {
                uint16_t subObjIndex = table.objectCount++;
                table.objectPaths.push_back({FlatObjectPath::Source::SubObject, objectTreeIndex,
                                             (uint16_t)(bindingRange.subObjectIndex + i)});

                FlatArgBufferDesc desc;
                desc.objectTreeIndex = subObjIndex;
                desc.bindingDataRegister = (uint16_t)rangeOffset.buffer;
                desc.bufferSize = (uint32_t)argTypeLayout->getSize();
                desc.dataSize = (uint32_t)pbLayout->getElementTypeLayout()->getSize();
                desc.firstEntry = (uint32_t)table.argBufferEntries.size();
                desc.entryCount = 0;
                desc.pbLayout = pbLayout;  // Per-PB layout; consumed by bindAsRootFlat.

                // Pre-compute uniform data copy commands (flattened from recursive layout walk)
                desc.firstDataCopy = (uint32_t)table.argBufferDataCopies.size();
                flattenOrdinaryDataCopies(
                    argTypeLayout, pbLayout->getElementTypeLayout(), 0, 0, table.argBufferDataCopies);
                desc.dataCopyCount = (uint32_t)table.argBufferDataCopies.size() - desc.firstDataCopy;

                // Pre-compute per-binding offsets within the argument buffer
                for (uint32_t bri = 0; bri < pbLayout->getBindingRangeCount(); ++bri)
                {
                    const auto& brInfo = pbLayout->m_bindingRanges[bri];
                    uint32_t count2 = brInfo.count;

                    FlatArgBufferBindingEntry::Type entryType;
                    bool isMutable = false;
                    bool skip = false;

                    switch (brInfo.bindingType)
                    {
                    case slang::BindingType::Texture:
                        entryType = FlatArgBufferBindingEntry::Type::Texture; break;
                    case slang::BindingType::MutableTexture:
                        entryType = FlatArgBufferBindingEntry::Type::Texture; isMutable = true; break;
                    case slang::BindingType::Sampler:
                        entryType = FlatArgBufferBindingEntry::Type::Sampler; break;
                    case slang::BindingType::RawBuffer:
                    case slang::BindingType::TypedBuffer:
                        entryType = FlatArgBufferBindingEntry::Type::Buffer; break;
                    case slang::BindingType::MutableRawBuffer:
                    case slang::BindingType::MutableTypedBuffer:
                        entryType = FlatArgBufferBindingEntry::Type::Buffer; isMutable = true; break;
                    case slang::BindingType::RayTracingAccelerationStructure:
                        entryType = FlatArgBufferBindingEntry::Type::AccelerationStructure; break;
                    default:
                        skip = true; break;
                    }

                    if (skip) continue;

                    SlangInt setIdx = argTypeLayout->getBindingRangeDescriptorSetIndex(bri);
                    SlangInt rangeIdx = argTypeLayout->getBindingRangeFirstDescriptorRangeIndex(bri);
                    SlangInt argOffset = argTypeLayout->getDescriptorSetDescriptorRangeIndexOffset(setIdx, rangeIdx);

                    for (uint32_t j = 0; j < count2; ++j)
                    {
                        FlatArgBufferBindingEntry entry;
                        entry.slotIndex = (uint16_t)(brInfo.slotIndex + j);
                        entry.byteOffset = (uint32_t)(argOffset + j * sizeof(uint64_t));
                        entry.type = entryType;
                        entry.isMutable = isMutable;
                        table.argBufferEntries.push_back(entry);
                        desc.entryCount++;
                    }
                }

                table.argBuffers.push_back(desc);
                rangeOffset += subObjectRange.stride;
            }
            continue;
        }

        if (bindingRange.bindingType != slang::BindingType::ConstantBuffer)
            continue;

        uint32_t count = bindingRange.count;
        BindingOffset rangeOffset = offset;
        rangeOffset += subObjectRange.offset;
        BindingOffset rangeStride = subObjectRange.stride;

        for (uint32_t i = 0; i < count; ++i)
        {
            uint16_t subObjIndex = table.objectCount++;
            table.objectPaths.push_back({FlatObjectPath::Source::SubObject, objectTreeIndex,
                                         (uint16_t)(bindingRange.subObjectIndex + i)});
            buildFlatEntries(table, subObjectLayout, rangeOffset, subObjIndex, false);
            rangeOffset += rangeStride;
        }
    }
}

const FlatBindingTable& RootShaderObjectLayoutImpl::getFlatBindingTable()
{
    // Thread-safe one-time build. Worker threads binding the same
    // pipeline's shader object for the first time all land here; the
    // per-layout `m_flatBindingTableOnce` makes exactly one thread run
    // the build while the rest wait, then all return the same table.
    std::call_once(m_flatBindingTableOnce, [this]() {
    m_flatBindingTable.objectCount = 1; // root = index 0
    m_flatBindingTable.objectPaths.push_back({FlatObjectPath::Source::Root, 0, 0});

    // Root level: ordinary data offset is special (absolute, not relative)
    uint32_t rootOrdinaryDataSize = getTotalOrdinaryDataSize();
    if (rootOrdinaryDataSize > 0)
    {
        FlatOrdinaryDataEntry entry;
        entry.objectTreeIndex = 0;
        entry.bufferRegister = 0;
        entry.dataSize = rootOrdinaryDataSize;
        m_flatBindingTable.ordinaryData.push_back(entry);
        // Root ordinary data buffer offset is handled separately in bindAsRoot
    }

    // Build entries for root's own binding ranges (with offset 0)
    // Then recurse into sub-objects
    buildFlatEntries(m_flatBindingTable, this, BindingOffset(), 0, true);

    // Entry points
    for (size_t i = 0; i < m_entryPoints.size(); ++i)
    {
        auto* epLayout = m_entryPoints[i].layout.get();
        BindingOffset epOffset = m_entryPoints[i].offset;
        uint16_t epIndex = m_flatBindingTable.objectCount++;
        m_flatBindingTable.objectPaths.push_back({FlatObjectPath::Source::EntryPoint, 0, (uint16_t)i});
        buildFlatEntries(m_flatBindingTable, epLayout, epOffset, epIndex, false);
    }

    // Compute right-sized array bounds from the flat table
    for (const auto& e : m_flatBindingTable.entries)
    {
        if (e.type == FlatBindingEntry::Type::Buffer)
        {
            m_flatBindingTable.maxBufferRegister = std::max(m_flatBindingTable.maxBufferRegister,
                                                            (uint16_t)(e.registerIndex + 1));
            if (e.isMutable) m_flatBindingTable.maxUsedRWResources++;
            else m_flatBindingTable.maxUsedResources++;
        }
        else if (e.type == FlatBindingEntry::Type::Texture)
        {
            m_flatBindingTable.maxTextureRegister = std::max(m_flatBindingTable.maxTextureRegister,
                                                             (uint16_t)(e.registerIndex + 1));
            if (e.isMutable) m_flatBindingTable.maxUsedRWResources++;
            else m_flatBindingTable.maxUsedResources++;
        }
        else if (e.type == FlatBindingEntry::Type::Sampler)
        {
            // samplers don't need usedResource tracking
        }
    }
    // Account for ordinary data buffers and argument buffers in buffer register count
    for (const auto& od : m_flatBindingTable.ordinaryData)
        m_flatBindingTable.maxBufferRegister = std::max(m_flatBindingTable.maxBufferRegister,
                                                        (uint16_t)(od.bufferRegister + 1));
    for (const auto& ab : m_flatBindingTable.argBuffers)
    {
        m_flatBindingTable.maxBufferRegister = std::max(m_flatBindingTable.maxBufferRegister,
                                                        (uint16_t)(ab.bindingDataRegister + 1));
        // Don't add full entryCount — most entries in large bindless arrays are
        // unbound (descriptor handles, not resource bindings). Those get residency
        // via the bindless descriptor set's bulk useResources. Add a reasonable
        // cap per arg buffer for the directly-bound resources.
        m_flatBindingTable.maxUsedResources += std::min(ab.entryCount, (uint32_t)2048);
    }

    m_flatBindingTable.built = true;
    });

    return m_flatBindingTable;
}

} // namespace rhi::metal
