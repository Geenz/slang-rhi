struct Payload
{
    float4 color;
};

[numthreads(1, 1, 1)]
[shader("amplification")]
void objectMain(in uint3 dtid : SV_DispatchThreadID)
{
    Payload p;
    p.color = float4(1, 0, 0, 1);
    DispatchMesh(1, 1, 1, p);
}

struct MeshOutput
{
    float4 position : SV_Position;
    float4 color : COLOR;
};

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
[shader("mesh")]
void meshMain(in payload Payload p, out vertices MeshOutput verts[3], out indices uint3 tris[1])
{
    SetMeshOutputCounts(3, 1);

    MeshOutput v0;
    v0.position = float4(-1, -1, 0.5, 1);
    v0.color = p.color;
    verts[0] = v0;

    MeshOutput v1;
    v1.position = float4(1, -1, 0.5, 1);
    v1.color = p.color;
    verts[1] = v1;

    MeshOutput v2;
    v2.position = float4(0, 1, 0.5, 1);
    v2.color = p.color;
    verts[2] = v2;

    tris[0] = uint3(0, 1, 2);
}

[shader("fragment")]
float4 fragmentMain(float4 color : COLOR) : SV_Target
{
    return color;
}
