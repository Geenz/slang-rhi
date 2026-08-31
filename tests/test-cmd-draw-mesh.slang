struct MeshOutput
{
    float4 position : SV_Position;
    float4 color : COLOR;
};

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
[shader("mesh")]
void meshMain(out vertices MeshOutput verts[3], out indices uint3 tris[1])
{
    SetMeshOutputCounts(3, 1);

    MeshOutput v0;
    v0.position = float4(-1, -1, 0.5, 1);
    v0.color = float4(1, 0, 0, 1);
    verts[0] = v0;

    MeshOutput v1;
    v1.position = float4(1, -1, 0.5, 1);
    v1.color = float4(0, 1, 0, 1);
    verts[1] = v1;

    MeshOutput v2;
    v2.position = float4(0, 1, 0.5, 1);
    v2.color = float4(0, 0, 1, 1);
    verts[2] = v2;

    tris[0] = uint3(0, 1, 2);
}

[shader("fragment")]
float4 fragmentMain(float4 color : COLOR) : SV_Target
{
    return color;
}
