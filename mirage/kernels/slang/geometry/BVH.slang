// Mirrors mirage/mirage/core/BVH.h's BVHNode layout (host-side builder stays in
// C++ - this is only the flat node struct + fetch, consumed by traversal code
// in Trace.slang). GpuLayout.h's GpuBVHNode is the byte-identical C++ mirror.

struct BVHNode
{
    float3 lower;
    float3 upper;
    uint leftIndex;
    uint rightIndexAndLeaf; // bits[0:30] = rightIndex, bit 31 = leaf

    bool IsLeaf() { return (rightIndexAndLeaf & 0x80000000u) != 0u; }
    uint RightIndex() { return rightIndexAndLeaf & 0x7FFFFFFFu; }
};

BVHNode FetchNode(StructuredBuffer<BVHNode> nodes, uint baseOffset, uint index)
{
    return nodes[baseOffset + index];
}
