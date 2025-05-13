import torch
from core.sac.buffers.segment_tree import SegmentTree, SumTree

def test_segment_tree_sum_and_update():
    tree = SegmentTree(capacity=8, operation="sum", neutral_element=0.0)
    for i in range(8):
        tree.tree[tree.capacity - 1 + i] = float(i + 1)
    for i in reversed(range(tree.capacity - 1)):
        tree.tree[i] = tree.tree[2 * i + 1] + tree.tree[2 * i + 2]
    assert abs(tree.total() - sum(range(1, 9))) < 1e-5
    tree.tree[tree.capacity - 1] = 100.0
    for i in reversed(range(tree.capacity - 1)):
        tree.tree[i] = tree.tree[2 * i + 1] + tree.tree[2 * i + 2]
    assert tree.total() > 100

def test_sum_tree_api():
    tree = SumTree(capacity=4)
    tree.update(0, 1.0)
    tree.update(1, 2.0)
    tree.update(2, 3.0)
    tree.update(3, 4.0)
    assert abs(tree.sum() - 10.0) < 1e-5
    tree.clear()
    assert tree.sum() == 0.0