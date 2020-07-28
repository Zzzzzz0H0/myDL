from common import TreeNode, gen_root
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    """
    递归实现:
    每次将镜像的2个节点进行判断，必须值相等，并且这2个节点的自节点也是镜像的
    """
    def isSymmetric(self, root: TreeNode) -> bool:
        if root is None:
            return True
        def dfs(node1, node2):
            if (node1 is None) and (node2 is None):
                return True
            elif (node1 is None ) and (node2 is not None):
                return False
            elif (node1 is not None) and (node2 is None ):
                return False
            else:
                return (node1.val == node2.val) and (dfs(node1.left, node2.right) and (dfs(node1.right, node2.left)))
        return dfs(root.left, root.right)

root = gen_root()
s = Solution()
print(s.isSymmetric(root))
