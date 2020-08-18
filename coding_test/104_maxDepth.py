from common import TreeNode,gen_root
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # 减少参数传递,效果和解法1差不多
        def dfs(node):
            if node is None:
                # 叶子节点的高度都为0
                return 0
            # 其他节点的高度为下一层高度加1
            return max(dfs(node.left), dfs(node.right)) + 1

        return dfs(root)
    def maxDepth1(self, root: TreeNode) -> int:
        # 深度优先
        def dfs(node, depth):
            # 左右子树都不为空
            if not node:
                return depth
            else:
                return max(dfs(node.left, depth + 1), dfs(node.right, depth + 1))
        return dfs(root, 0)


root = gen_root()
s = Solution()
print(s.maxDepth(root))
