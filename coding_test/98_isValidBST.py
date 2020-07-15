#  Definition for a binary tree node.
from common import TreeNode,gen_root
class Solution:
    def isValidBST1(self, root) -> bool:
        """
        满足二叉树搜索树说明该二叉树的中序遍历是单调递增的
        使用栈中序遍历该二叉树，然后看是否是有序的
        """
        stack = [root]
        vals = []
        while(stack):
            node = stack.pop()
            if isinstance(node, TreeNode):
                stack.extend([node.right, node.val, node.left])
            if isinstance(node, int):
                vals.append(node)
        # 需要考虑重复元素，二叉搜索树是没有重复元素的
        return vals == sorted(list(set(vals)))

    def isValidBST2(self, root) -> bool:
        """
        优化：使用递归判断
        对root左边的节点，val值都应该满足大于最小值，并且小于上层节点的最小值，
        对root右边的节点，val值都应该满足小于最大值，并且大于上层节点的最大值，
        最左边的一条线，永远单调递减，
        最右边的一条线，永远单调递增。
        """
        def dfs(node, left_val, right_val):
            if node is None:
                return True
            print(node.val, left_val, right_val)
            return (left_val < node.val < right_val) and \
                   (dfs(node.left, left_val, node.val)) and \
                   (dfs(node.right, node.val, right_val))
        # 使用float('-inf')和float('inf')来表示最大值和最小值
        return dfs(root, float('-inf'), float('inf'))

    def isValidBST(self, root) -> bool:
        """
        算法2并没有比算法1快多少。。
        优化算法1，实时进行比较
        提交后发现也没有快多少。。
        """
        stack = [root]
        # 需要实时更新的最小值
        min_val = float('-inf')
        while(stack):
            node = stack.pop()
            if isinstance(node, TreeNode):
                stack.extend([node.right, node.val, node.left])
            if isinstance(node, int):
                if node <= min_val:
                    # 应该为单调递增
                    return False
                else:
                    # 更新最小值
                    min_val = node
        # 需要考虑重复元素，二叉搜索树是没有重复元素的
        return True

            

s = Solution()
root = gen_root()
print(s.isValidBST(root))


if __name__ == "__main__":
    pass
