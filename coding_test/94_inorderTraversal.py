from typing import List
# Definition for a binary tree node.
class TreeNode:
   def __init__(self, x):
       self.val = x
       self.left = None
       self.right = None

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        # 递归解法
        # 中序遍历：先left,再中间，后right
        if root is None:
            return []
        result = []
        def dfs(node):
            if node.left is not None:
                dfs(node.left)
            result.append(node.val)
            if node.right is not None:
                dfs(node.right)
        dfs(root)
        return result
            
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        """
        借用栈将二叉树的所有节点入栈再出栈，即可遍历所有节点。
        初始化时所有节点都是未标记的，需要遍历节点的左右节点时，将
        其标记为已读状态，并且可以根据先序，中序，后序入栈，遇到叶子节点
        则开始出栈，出栈时对已经标记的节点（说明它的子节点已经遍历完）
        直接输出即可，直到栈空为止。
        tirck: python类型的节点可以根据节点值为TreeNode还是int区分是否是
        已读节点。
        """
        # 初始化栈
        stack = [root]
        result = []
        while(stack):
            # 栈顶元素
            node = stack.pop()
            # 未读节点
            if isinstance(node, TreeNode):
                # 中序遍历，入栈顺序为right，val, left(与出栈顺序相反)
                stack.extend([node.right, node.val, node.left])
            # 已读节点,输出
            if isinstance(node,int):
                result.append(node)
        return result





def gen_root():
    root = TreeNode(1)
    node1 = TreeNode(2)
    node2 = TreeNode(3)
    root.right = node1
    node1.left = node2
    return root
        
s = Solution()
l = [1, None, 2, 3]
root = gen_root()
print(s.inorderTraversal(root))

