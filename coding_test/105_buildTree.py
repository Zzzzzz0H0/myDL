from common import TreeNode
from typing import List

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def splitTree(preorder, inorder):
            if len(preorder) == 0:
                return None
            # 先序的第一个元素为该二叉树的中间节点
            val = preorder[0]
            index = inorder.index(val)
            # 生成中间节点
            node = TreeNode(val)
            # 左子树
            node.left = splitTree(preorder[1:index+1], inorder[:index])
            # 右子树
            node.right = splitTree(preorder[index+1:], inorder[index+1:])
            return node
        return splitTree(preorder, inorder)

s = Solution()
preorder = [3,9,20,15,7]
inorder =[9,3,15,20,7]
print(s.buildTree(preorder, inorder))
