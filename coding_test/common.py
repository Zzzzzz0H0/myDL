#  Definition for a binary tree node.
class TreeNode:
   def __init__(self, x):
       self.val = x
       self.left = None
       self.right = None

def gen_root():
    root = TreeNode(4)
    node1 = TreeNode(1)
    node2 = TreeNode(3)
    root.right = node2
    root.left = node1
    return root
