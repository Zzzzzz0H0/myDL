from functools import lru_cache
class Solution:
    def numTrees(self, n: int) -> int:
        """
        动态规划：
        1. 目标n个节点的树二叉搜索树共有多少种。
        2. 状态迁移
        求解二叉树的种数为分别以i为顶点的二叉树种树之和，i的取值范围为（0,n）
        以i为顶点的二叉树种数等于左半子树的种树乘以右半子树的种树
        3. 初始状态0个节点的种数只有1种，1个节点的种数也只有1种。
        """
        # 使用记忆缓存，参数需要可哈希
        @lru_cache
        def f(n):
            if n == 0 or n == 1:
                return 1
            return sum( f(i) * f(n-i-1)  for i in range(n))
        return f(n)




s = Solution()
n = 4
print(s.numTrees(n))
