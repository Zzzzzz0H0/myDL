from typing import List
class Solution:
    # 深度优先搜索解法
    def generateParenthesis1(self, n: int) -> List[str]:
        result = []
        def dfs(cur_str, left, right):
            """dfs

            :param cur_str: 当前字符串状态
            :param left: 剩余的左括号数量
            :param right: 剩余的有括号数量
            """
            if left == 0 and right == 0:
                result.append(cur_str)
                return 
            if left > 0:
                dfs(cur_str + '(', left - 1, right)
            if right > 0 and left < right:
                dfs(cur_str + ')', left, right - 1)
        dfs('', n, n)
        return result
    
    def generateParenthesis(self, n: int) -> List[str]:
        # 动态回归法
        # i个括号的组合等于（）加j个括号的组合加i -j -1 个括号的组合之和。
        dp = [None for i in range(n+1)]
        # 0个括号为一个元素，否则无法算出dp[1]
        dp[0] = ['']
        # 遍历算出n个括号时的所有组合,返回dp[n]
        for i in range(1, n+1):
            dpi = []
            # 对i个括号来说，需要0到i-1个括号的组合加（）再加剩余括号的组合
            for j in range(i):
                left = dp[j]
                right = dp[i - j -1]
                for li in left:
                    for ri in right:
                        dpi.append('(' + li + ')' + ri)
            dp[i] = dpi
        return dp[n]
s = Solution()
print(s.generateParenthesis(3))
