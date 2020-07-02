class Solution:
    def uniquePaths1(self, m: int, n: int) -> int:
        """
        动态规划：
        （1) 定义目标： 到Finish（m, n）的路径f(m,n) 总共有total条
        （2）迁移方程:  f(m ,n) = f(m-1, n) + f(m , n-1)
         (3) 初始状态： f(1, n) = 1, f(m, 1) = 1 (只能一直向一个方向走)
        """
        dp = [[0 for _ in range(n)] for _ in range(m)] 
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for i  in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]

    def uniquePaths(self, m: int, n: int) -> int:
        """
        上面的解法 使用了二维数组，实际上下一行的值只依赖上一行的值，
        所以只需要一维数组即可
        """
        dp = [1 for _ in range(n)]
        for i in range(1, m):
            for j in range(1, n):
                dp[0] = 1 # 相当于dp[0, n] = 1
                dp[j] = dp[j-1] + dp[j] # 迁移方程
        return dp[n-1]

        print(dp)
s = Solution()
m,n =7, 2
print(s.uniquePaths(m, n))
