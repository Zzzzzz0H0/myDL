from typing import List
class Solution:
    def minPathSum1(self, grid: List[List[int]]) -> int:
        """
        动态规划：
        1. 定义目标：在每个位置[i,j]上的最短路径为dp[i,j]
        2. 迁移方程：dp[i,j] = min(dp[i-1,j] + grid[i-1, j],dp[i, j-1] + dp[i,j-1])
        3. 初始化：dp[0,0] = grid[0][0]
        """
        row, col = len(grid),len(grid[0])
        dp = [[0 for _ in range(col)] for _ in range(row)] 
        for i in range(row):
            for j in range(col):
                if (i== 0) and (j == 0):
                    dp[i][j] = grid[i][j]
                elif (i == 0) and (j != 0):
                    dp[i][j] = dp[i][j-1] + grid[i][j]
                elif (j ==  0) and (i != 0):
                    dp[i][j] = dp[i-1][j] + grid[i][j]
                else:
                    dp[i][j] = min(dp[i-1][j]+grid[i][j], dp[i][j-1]+grid[i][j])
        return dp[row-1][col-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        row, col = len(grid), len(grid[0])
        dp = [grid[0][0] for _ in range(col)]
        for i in range(1, col):
            dp[i] = dp[i-1] +  grid[0][i]
        for i in range(1, row):
            dp[0] = dp[0] + grid[i][0]
            for j in range(1, col):
                dp[j] = min(dp[j-1], dp[j])
                dp[j] += grid[i][j]
        return dp[-1]

s = Solution()
grid = [
  [1,2,3],
  [4,5,6],
]
print(s.minPathSum(grid))
