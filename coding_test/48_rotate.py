from typing import List
class Solution:
    # 倒序->解包->打包->map->切片赋值
    #matrix[:] = map(list,zip(*matrix[::-1]))
    def rotate1(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        先将矩阵转置，再将每一行翻转即可
        """
        # 转置
        n = len(matrix)
        for i in range(n):
            for j in range(i, n):
                matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]
        print(matrix)
        #翻转每一行 
        for i in range(n):
            matrix[i].reverse()
        return matrix

    def rotate(self, matrix):
        """
        针对每个元素的顺时针旋转90度进行处理
        对方阵来说，每个元素matrix[i][j], 方阵长度为n*n
        轴对称:（对（0,0）(1,1)..(n,n)这条线来说）元素的坐标为matrix[j][i]
        顺时针旋转90度：元素的坐标为matrix[j][n-i-1] (如（0，0）->(0,3) -> (3,3) -> (3,0) )
        逆时针旋转90度：元素的坐标为matrix[n-j-1][i] (如（0,0)-> (3,0)->(3,3)->(0,3))
        """
        n = len(matrix)
        for i in range(n//2 + n%2):
            # 只需行列各旋转一半元素即可
            for j in range(n//2):
                #辅助数组只需4个元素即可
                temp = [0] * 4
                row, col = i, j
                # 记录初始位置的4个值
                for k in range(4):
                    temp[k] = matrix[row][col]
                    row, col = col, n - row - 1
                # 将4个值都旋转90度
                for k in range(4):
                    # %4是将temp的开始位置平移
                    matrix[row][col] = temp[(k-1)%4]
                    row, col = col, n - row - 1
        print(matrix)
s = Solution()
matrix = [
  [1,2,3],
  [4,5,6],
  [7,8,9]
]
"""
先转置为：
[1,4,7],
[2,5,8],
[3,6,9]
再翻转每一行：
[7,4,1],
[8,5,2],
[9,6,3]
"""
print(s.rotate(matrix))
