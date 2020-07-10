from typing import List
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        """
        从第一个字符开始深度优先搜索上下左右四个方向，
        找到满足的字符串，同时不能走回头路
        """
        row = len(board)
        col = len(board[0])
        def dfs(i, j ,k, visited):
            """dfs

            :param i: 行下标
            :param j: 列下标
            :param k: word 下标
            :param visited: 已经匹配的字符位置
            """
            #只有找到最后一个字符才返回true
            print('ij',i,j)
            print('visited', visited)
            if k == len(word):
                return True
            # 四个寻找方向上下左右
            direction = [[-1,0],[1,0],[0,-1], [0,1]]
            # 朝每个方向走一步
            for direct in direction:
                tempi = i + direct[0]
                tempj = j + direct[1]
                print('tempij', tempi, tempj)
                #  visited.append((tempi,tempj))
                # 继续往下找的条件是不能越界并且能找到第k个字符
                if  (0<= tempi < row) and (0<= tempj < col) and (word[k] == board[tempi][tempj]) and ((tempi,tempj) not in visited):
                    visited.append((tempi, tempj))
                    if  dfs(tempi, tempj, k + 1, visited):
                        return True
                    else:
                        visited.pop()
            # 默认返回False
            return False
        for i in range(row):
            for j in range(col):
                # 每个位置都可能成为起点,并且能搜索到word则返回true
                if board[i][j] == word[0] and dfs(i,j,1,[(i,j)]):
                    return True
        return False


s = Solution()
board = [
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
word = 'ABFD'
#  board = [["A","B","C","E"],
         #  ["S","F","C","S"],
         #  ["A","D","E","E"]]
#  word = 'SEE'
print(s.exist(board, word))
