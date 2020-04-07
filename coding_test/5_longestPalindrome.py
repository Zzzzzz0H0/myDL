class Solution:
    def longestPalindrome(self, s: str) -> str:
        length = len(s)
        max_len = 1
        start = 0
        dp = [[False for i in range(length)] for i in range(length)]
        for j in range(1, length):
            dp[j][j] = True
            for i in range(length):
                if s[i] == s[j]:
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i+1][j-1]
                else:
                    dp[i][j] = False
                if dp[i][j]:
                    cur_len = j - i + 1
                    if cur_len > max_len:
                        start = i 
                        max_len = cur_len
        return s[start:start+ max_len]

