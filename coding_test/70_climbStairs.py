class Solution:
    def climbStairs(self, n: int) -> int:
        """
        动态规划：
        1. 设定目标：n级台阶有m种爬法
        2. 迁移方程：m(n)=m(n-1) + m(n-2)
        3. 初始状态：m（1）= 1, m(2) = 2 
        """
        if n == 1:
            return 1
        elif n == 2:
            return 2
        else:
            temp1, temp2 = 1, 2 
            for i in range(2, n):
                temp1, temp2 = temp2, temp1 + temp2
        return temp2

    def climbStairs(self, n: int) -> int:
        """
        递归实现:超出时间限制
        """
        def f(n):
            if n <=2:
                return n
            else:
                return f(n-1) + f(n-2)
        return f(n)

        

s = Solution()
n = 5
print(s.climbStairs(n))
