from typing import List
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        """
        求数组中顺序差值最大的2个元素
        1.以第一个元素为最低点，找升区间，计算第一个上升区间的最高价格
        2.如果遇到比最低点更低的值，更新最低点计算第二个上升区间的最高价格，
        3. 比较第一第二个区间，取较大值
        4.重复2，3，遍历完整个数组
        """
        # 初始值为0
        max_p = 0
        length = len(prices)
        #第一个低点位置为0
        low_i = 0
        temp = 0
        for i in range(1, length):
            # 更新低点位置
            if prices[i] < prices[low_i]:
                low_i = i
            # 判断是否是上升区间,更新最大利润
            if prices[i] - prices[low_i] > temp:
                temp = prices[i] - prices[low_i]
                max_p = max(max_p, temp)
        return max_p

s = Solution()
prices = [7,1,5,3,6,4,2, 10]
s.maxProfit(prices)
