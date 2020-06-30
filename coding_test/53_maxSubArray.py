from typing import List 
class Solution:
    def maxSubArray1(self, nums: List[int]) -> int:
        """
        动态规划法
        1.定义问题解：以第i个元素结尾的数组a最大字序和为f(i),则求f(n)即n个元素的最大子序和
        2.状态转移方程： f(i) = max(f(i-1)+a[i], a[i])
        3.初始状态：f(0) = a[0]
        """
        max_all = nums[0]
        max_pre = nums[0]
        for i in range(1, len(nums)):
            max_pre = max(max_pre+nums[i], nums[i])
            max_all = max(max_all, max_pre)
            print(i, max_all, max_pre)
        return max_all

    def maxSubArray(self, nums: List[int]) -> int:
        """
        递归分治法
        最大子序和要么在左半边，要么在右半边，或者中间
        """
        n = len(nums)
        # 递归终止条件
        if n == 1:
            return nums[0]
        # 中间位置
        mid = n // 2
        # 求左半边最大值
        max_l = self.maxSubArray(nums[:mid])
        # 求右半边最大值,不能将mid + 1，否则列表为空，无限循环
        max_r = self.maxSubArray(nums[mid:n])
        # 求中间最大值,等于左半边最大和加右边最大和
        mid_l, mid_lm = 0, nums[mid-1]  # 初始最大值应该为数组中的值，但是要防止越界
        mid_r, mid_rm = 0, nums[mid]    # mid 最小为1，所以可以-1 
        #  mid_l, mid_lm = nums[mid]    # 右边+1 可能越界，所以不能+1
        #  mid_r, mid_rm = nums[mid+1]
        for i in range(mid-1, -1, -1):
            mid_l = mid_l + nums[i]
            mid_lm = max(mid_l, mid_lm)
        for i in range(mid, n):
            mid_r = mid_r + nums[i]
            mid_rm = max(mid_r, mid_rm)
        return max(max_l, max_r, mid_lm + mid_rm)
        
s = Solution()
nums = [-2,1,-3,4,-1,2,1,-5,4]
#nums = [3,-2,-3,-3,1,3,0]
#nums = [-2, -1]
print(s.maxSubArray(nums))
