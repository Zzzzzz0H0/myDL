from typing import List
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        """
        贪心算法：算出每个位置能到达的最远位置jump_max
        到倒数第二个位置时，该值如果大于等于长度-1（最后
        一个元素的下标），则返回True，否则返回False
        """
        length = len(nums)
        jump_max = nums[0]
        for i in range(length-1):
            if i <= jump_max:
                jump_max = max(jump_max, i + nums[i])
        return jump_max >= length-1



s = Solution()
#nums = [2,3,1,1,4]
nums = [3,2,1,0,4]
print(s.canJump(nums))

