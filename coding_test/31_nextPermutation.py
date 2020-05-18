from typing import List
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        length = len(nums)
        for i in reversed(range(0, length)):
            if nums[i] > nums[i-1]:
                break
        t = i - 1 
        print(t, nums[t])
        if t != 0:
            for j in reversed(range(t, length)):
                if nums[j] > nums[t]:
                    break 
            print(j, nums[j])
            nums[t], nums[j] = nums[j], nums[t]
        else:
            i = 0
        nums[i:] = reversed(nums[i:])
        print(nums)
s = Solution()
#  s.nextPermutation([1,2,7,4,3,1])
s.nextPermutation([3,2,1])
