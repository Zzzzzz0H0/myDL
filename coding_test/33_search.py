from typing import List
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if nums == []:
            return -1
        left = 0
        right = len(nums) -1
        while(left < right):
            mid = int((left + right) / 2)
            if nums[mid] == target:
                return mid
            if nums[mid] < nums[right]:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid + 1
            else:
                if nums[left] <= target < nums[mid]:
                    right = mid -1
                else:
                    left = mid + 1
        if nums[left] == target:
            return left
        else:
            return -1


s = Solution()
#  nums = [4,5,6,7,0,1,2]
nums = [1,3]
target = 1
print(s.search(nums, target))
