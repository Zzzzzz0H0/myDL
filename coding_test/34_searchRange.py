from typing import List
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        length = len(nums)
        def search(left, right, right_on=False):
            while left < right:
                mid = (left + right)//2
                if nums[mid] < target or (right_on and nums[mid] == target):
                    # 跳出时为满足条件的值偏移一位
                    left = mid + 1
                else:
                    right = mid
            return left
        start = search(0, length, right_on=False)
        print(start, length)
        if start == length or  nums[start] != target:
            # 如果没找到，返回-1
            return  -1, -1
        else:
            # 找满足提交的最右边
            end = search(0, length, right_on=True) - 1
        return start, end


s = Solution()
nums = [5,7,7,8,8,10]
target = 11
print(s.searchRange(nums, target))
