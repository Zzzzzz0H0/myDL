class Solution:
    def threeSum(self, nums):
        result = []
        nums.sort()
        lenth = len(nums)
        if lenth < 3:
            return []
        for i in range(lenth):
            if nums[i] > 0:
                break
            # skip repeat value
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            left = i + 1
            right = lenth - 1
            while left < right:
                if nums[i] + nums[left] + nums[right] == 0:
                    result.append([nums[i], nums[left], nums[right]])
                    # skip repeat value
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif nums[i] + nums[left] + nums[right] > 0:
                    right -= 1
                else:
                    left += 1
        return result

s = Solution()
nums = [-1, 0, 1, 2, -1, -4]
assert [
    [-1, -1, 2],
    [-1, 0, 1]
] == s.threeSum(nums) 
