# /user/bin/env python
class Solution:
    def maxArea(self, height) -> int:
        left, right, max_area = 0, len(height) -1, 0
        while left < right:
            if height[left] > height[right]:
                max_area = max(max_area,(right - left) * height[right])
                right -= 1
            else:
                max_area = max(max_area, (right - left) * height[left])
                left += 1
        return max_area

solution = Solution()
height = [1,8,6,2,5,4,8,3,7]
assert 49 == solution.maxArea(height)
