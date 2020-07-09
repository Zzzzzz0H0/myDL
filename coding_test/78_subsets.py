from typing import List
from copy import deepcopy
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        动态优化：
        定义目标：n个元素的数组的所有子集为s(n)
        迁移方程：s(n) = s(n-1) + s(n-1)的每个元素append 第n个元素 
        初始状态：s[0] = [[]]
        """
        result = [[]]
        for i in range(len(nums)):
            # 注意这里需要使用深拷贝
            dp = deepcopy(result)
            for j in dp:
                j.append(nums[i])
            result.extend(dp)
        return result

s = Solution()
nums = [1, 2, 3]
print(s.subsets(nums))

