from typing import List
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        pers = []
        if nums == []:
            return []
        length = len(nums)
        def dfs(per, left, used):
            if len(per) == length:
                pers.append(per.copy())
            for i in left:
                if i not in used:
                    used.add(i)
                    per.append(i)
                    dfs(per, left, used)
                    per.pop()
                    used.remove(i)
        # 使用userd来保存使用过的值，set
        # 内部用dict实现，查询为O1
        used = set()
        dfs([], nums, used)
        return pers

s = Solution()
nums = [1,2,3]
print(s.permute(nums))
