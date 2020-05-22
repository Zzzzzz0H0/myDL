from typing import List
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        length = len(candidates)
        if length == 0:
            return []
        combins = []
        # 排序后可以剪枝，降低复杂度
        candidates.sort()
        def dfs(new_target, left, right, combin):
            if new_target == 0:
                combins.append(combin.copy())
            for i in range(left, right):
                n = candidates[i]
                k = new_target - n
                if k < 0:
                    break
                combin.append(n)
                # 下一次回溯时可以使用当前值，所有从
                # i开始，如果不能重复使用，则从i + 1
                dfs(k, i, right, combin)
                combin.pop()
        dfs(target, 0, length, [])
        return combins

s = Solution()
candidates = [2,3,6,7]
target = 8
s.combinationSum(candidates, target)
