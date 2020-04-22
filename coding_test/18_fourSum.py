from typing import List 
class Solution:
    # 三指针法
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        result = []
        length = len(nums)
        if length < 4:
            return result
        nums.sort()
        # 第一个指针
        for i in range(length-3):
            # 最小值都大于target，退出
            if nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target: break
            # 去掉重复值,但需要第一个值之后如[0,0,0,0], 不能一开始就跳
            if i>0 and nums[i] == nums[i-1]:continue
            # 最大值小于target，结束本轮循环
            if nums[i] + nums[length-1] + nums[length-2] + nums[length-3] < target: continue
            # 第二个指针
            for j in range(i+1, length-2):
                if nums[i] + nums[j] + nums[j+1] + nums[j+2] > target: break
                if nums[i] + nums[j] + nums[length-2] + nums[length-1] < target: continue
                if j>i+1 and nums[j] == nums[j-1]: continue
                k = j + 1
                right = length -1
                while(k < right):
                    if nums[i] + nums[j] + nums[k] + nums[right] == target:
                        result.append([nums[i], nums[j], nums[k], nums[right]])
                        # 去掉重复值,需要k < right 先满足
                        while( k< right and nums[k+1] == nums[k]): k += 1
                        while( k< right and nums[right-1] == nums[right]): right -= 1
                        k +=1 
                        right -= 1
                    elif nums[i] + nums[j] + nums[k] + nums[right] > target:
                        right -= 1
                    else: k += 1
        return result





        
s = Solution()
nums = [1, 0, -1, 0, -2, 2]
target = 0
print(s.fourSum(nums, target))
