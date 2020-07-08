from typing import List
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        因为只有三个可能，所以只要有2个指针pl, pr分别指向2边0和2的位置
        遍历一次数组，将0值和pl值交换，将2值和pr交换，剩余的1值自然在中间
        """
        """ 关键点说明:
        题解少了一个状态维持的关键说明：
        curr 左边全都是0/1的有序序列！
        curr位置是0时，与左边的0位置交换，因为保证curr左边全是0/1, 所以交换过来的必然是0/1，状态维持住了；
        curr位置是2时，交换后，curr不能移动，因为一移动，没法保证交换过来的是0/1；所以这里不移动；这时状态也维持住了
        只要我们保证curr左边的都是0,1，才移动，那么0都被放到左边，2都被放到右边，中间自然是1了； 综上，这是一个关键状态说明，有了这个说明，逻辑才更加清楚
        """
        print(nums)
        if nums == []:
            return []
        length = len(nums)
        i , pl,pr =0, 0,length-1
        while i <= pr:
            if nums[i] == 0:
                nums[pl], nums[i] = nums[i],nums[pl]
                pl += 1
                i += 1
                print('li {},left {}, right {}, nums{}'.format(i, pl, pr, nums))
            elif nums[i] == 2:
            # 此时不能i++，换过来的可能为1，需要和pl比较才能跳
                nums[i], nums[pr] = nums[pr], nums[i]
                pr -= 1
                print('ri {},left {}, right {}, nums{}'.format(i, pl, pr, nums))
            else:
                pass
                i += 1
        print(nums)
s = Solution()
nums = [2,0,2,1,1,0]
s.sortColors(nums)
