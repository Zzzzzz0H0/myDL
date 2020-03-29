#!/usr/bin/env python
# -*- coding: utf-8 -*-
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        tmp = dict()
        for index, n in enumerate(nums):
            if target -n in tmp.keys():
                return tmp[target - n], index
            tmp[n] = index
