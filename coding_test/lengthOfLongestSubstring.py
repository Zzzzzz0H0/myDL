from collections import deque
class Solution1:
    # 解法1,滑动窗口， 速度一般，占用空间较多
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 初始值为0，保证空字符串也可以兼容
        len_max_sub = 0
        max_sub = '' 
        for si in s:
            if si in max_sub:
                index = max_sub.index(si)
                max_sub = max_sub[index+1:]
            max_sub += si
            # 每滑动一次，更新一次最新值
            len_max_sub = max(len_max_sub, len(max_sub))
        return len_max_sub

class Solution2:
    # 解决内存占用太多的问题
    def lengthOfLongestSubstring(self, s:str) -> int:
        len_max_sub = 0
        ignore_index = -1
        mapping = {}
        for index, si in enumerate(s):
            if si in mapping.keys() and mapping[si] > ignore_index:
                ignore_index = mapping[si] 
            mapping[si] = index
            len_max_sub = max(index - ignore_index, len_max_sub)
        return len_max_sub

s = Solution2()
test_str1 = 'abcabcbb'
test_str2 = 'bbbbb'
test_str3 = 'pwwkew'
test_str4 = 'au'
test_str5 = 'dvdf'
test_str6 = 'nfpdmpi'
test_str7 = ''
assert 3 == s.lengthOfLongestSubstring(test_str1)
assert 1 == s.lengthOfLongestSubstring(test_str2)
assert 3 == s.lengthOfLongestSubstring(test_str3)
assert 3 == s.lengthOfLongestSubstring(test_str5)
assert 5 == s.lengthOfLongestSubstring(test_str6)
assert 0 == s.lengthOfLongestSubstring(test_str7)

