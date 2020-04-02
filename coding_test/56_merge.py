class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        merged = []
        intervals.sort(key=lambda x: x[0])
        for interval in intervals:
            if (not merged) or (interval[0] > merged[-1][1]):
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged

s = Solution()
intervals1 = [[1,3],[2,6],[8,10],[15,18]]
intervals1_ret = [[1,6],[8,10],[15,18]]
assert intervals1_ret == s.merge(intervals1)
