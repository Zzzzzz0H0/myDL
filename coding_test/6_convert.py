class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows < 2:
            return s
        des_str = ['' for _ in range(numRows)]
        index = 0
        flag = -1
        for si in s:
            if (index == 0) or (index == numRows -1):
                flag = - flag
            des_str[index] += si
            index = index + flag

        print(des_str)
        return ''.join(des_str)


solution = Solution()
s = "LEETCODEISHIRING"
numRows = 4
print(solution.convert(s, numRows))
assert "LDREOEIIECIHNTSG" == solution.convert(s, numRows)
