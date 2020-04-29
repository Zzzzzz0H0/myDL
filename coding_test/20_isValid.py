class Solution:
    def isValid(self, s: str) -> bool:
        d = {'?':'?', '[':']', '{':'}', '(':')'}
        stack = ['?']
        for si in s:
            if si in d.keys():
                stack.append(si)
            else:
                if si != d[stack.pop()]:
                    return  False
                
        return len(stack) == 1


s = Solution()
s1 = "{[]}"
s2 = "([)]"
print(s.isValid(s1))
print(s.isValid(s2))
