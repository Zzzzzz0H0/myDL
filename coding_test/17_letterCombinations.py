class Solution1:
    # 每次组合2个数字，前面所有组合+本次需要组合的数字
    # '234' 先将2和3的所有组合情况求出，再求出前面组合和4组合的所有情况
    def letterCombinations(self, digits):
        if digits == []:
            return []
        digit_letter_dict = {
            '2':'abc', '3':'def', '4':'ghi',
            '5':'jkl', '6':'mno', '7':'pqrs',
            '8':'tuv', '9':'wxyz'
        }
        combins = [''] 
        for di in digits:
            combins = [ j + i for j in combins  for i in digit_letter_dict[di]]
        return combins

class Solution2:
    # 使用队列实现Solution1
    def letterCombinations(self, digits):
        if digits == '':
            return []
        digit_letter_dict = {
            '2':'abc', '3':'def', '4':'ghi',
            '5':'jkl', '6':'mno', '7':'pqrs',
            '8':'tuv', '9':'wxyz'
        }
        phone = ['abc','def','ghi','jkl','mno','pqrs','tuv','wxyz']
        queue = ['']
        for digit in digits:
            for _ in range(len(queue)):
                temp = queue.pop(0)
                #  for letter in digit_letter_dict[digit]:
                # 使用ascii法比字典更快,但在1中这样会更慢，调用次数多的时候使用hashmap
                for letter in phone[ord(digit) - 50]:
                    queue.append(temp + letter)
        print(queue)
        return queue



class Solution3:
    # 回溯解法，即求树的深度优先遍历
    # '234' 对应的树如下
    #              root
    #          a     b     c
    #        d e f d e f d e f
    # 回溯条件： 是叶子节点时回溯
    def letterCombinations(self, digits):
        if digits == '':
            return []
        results = []
        digit_letter_dict = {
            '2':'abc', '3':'def', '4':'ghi',
            '5':'jkl', '6':'mno', '7':'pqrs',
            '8':'tuv', '9':'wxyz'
        }
        def back_trace(combins, digits):
            if digits == '':
                results.append(combins)
            else:
                for letter in digit_letter_dict[digits[0]]:
                    back_trace(combins+letter, digits[1:])
        back_trace('', digits)
        return results
        
s = Solution2()
digits = '23'
assert ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"] == s.letterCombinations(digits)
