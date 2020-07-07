"""
python3 pandas 写unicode 字符串问题
"""
import pandas as pd 
# x 中有个控制字符
x = [u'string with some unicode: \x16', '中文']

# stackoverflow
# https://stackoverflow.com/questions/28837057/pandas-writing-an-excel-file-containing-unicode-illegalcharactererror
print(x)
#['string with some unicode: \x16']
data = pd.DataFrame(x)
# 使用pandas写控制字符到excel
#  data.to_excel('test.xlsx')
#  报错：
    #  raise IllegalCharacterError
#  openpyxl.utils.exceptions.IllegalCharacterError
#  理由：
#  \x16是空格，特殊字符写入excle，opeyxl不支持
#  字符时无法解析
#  解决方案：
#  （1）
#  data = data.applymap(lambda x: x.encode('unicode_escape').decode('utf-8')
                     #  if isinstance(x, str) else x)
#  data.to_excel('/Users/zhuheng/workspace/zhuheng/myDL/Python/learning_note/test.xlsx')
#  效果： 不报错，将字符写 入了excle文件，但是字符是\x形式的
#  问题： 会将本来utf-8编码的字符，如汉字变成unicode形式的
#  (2)
#   更换engine
#data.to_excel('/Users/zhuheng/workspace/zhuheng/myDL/Python/learning_note/test1.xlsx', engine='xlsxwriter')
#   效果： 成功写入文件，\x16也成功转为空格



