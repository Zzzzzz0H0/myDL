"""
python 的深浅拷贝
"""
#1. python 默认的赋值是浅拷贝，如对列表的赋值
In [1]: s = [1,2,3]
In [4]: s2 = s
In [7]: s2[0]= 2
In [8]: s
Out[8]: [2, 2, 3]
# 修改s2的元素会影响s的元素值

#2. [:]运算对不可变元素是深拷贝
In [10]: s
Out[10]: [2, 2, 3]
In [9]: s2 = s[:]
In [11]: s2[0]= 1
In [12]: s
Out[12]: [2, 2, 3]
# 此时对s2的修改不影响s的值

#3. [:]运算对可变元素是浅拷贝
In [13]: s = [1,2,[3,4,5]]
In [14]: s2= s[:]
In [15]: s2[2][0] = 6
In [16]: s2
Out[16]: [1, 2, [6, 4, 5]]
In [18]: s
Out[18]: [1, 2, [6, 4, 5]]
# 对s2中的可变元素如列表修改时，同样会影响s的值

#4. 深拷贝使用deepcopy函数
In [18]: s
Out[18]: [1, 2, [6, 4, 5]]
In [19]: import copy
In [20]: s3 = copy.deepcopy(s)
In [21]: s
Out[21]: [1, 2, [6, 4, 5]]
In [22]: s3
Out[22]: [1, 2, [6, 4, 5]]
In [23]: s3[2][0] = 3
In [24]: s
Out[24]: [1, 2, [6, 4, 5]]
# 此时修改s3的值已经不会影响s了