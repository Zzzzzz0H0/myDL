import copy
alist = [1,2,[3,4]]
#  blist = alist[:]
#  blist = alist.copy()
blist = copy.copy(alist)
# list 对象只有浅拷贝，没有deepcopy方法
# 列表中保存的是元素的地址，浅拷贝只拷贝这些
# 地址，而不会重新开辟内存，对不可变对象，如
# 数字，使用浅拷贝就行，因为元素地址不会再变，
# 对于可变元素如二维数组，拷贝时就需要考虑是否
# 需要深拷贝来开辟地址空间。
blist = alist.deepcopy()
# copy模块的deepcopy方法才是深拷贝
#  blist = copy.deepcopy(alist)
# 切片对一维数组的拷贝是有效的，但是对二维的数组拷贝是无效的,和copy效果一致
alist[0] = 6
alist[2].append(5)
print(blist)
