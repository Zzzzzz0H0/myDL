from typing import List
class ListNode:
   def __init__(self, x):
       self.val = x
       self.next = None

def gen_List(l:List) -> List[ListNode]:
    length = len(l)
    l_list = []
    head = ListNode(l[0])
    temp = head
    for i in l[1:]:
        node = ListNode(i)
        # 链接链表
        temp.next = node 
        temp = temp.next
    return head

# 打印链表
def printlist(head):
    while(head):
        print(head.val)
        head = head.next

class Solution:
    def mergeTwoLists1(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 is None:
            return l2 
        if l2 is None:
            return l1
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 and l2:
            if l1.val > l2.val:
                l1, l2 = l2, l1
            l1.next = self.mergeTwoLists(l1.next, l2)
        # 防止l1为None，此时需返回l2
        return l1 or l2



s = Solution()
l1 = gen_List([1,2,4])
l2 = gen_List([1,2,3])
merge = s.mergeTwoLists(l1,l2)
printlist(merge)
