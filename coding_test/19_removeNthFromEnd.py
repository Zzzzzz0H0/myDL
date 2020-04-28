# Definition for singly-linked list.
from typing import List
class ListNode:
   def __init__(self, x):
       self.val = x
       self.next = None

class Solution:
    # 快慢指针法，快指针先走n步，慢指针开始走，等快指针走到链表末端时，慢指针就是倒数第n个
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        if head.next == None:
            return None
        new_node = ListNode(0)
        new_node.next = head
        faster = slower = new_node
        i = 0
        while(faster.next):
                faster = faster.next
                i += 1
                if(i>n):
                    slower = slower.next
        slower.next = slower.next.next
        return new_node.next

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

l = [1,2,3,4,5]
head = gen_List(l)
s = Solution()
new_head = s.removeNthFromEnd(head, 3)
printlist(new_head)
