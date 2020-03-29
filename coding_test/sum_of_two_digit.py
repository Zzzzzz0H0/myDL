#!/usr/bin/env python
# -*- coding: utf-8 -*-
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def addTowNumbers(self, l1:ListNode, l2:ListNode)->ListNode:
        pre_node = ListNode(0)
        last_node = pre_node
        t = 0
        while t or l1 or l2:
            t, val = divmod(t + (l1.val if l1 else 0) + (l2.val if l2 else 0), 10)
            last_node.next = ListNode(val)
            last_node = last_node.next 
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return pre_node.next 

def generate_list(l):
    pre_node = ListNode(0)
    last_node = pre_node 
    for val in l:
        node = ListNode(val)
        last_node.next = node
        last_node = node 
    return pre_node.next

def print_listnode(ld):
    while ld:
        print('%d, ' %ld.val)
        ld = ld.next

l1 = generate_list([2, 4, 3, 3])
l2 = generate_list([5, 6, 4])
s = Solution() 
print_listnode(s.addTowNumbers(l1, l2))

