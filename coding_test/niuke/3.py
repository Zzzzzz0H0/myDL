#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re 

s = '1+2+31-3'
nn = '0123456789'
result = 0
i = 0
s = '+'+s
while(i < len(s)):
    if i < len(s) and s[i] in  "+-" :
        f = s[i]
        t = ''
        i += 1
        while(i < len(s) and s[i] in nn):
            t += s[i]
            i += 1
        result += int(f + t)
print(result)

