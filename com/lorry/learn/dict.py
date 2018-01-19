#!/usr/bin/python
# -*- coding: UTF-8 -*-

dict = {'我': 1}
if ('我' in dict):
    dict['我'] = 2
if (dict.has_key("我")):
    dict['我'] = 3


for key in dict.keys():
    print key, dict[key]



