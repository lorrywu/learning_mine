#!/usr/bin/python
# -*- coding: UTF-8 -*-

import  time

ticks = time.localtime(time.time())
print ticks
ticks = time.strftime(format(["%Y-%m-%d %H:%M:%S", time.localtime()]))
print "now is : ", ticks

