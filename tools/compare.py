#!/usr/bin/env python

import sys

original = open(sys.argv[1], "r")
compared = open(sys.argv[2], "r")

total = 0
different = 0
for oline in original:
        cline = compared.readline()
        if int(oline) != int(cline):
                different += 1
        total += 1

dpercent = (float(different) * 100.0) / float(total)
print "%.2f" % (100.0 - dpercent)

