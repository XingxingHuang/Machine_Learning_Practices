#!/usr/bin/python

import sys

fantasticTotal = 0
fantastically_list = []

for line in sys.stdin:
    data_mapped = line.strip().split("\t")

    if len(data_mapped) != 2:
        continue

    id, word = data_mapped

    if 'fantastic' in word.lower():
        if 'fantastically' not in word.lower():
            fantasticTotal += 1

    if 'fantastically' in word.lower():
        fantastically_list.append(int(id))
        fantastically_list = sorted(fantastically_list)

print fantasticTotal, fantastically_list