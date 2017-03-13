#!/usr/bin/python

import sys
import csv

reader = csv.reader(sys.stdin, delimiter='\t')

for line in reader:
    id = line[0]
    body = line[4]

    words = body.split()
    for word in words:
        if 'fantastic' in word.lower():
            #if 'fantastically' not in word.lower():
            print "{0}\t{1}".format(id, word)