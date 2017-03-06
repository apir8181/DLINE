
import sys
import random

edge_path = sys.argv[1]
num_part = int(sys.argv[2])

num_edges = 0
with open(edge_path) as in_file:
    for line in in_file: num_edges += 1

with open(edge_path) as in_file, \
     open('./edge_part.txt', 'w') as out_file:
    for i in range(num_part):
        size = num_edges / num_part
        if i == num_part - 1: size = num_edges - i * size
        for j in xrange(size):
            args = in_file.readline().strip().split()
            src, dst = int(args[0]), int(args[1])
            out_file.write('%d %d %d\n' % (src, dst, i))

