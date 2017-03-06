
import sys
import random

edge_path = sys.argv[1]
num_part = int(sys.argv[2])

with open(edge_path) as in_file, \
     open('./edge_part.txt', 'w') as out_file:
    for line in in_file:
        args = line.strip().split()
        src, dst = int(args[0]), int(args[1])
        host = random.randint(0, num_part - 1)
        out_file.write('%d %d %d\n' % (src, dst, host))
