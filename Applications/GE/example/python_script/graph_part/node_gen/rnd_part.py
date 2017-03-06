import sys
import random

if __name__ == "__main__":
    num_node = int(sys.argv[1])
    num_part = int(sys.argv[2])

    with open('./node_part.txt', 'w') as out_file:
        for i in xrange(num_node):
            host = random.randint(0, num_part - 1)
            out_file.write('%d %d\n' % (i, host))

