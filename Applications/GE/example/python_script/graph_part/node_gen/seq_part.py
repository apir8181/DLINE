
import sys

if __name__ == "__main__":
    num_node = int(sys.argv[1])
    num_part = int(sys.argv[2])

    with open('./node_part.txt', 'w') as out_file:
        for i in xrange(num_part):
            size = num_node / num_part
            offset = size * i
            if i == num_part - 1: size = num_node - offset
            for j in xrange(size):
                out_file.write('%d %d\n' % (offset + j, i))

