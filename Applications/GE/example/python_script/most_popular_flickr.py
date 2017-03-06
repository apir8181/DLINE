
import sys
from collections import defaultdict

if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    group_count = defaultdict(int)
    with open(in_path) as in_file:
        for line in in_file:
            args = line.strip().split()
            for arg in args[1:]:
               group_count[int(arg)] += 1

    sort_count = [(k, v) for k, v in group_count.iteritems()]
    sort_count = sorted(sort_count, key=lambda x: -x[1])
    popular_group = {x[0]: i for i, x in enumerate(sort_count[:5])}

    with open(in_path) as in_file, \
         open(out_path, 'w') as out_file:
        for line in in_file:
            args = line.strip().split()
            node = int(args[0])
            groups = [int(x) for x in args[1:]]
            need_node = False
            for group in groups:
                if group in popular_group:
                    need_node = True
            if need_node:
                out_file.write('%d' % node)
                for group in groups:
                    if group in popular_group:
                        out_file.write(' %d' % popular_group[group])
                out_file.write('\n')
