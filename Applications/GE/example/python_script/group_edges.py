
import sys


def read_num_nodes(edge_path):
    max_id = -1
    with open(edge_path) as in_file:
        for line in in_file:
            args = line.strip().split()
            src, dst = int(args[0]), int(args[1])
            max_id = max(max_id, src, dst)
    return max_id + 1

def group_into_nodes(group_edges, edge_path):
    with open(edge_path) as in_file:
        for line in in_file:
            args = line.strip().split()
            src, dst, w = int(args[0]), int(args[1]), float(args[2])
            group_edges[src].append((dst, w))

def save_group_edges(group_edges, out_path):
    with open(out_path, 'w') as out_file:
        for node, edges in enumerate(group_edges):
            num_edges = len(edges)
            out_file.write('%d %d' % (node, num_edges))
            for x in edges:
                out_file.write(' %d %f' % (x[0], x[1]))
            out_file.write('\n')

if __name__ == "__main__":
    edge_path = sys.argv[1]
    out_path = sys.argv[2]
    num_nodes = read_num_nodes(edge_path)
    group_edges = [[] for i in range(num_nodes)]
    group_into_nodes(group_edges, edge_path)
    save_group_edges(group_edges, out_path)
