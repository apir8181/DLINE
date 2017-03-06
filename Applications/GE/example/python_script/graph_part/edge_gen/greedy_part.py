
import sys
import random

def read_meta(edges_group_path):
    num_nodes = 0
    num_edges = 0
    with open(edges_group_path) as in_file:
        for line in in_file:
            args = line.strip().split()
            num_nodes += 1
            num_edges += int(args[1])
    return num_nodes, num_edges

def graph_partition(num_part, C, edges_group_path, node_part_path, edge_part_path):
    def greedy_score(host_set, node_set, C):
        num_inter = len(host_set & node_set)
        num_host = len(host_set)
        return num_inter * (1 - num_host / C)

    def random_score(host_set, node_set, C):
        return random.random()

    def balance_score(host_set, node_set, C):
        num_host = len(host_set)
        return 1 - num_host / C

    host2files = [open(edge_part_path + '_' + str(i), 'w') for i in range(num_part)]
    host2nodes = [set() for i in range(num_part)]
    with open(edges_group_path) as in_file:
        for line in in_file:
            args = line.strip().split()
            node, num_edge = int(args[0]), int(args[1])
            S = {int(args[2*(i+1)]) for i in range(num_edge)}
            # find host with max score
            max_idx, max_score = random.randint(0, num_part - 1), float('-inf')
            for host in range(num_part):
                s = balance_score(host2nodes[host], S, C)
                if max_score < s:
                    max_idx = host
                    max_score = s
            host2nodes[max_idx].add(node)
            host2files[max_idx].write('%s' % line)

    with open(node_part_path, 'w') as node_file:
        for host, nodes in enumerate(host2nodes):
            for node in nodes:
                node_file.write("%d %d\n" % (node, host))

    for i in range(num_part): host2files[i].close()


def num_cuts(num_nodes, node_part_path, edges_group_path):
    node2host = [-1 for i in range(num_nodes)]

    with open(node_part_path) as in_file:
        for line in in_file:
            args = line.strip().split()
            node, host = int(args[0]), int(args[1])
            node2host[node] = host

    cuts = 0
    with open(edges_group_path) as in_file:
        for line in in_file:
            args = line.strip().split()
            node, num_edge = int(args[0]), int(args[1])
            host = node2host[node]
            dsts = [int(args[2*(i+1)]) for i in range(num_edge)]
            cuts += sum( [node2host[dst] != host for dst in dsts] )

    return cuts


if __name__ == "__main__":
    edges_group_path = sys.argv[1]
    num_part = int(sys.argv[2])
    node_part_path = sys.argv[3]
    edge_part_path = sys.argv[4]

    num_nodes, num_edges = read_meta(edges_group_path)
    print "num nodes: %d, num_edges: %d" % (num_nodes, num_edges)

    C = num_nodes * 1.1 / num_part
    graph_partition(num_part, C, edges_group_path, node_part_path, edge_part_path)

    cuts = num_cuts(num_nodes, node_part_path, edges_group_path)
    print "Cut ratio: %f" % (1.0 * cuts / num_edges)


