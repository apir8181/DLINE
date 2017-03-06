
import sys

def read_node_part(node_num, node_part_path):
    node2host = [0 for i in range(node_num)]
    with open(node_part_path) as in_file:
        for line in in_file:
            args = line.strip().split()
            src, host = int(args[0]), int(args[1])
            node2host[src] = host
    return node2host


def calculate_borrow_nodes(node2host, edge_part_path):
    ans = 0
    with open(edge_part_path) as in_file:
        for line in in_file:
            args = line.strip().split()
            src, dst, host = int(args[0]), int(args[1]), int(args[2])
            if node2host[src] != host: ans += 1
            if node2host[dst] != host: ans += 1
    return ans

if __name__ == "__main__":
    node_num = int(sys.argv[1])
    node_part_path = sys.argv[2]
    edge_part_path = sys.argv[3]

    node2host = read_node_part(node_num, node_part_path)
    ans = calculate_borrow_nodes(node2host, edge_part_path)

    print "node number: %d, node part: %s, edge part: %s, borrow_nodes: %s" % (
            node_num, node_part_path, edge_part_path, ans)
