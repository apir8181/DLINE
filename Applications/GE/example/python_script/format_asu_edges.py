
import math
import os
import sys
from subprocess import call
from collections import defaultdict

def save_edges(in_edge_path, out_edge_path):
    with open(in_edge_path) as in_file, \
         open(out_edge_path, "w") as out_file:
        for line in in_file:
            args = line.strip().split(",")
            src, dst = int(args[0]) - 1, int(args[1]) - 1
            out_file.write("%d %d 1\n" % (src, dst))
            out_file.write("%d %d 1\n" % (dst, src))

def save_degree(in_edge_path, out_degree_path):
    degree = defaultdict(float)
    with open(in_edge_path) as in_file, \
         open(out_degree_path, "w") as out_file:
        for line in in_file:
            args = line.strip().split(",")
            src, dst = int(args[0]) - 1, int(args[1]) - 1
            degree[src] += 1
            degree[dst] += 1
        for k, v in degree.iteritems():
            value = math.pow(v, .75)
            out_file.write("%d %f\n" % (k, value))

def save_group(in_group_path, out_group_path):
    m = {}
    with open(in_group_path) as in_file, \
         open(out_group_path, "w") as out_file:
        for line in in_file:
            args = line.strip().split(",")
            node, label = int(args[0]) - 1, int(args[1]) - 1
            if node not in m:
                m[node] = []
            m[node].append(label)

        for node, labels in m.iteritems():
            label_msg = ' '.join(map(lambda x: str(x), labels))
            out_file.write("%d %s\n" % (node, label_msg))

if __name__ == "__main__":
    data_dir = sys.argv[1]
    save_dir = sys.argv[2]
    in_edge_path = os.path.join(data_dir, "edges.csv")
    in_group_path = os.path.join(data_dir, "group-edges.csv")
    out_edge_path = os.path.join(save_dir, "edges.txt")
    out_degree_path = os.path.join(save_dir, "degree.txt")
    out_group_path = os.path.join(save_dir, "group.txt")

    save_edges(in_edge_path, out_edge_path)
    save_degree(in_edge_path, out_degree_path)
    save_group(in_group_path, out_group_path)
