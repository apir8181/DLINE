
import sys

def count_num_nodes(edge_path):
    max_id = 0
    with open(edge_path) as in_file:
        for line in in_file:
            args = line.strip().split()
            src, dst = int(args[0]), int(args[1])
            max_id = max(max_id, src, dst)
    return max_id + 1

def count_num_edges(edge_path):
    count = 0
    with open(edge_path) as in_file:
        for line in in_file:
            count += 1
    return count


edge_path = sys.argv[1]
num_part = int(sys.argv[2])

num_nodes = count_num_nodes(edge_path)
num_edges = count_num_edges(edge_path)
nodes_delta = num_nodes / num_part * .1
nodes_max = nodes_delta
node2host = [-1 for i in xrange(num_nodes)]
host2nodeNum = [0 for i in xrange(num_part)]
host2edgeNum = [0 for i in xrange(num_part)]

with open(edge_path) as in_file, \
     open('./edge_part.txt', 'w') as edge_file, \
     open('./node_part.txt', 'w') as node_file:
    for line in in_file:
        args = line.strip().split()
        src, dst = int(args[0]), int(args[1])
        if node2host[src] == -1 and node2host[dst] == -1:
            # rule 1. if src & dst not map, find the host with
            #   least nodes. find the host with least edges if duel.
            min_host, min_num = -1, sys.maxint
            for host, num in enumerate(host2nodeNum):
                if host2nodeNum[host] >= nodes_max:
                    continue
                elif min_num < num:
                    continue
                elif min_num > num:
                    min_host = host
                    min_num = num
                elif host2edgeNum[min_host] > host2edgeNum[host]:
                    min_host = host
            node2host[src] = node2host[dst] = min_host
            host2nodeNum[min_host] += 2
            host2edgeNum[min_host] += 1
            edge_file.write('%d %d %d\n' % (src, dst, min_host))
        elif (node2host[src] != -1 and node2host[dst] == -1) or \
                (node2host[src] == -1 and node2host[dst] != -1):
            # rule 2. if src or dst already mapped, assign nodes to that mapped host.
            #  if that host has no capacity, find the host with least nodes.
            target = src if node2host[src] == -1 else dst
            host = node2host[dst] if node2host[src] == -1 else node2host[src]
            if host != -1 and host2nodeNum[host] < nodes_max:
                node2host[target] = host
                host2nodeNum[host] += 1
                host2edgeNum[host] += 1
                edge_file.write('%d %d %d\n' % (src, dst, host))
            else:
                min_host, min_num = -1, sys.maxint
                for host, num in enumerate(host2nodeNum):
                    if host2nodeNum[host] >= nodes_max:
                        continue
                    elif min_num < num:
                        continue
                    elif min_num > num:
                        min_host = host
                        min_num = num
                    elif host2edgeNum[min_host] > host2edgeNum[host]:
                        min_host = host
                node2host[target] = min_host
                host2nodeNum[min_host] += 1
                host2edgeNum[min_host] += 1
                edge_file.write('%d %d %d\n' % (src, dst, min_host))
        elif node2host[src] != -1 and node2host[dst] != -1:
            # rule 3. if src & dst already mapped.
            if node2host[src] == node2host[dst]:
                host2edgeNum[min_host] += 1
                edge_file.write('%d %d %d\n' % (src, dst, node2host[src]))
            else:
                target_host = node2host[src] if host2edgeNum[node2host[src]] < host2edgeNum[node2host[dst]] else node2host[dst]
                host2edgeNum[target_host] += 1
                edge_file.write('%d %d %d\n' % (src, dst, target_host))
        # check if node capacity is full
        is_full = True
        for host, num in enumerate(host2nodeNum):
            if num < nodes_max: is_full = False
        if is_full:
            nodes_max += nodes_delta

    for node, host in enumerate(node2host):
        node_file.write('%d %d\n' % (node, host))


    print "Host Node Num"
    for host, num in enumerate(host2nodeNum):
        print "\t%d %d" % (host, num)

    print "Host Edge Num"
    for host, num in enumerate(host2edgeNum):
        print "\t%d %d" % (host, num)
