
import sys

if __name__ == "__main__":
    degree_path = sys.argv[1]
    out_path = sys.argv[2]
    proption = float(sys.argv[3])

    with open(degree_path) as in_file, \
         open(out_path, "w") as out_file:
        X = []
        for line in in_file:
            args = line.strip().split()
            node = int(args[0])
            degree = float(args[1])
            X.append((node, degree))

        X = sorted(X, key=lambda x: -x[1])
        save_size = int(len(X) * proption)

        for i in range(save_size):
            node, weight = X[i]
            out_file.write("%d %f\n" % (node, weight))
