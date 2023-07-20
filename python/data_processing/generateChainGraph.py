import random

if __name__ == '__main__':
    filename = '/mnt/data/test/'
    edgesFileName = filename + 'edges_raw.txt'
    featsClassFileName = filename + 'featsClass_raw.txt'

    edgesFile = open(edgesFileName, 'w+')
    featsClassFile = open(featsClassFileName, 'w+')

    # insert nodes, feat_size=4,class=16
    str_node = [i for i in range(16)]

    str_node[0] = "0\t0\t0\t0\t0\t0\n"
    str_node[1] = "1\t0\t0\t0\t1\t1\n"
    str_node[2] = "2\t0\t0\t1\t0\t2\n"
    str_node[3] = "3\t0\t0\t1\t1\t3\n"
    str_node[4] = "4\t0\t1\t0\t0\t4\n"
    str_node[5] = "5\t0\t1\t0\t1\t5\n"
    str_node[6] = "6\t0\t1\t1\t0\t6\n"
    str_node[7] = "7\t0\t1\t1\t1\t7\n"
    str_node[8] = "8\t1\t0\t0\t0\t8\n"
    str_node[9] = "9\t1\t0\t0\t1\t9\n"
    str_node[10] = "10\t1\t0\t1\t0\t10\n"
    str_node[11] = "11\t1\t0\t1\t1\t11\n"
    str_node[12] = "12\t1\t1\t0\t0\t12\n"
    str_node[13] = "13\t1\t1\t0\t1\t13\n"
    str_node[14] = "14\t1\t1\t1\t0\t14\n"
    str_node[15] = "15\t1\t1\t1\t1\t15\n"

    for i in range(16):
        featsClassFile.write(str_node[i])

    for i in range(15):
        edge = str(i) + "\t" + str(i + 1)+"\n"
        edge_r = str(i + 1) + "\t" + str(i)+"\n"
        edgesFile.write(edge)
        edgesFile.write(edge_r)

    edgesFile.write("0\t8\n")
    edgesFile.write("8\t0\n")

    edgesFile.close()
    featsClassFile.close()
