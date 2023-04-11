from ogb.nodeproppred import NodePropPredDataset
import numpy as np
# root = "/mnt/data"
dataset = NodePropPredDataset( name="ogbn-papers100M")
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
data = dataset[0][0]

train_num=11105995
val_num=125265
test_num=214338
node_num=11445598
fpFeatClass = '/mnt/data/ogbn-papers100M/featsClass.txt'
fpEdge = '/mnt/data/ogbn-papers100M/edges.txt'

fwFeatClass = open(fpFeatClass, 'w+')
fwEdge = open(fpEdge, 'w+')

edges=np.array(data['edge_index'])

edgeNum=len(edges[0])
for i in range(edgeNum):
    if edges[0][i]<node_num and edges[1][i]<node_num:
        edge_tmp=str(edges[0][i])+'\t'+str(edges[1][i])+'\n'
        fwEdge.write(edge_tmp)

nodeNum=len(data['x'])
feats=np.array(data['x'])
classes=np.array(data['y'])

for i in range(nodeNum):
    if i < node_num:
        featClass_tmp=str(i)+'\t'+'\t'.join(map(str,feats[i]))+'\t'+str(classes[i][0])+'\n'
        fwFeatClass.write(featClass_tmp)

fwFeatClass.close()
fwEdge.close()
print("aaa")