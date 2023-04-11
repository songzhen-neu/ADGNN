import linecache
from ecgraph.context import context

cluster_num=context.glContext.config['sample_num'][1]
worker_num=context.glContext.config['worker_num']
fileName=context.glContext.config['data_path']
if __name__=='__main__':
    metisFileName = fileName + '/nodesPartition' + '.metis' + str(cluster_num) + '.txt'
    outputFileName = fileName + '/nodesPartition' + '.clustergcn'+str(cluster_num)+'-' + str(worker_num) + '.txt'
    sub_node = set()
    fileOutput=open(outputFileName,'w+')
    sub_node_str=''
    cluster_num_each=int(cluster_num/worker_num)


    for i in range(cluster_num):
        line_choice = linecache.getline(metisFileName, i+1)[:-1]+'\t'
        sub_node_str+=line_choice

        if i%cluster_num_each==cluster_num_each-1:
            sub_node_str=sub_node_str[:-1]
            sub_node_str+='\n'
            fileOutput.write(sub_node_str)
            sub_node=set()
            sub_node_str=''

    fileOutput.close()
