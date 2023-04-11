# please refer to core/data_preprocess
from cmake.build.lib.pb11_ec import startPartition
from adgnn.context import context

if __name__=='__main__':
    startPartition(context.glContext.config['worker_num'],context.glContext.config['partitionMethod'],
                   context.glContext.config['data_num'],context.glContext.config['data_path'],
                   context.glContext.config['feature_dim'])
    print("partition end")


