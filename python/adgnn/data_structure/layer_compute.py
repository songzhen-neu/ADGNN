class Layer_Compute():
    def __init__(self,train_vertices,adj,id_old2new_dict,id_new2old_dict,fsthop_for_worker,rmt_nei_num, push_2_worker_nodes,v2wk_push):
        self.train_vertices=train_vertices
        self.adj=adj
        self.id_old2new_dict=id_old2new_dict
        self.id_new2old_dict=id_new2old_dict
        self.fsthop_for_worker=fsthop_for_worker
        self.rmt_nei_num=rmt_nei_num
        self.push_2_worker_nodes= push_2_worker_nodes
        self.v2wk_push=v2wk_push
