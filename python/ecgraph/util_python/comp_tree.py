from python.ecgraph.context import context


def generateAggEmb(compGraph):
    agg_emb={}
    layer_num = context.glContext.config['layer_num']
    layer_count = layer_num
    queue = []
    breadthFirstTraversal(compGraph, layer_count, queue,agg_emb)
    return agg_emb



def breadthFirstTraversal(comp_graph, layer_count, queue,agg_emb):
    if comp_graph is None:
        return
    if comp_graph.left is not None:
        queue.append(comp_graph.left)
    if comp_graph.right is not None:
        queue.append(comp_graph.right)
    if comp_graph.operator == 'spmm':
        agg_emb['nei_embs' + str(layer_count)] = comp_graph.right.tensor.detach().numpy()
        agg_emb['agg_embs' + str(layer_count)] = comp_graph.tensor.detach().numpy()
        layer_count -= 1
    if len(queue) != 0:
        next_root = queue.pop(0)
        breadthFirstTraversal(next_root, layer_count, queue,agg_emb)
    return
