from adgnn.context import context as context
import torch


def assignParam(model):
    if context.glContext.config['id'] == 0:
        serverNum = context.glContext.config['server_num']
        # laynum=len(model.gc)
        # for i in range(1, laynum + 1):
        #     context.glContext.parameters['w' + str(i)] = model.gc[i].weight.tensor.data.flatten().detach().tolist()
        #     context.glContext.parameters['b' + str(i)] = model.gc[i].bias.tensor.data.flatten().detach().tolist()


        parameters = context.glContext.parameters
        parametersForServer = context.glContext.parametersForServer

        for i in range(serverNum):
            parametersForServer[str(i)] = {}
        for key in parameters.keys():
            num_p = int(len(parameters[key]) / serverNum)
            for i in range(serverNum):
                if i != (serverNum - 1):
                    parametersForServer[str(i)][key] = parameters[key][i * num_p:(i + 1) * num_p]
                else:
                    parametersForServer[str(i)][key] = parameters[key][i * num_p:]

        for i in range(context.glContext.config['server_num']):
            context.glContext.dgnnServerRouter[i].initParameter(
                context.glContext.config['worker_num'],
                context.glContext.config['server_num'],
                context.glContext.config['feature_dim'],
                context.glContext.config['hidden'],
                context.glContext.config['class_num'],
                context.glContext.config['id'],
                context.glContext.parametersForServer[str(i)]
            )
        print('assign parameter end!')



def updateParam(model):
    # laynum = context.glContext.config['layer_num']
    # for i in range(1, laynum + 1):
    #     context.glContext.gradients['w' + str(i)] = context.glContext.gradients['w' + str(i)].flatten().numpy().tolist()
    #     context.glContext.gradients['b' + str(i)] = context.glContext.gradients['b' + str(i)].flatten().tolist()

    for id in context.glContext.gradients:
        context.glContext.gradients[id]=context.glContext.gradients[id].flatten().cpu().detach().numpy().tolist()
        # print('***********gradient'+id+'*****************')
        # print(context.glContext.gradients[id])

    server_num = context.glContext.config['server_num']
    for key in context.glContext.gradients:
        num_g = int(len(context.glContext.gradients[key]) / server_num)
        # agg_grad[key] = []
        for sid in range(server_num):
            if sid == server_num - 1:
                context.glContext.dgnnServerRouter[sid].server_updateParam(context.glContext.config['id'], sid,
                                                                           context.glContext.config['lr'],
                                                                           key,
                                                                           context.glContext.gradients[key][
                                                                           sid * num_g:])

            else:
                context.glContext.dgnnServerRouter[sid].server_updateParam(context.glContext.config['id'], sid,
                                                                           context.glContext.config['lr'],
                                                                           key,
                                                                           context.glContext.gradients[key][
                                                                           sid * num_g:(sid + 1) * num_g])
    params = {}
    for key in context.glContext.gradients:
        params[key] = []
        for sid in range(server_num):
            params[key].extend(context.glContext.dgnnServerRouter[sid].server_PullParams(key))

    layer_num = context.glContext.config['layer_num']
    feat_size = context.glContext.config['feature_dim']
    class_num = context.glContext.config['class_num']
    hidden = context.glContext.config['hidden']

    with torch.no_grad():
        for id in context.glContext.parameters.keys():
            model.parameters_collection[id].data=torch.FloatTensor(params[id]).reshape(model.parameters_collection[id].shape[0], model.parameters_collection[id].shape[1]).to(context.glContext.config['device'])


    context.glContext.dgnnServerRouter[0].server_Barrier()
    # with torch.no_grad():
    #     for i in model.parameters_collection.keys():
    #         model.parameters_collection[i].grad.zero_()

    with torch.no_grad():
        parameters=list(model.parameters())
        for param in parameters:
            param.grad.zero_()
        for id in context.glContext.parameters.keys():
            if model.parameters_collection[id].grad is not None:
                model.parameters_collection[id].grad.zero_()

        # model.gc[1].weight.grad.zero_()
        # model.gc[1].bias.grad.zero_()
        # model.gc[2].weight.grad.zero_()
        # model.gc[2].bias.grad.zero_()

