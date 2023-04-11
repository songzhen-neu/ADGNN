from adgnn.context import context as context
import torch


def assignParam(model):
    if context.glContext.config['id'] == 0:
        serverNum = context.glContext.config['server_num']
        laynum=len(model.gc)
        for i in range(1, laynum + 1):
            context.glContext.parameters['w' + str(i)] = model.gc[i].weight.tensor.data.flatten().detach().tolist()
            context.glContext.parameters['b' + str(i)] = model.gc[i].bias.tensor.data.flatten().detach().tolist()


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
    laynum = context.glContext.config['layer_num']
    for i in range(1, laynum + 1):
        context.glContext.gradients['w' + str(i)] = context.glContext.gradients['w' + str(i)].flatten().numpy().tolist()
        context.glContext.gradients['b' + str(i)] = context.glContext.gradients['b' + str(i)].flatten().tolist()

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


    for i in range(1, layer_num + 1):
        if i == 1:
            model.gc[i].weight.tensor.data = torch.FloatTensor(params['w' + str(i)]).reshape(feat_size, hidden[0])
            model.gc[i].bias.tensor.data = torch.FloatTensor(params['b' + str(i)]).reshape(hidden[0])

        elif i == layer_num:
            model.gc[i].weight.tensor.data = torch.FloatTensor(params['w' + str(i)]).reshape(hidden[-1], class_num)
            model.gc[i].bias.tensor.data= torch.FloatTensor(params['b' + str(i)]).reshape(class_num)

        else:
            model.gc[i].weight.tensor.data = torch.FloatTensor(params['w' + str(i)]).reshape(hidden[i - 2], hidden[i - 1])
            model.gc[i].bias.tensor.data = torch.FloatTensor(params['b' + str(i)]).reshape(hidden[i - 1])


    context.glContext.dgnnServerRouter[0].server_Barrier()

