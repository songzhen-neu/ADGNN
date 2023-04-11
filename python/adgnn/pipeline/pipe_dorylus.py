from adgnn.pipeline.pipeline import Pipeline
import torch
from adgnn.context.context import *
import numpy as np
from cmake.build.lib.pb11_ec import *
import adgnn.Function as F
from adgnn.util_python.timecounter import time_counter
from adgnn.util_python.remote_access import getLocEmbs


class PipeDorylus(Pipeline):
    def startPipe(self):
        batch_size = int(self.data['input']().shape[0] / self.batch_num)
        time_counter.start('div_batch')
        id_list, input_list = self.divideBatch(batch_size)
        time_counter.end('div_batch')
        embs_tmp = torch.FloatTensor()
        ids_tmp = []
        emb_size = 0
        time_counter.start('main')
        for i in range(self.batch_num):
            time_counter.start('main_eachbatch')
            time_counter.start('comp')
            push_id, push_emb = self.compFunc(node_ids=id_list[i], inputs=input_list[i], weight=self.data['weight'])
            time_counter.end('comp')
            emb_size = np.array(push_emb).shape[1]
            embs_tmp=torch.cat((embs_tmp,push_emb),dim=0)
            ids_tmp.extend(push_id)
            time_counter.start('comm')
            self.commFunc(i, push_id, push_emb)
            time_counter.end('comm')
            time_counter.end('main_eachbatch')
        time_counter.end('main')
        time_counter.start('getloc')
        loc_embs = getLocEmbs(self.layer_id,self.graph,ids_tmp, embs_tmp)
        time_counter.end('getloc')
        time_counter.start('getrmt')
        rmt_embs = glContext.dgnnClientRouterForCpp.getEntireEmbs(self.status, self.layer_id, emb_size)
        time_counter.end('getrmt')
        time_counter.start('build')
        entire_emb = torch.cat((torch.FloatTensor(loc_embs), torch.FloatTensor(rmt_embs)), dim=0)
        output = self.buildCompGraph(mm_output=torch.FloatTensor(embs_tmp), push_output=entire_emb)
        time_counter.end('build')
        return output


    def divideBatch(self, batch_size):
        id_list = []
        input_list = []
        input = self.data['input']()
        new2old_map = self.graph.layer_compute[self.layer_id - 1].id_new2old_dict
        for i in range(self.batch_num):
            if i != self.batch_num - 1:
                id_batch = []
                for j in range(i * batch_size, (i + 1) * batch_size):
                    id_batch.append(new2old_map[j])
                id_list.append(id_batch)
                id_range = range(i * batch_size, (i + 1) * batch_size)
                input_list.append(input[torch.LongTensor(id_range)])

            else:
                id_batch = []
                # input_batch = input[i * batch_size:]
                last_batch_size = len(input) % batch_size + batch_size
                for j in range(i * batch_size, i * batch_size + last_batch_size):
                    id_batch.append(new2old_map[j])
                id_list.append(id_batch)
                id_range = range(i * batch_size, i * batch_size + last_batch_size)
                input_list.append(input[torch.LongTensor(id_range)])

        return id_list, input_list

    def compFunc(self, **kwargs):
        node_ids = kwargs['node_ids']
        inputs = kwargs['inputs']
        weight = kwargs['weight']()
        time_counter.start('mm')
        result = torch.mm(inputs, weight)
        time_counter.end('mm')
        # print("input shape:{0}*{1}".format(inputs.shape[0],inputs.shape[1]))
        # print("weight shape:{0}*{1}".format(weight.shape[0],weight.shape[1]))
        # print('mm:{:.4f}'.format(time_counter.time_list['mm'][-1]))
        time_counter.start('trans2np')
        result = result
        time_counter.end('trans2np')
        return node_ids, result

    def commFunc(self, batch_id, push_id, push_emb):
        if batch_id == 0:
            Router.initReplyEmbs(self.status, self.layer_id, push_emb.shape[1], self.batch_num)
            glContext.dgnnServerRouter[0].server_Barrier()
        glContext.dgnnClientRouterForCpp.pushEmbsByIds(self.layer_id, batch_id, self.status, push_id,
                                                       push_emb)

    def buildCompGraph(self, **kwargs):
        # build mm node
        x2 = F.mm_pipe(kwargs['mm_output'], self.data['input'], self.data['weight'])
        x3 = F.push_pipe(kwargs['push_output'], x2, self.layer_id)
        return x3
