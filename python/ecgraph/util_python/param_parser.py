import argparse
from ecgraph.context import context


def parserInit():
    parser = argparse.ArgumentParser(description="Pytorch argument parser")
    parser.add_argument('--role_id', type=str, help='machine role and id')
    parser.add_argument('--ifctx', type=str, help='context from file or arguments delivering')
    parser.add_argument('--worker_server_num', type=str, help='the number of worker and server')
    parser.add_argument('--vtx_edge_feat_class_train_val_test', type=str, help='vtx_edge_feat_class_train_val_test')

    parser.add_argument('--hidden', type=str, help='hidden')
    parser.add_argument('--data_path', type=str, help='data_path')

    parser.add_argument('--iter_lr_pttMethod_printInterval', type=str, help='iter_lr_pttMethod')

    parser.add_argument('--servers', type=str, help='server ip')
    parser.add_argument('--workers', type=str, help='worker ip')

    parser.add_argument('--sampleInfo', type=str, help='sample info')
    parser.add_argument('--adConfig', type=str, help='adConfig')

    args = parser.parse_args()
    ifctx = str.split(args.ifctx, ',')

    if ifctx[0] == 'true':
        context.Context.ifctx = True
    else:
        context.Context.ifctx = False

    if context.Context.ifctx:
        print("using context configuration")
        role_id = str.split(args.role_id, ',')
        context.glContext.config['role'] = role_id[0]
        context.glContext.config['id'] = int(role_id[1])

        context.glContext.config['layer_num'] = len(context.glContext.config['hidden']) + 1
        context.glContext.config['emb_dims'].append(context.glContext.config['feature_dim'])
        context.glContext.config['emb_dims'].extend(context.glContext.config['hidden'])
        context.glContext.config['emb_dims'].append(context.glContext.config['class_num'])
    else:
        print("using argument configuration")
        role_id = str.split(args.role_id, ',')
        context.glContext.config['role'] = role_id[0]
        context.glContext.config['id'] = int(role_id[1])

        worker_server_num = str.split(args.worker_server_num, ',')
        context.glContext.config['worker_num'] = int(worker_server_num[0])
        context.glContext.config['server_num'] = int(worker_server_num[1])

        context.glContext.config['data_path'] = args.data_path

        context.glContext.config['hidden'] = [int(i) for i in str.split(args.hidden, ':')]
        context.glContext.config['layer_num'] = len(context.glContext.config['hidden']) + 1


        vtx_edge_feat_class_train_val_test = str.split(args.vtx_edge_feat_class_train_val_test, ',')
        context.glContext.config['data_num'] = int(vtx_edge_feat_class_train_val_test[0])
        context.glContext.config['edge_num'] = int(vtx_edge_feat_class_train_val_test[1])
        context.glContext.config['feature_dim'] = int(vtx_edge_feat_class_train_val_test[2])
        context.glContext.config['class_num'] = int(vtx_edge_feat_class_train_val_test[3])
        context.glContext.config['train_num'] = int(vtx_edge_feat_class_train_val_test[4])
        context.glContext.config['val_num'] = int(vtx_edge_feat_class_train_val_test[5])
        context.glContext.config['test_num'] = int(vtx_edge_feat_class_train_val_test[6])
        context.glContext.config['emb_dims'].append(context.glContext.config['feature_dim'])
        context.glContext.config['emb_dims'].extend(context.glContext.config['hidden'])
        context.glContext.config['emb_dims'].append(context.glContext.config['class_num'])

        iter_lr_pttMethod_printInterval = str.split(args.iter_lr_pttMethod_printInterval, ',')
        context.glContext.config['iterNum'] = int(iter_lr_pttMethod_printInterval[0])
        context.glContext.config['lr'] = float(iter_lr_pttMethod_printInterval[1])
        context.glContext.config['partitionMethod'] = iter_lr_pttMethod_printInterval[2]
        context.glContext.config['print_result_interval'] = int(iter_lr_pttMethod_printInterval[3])

        # sampNum_batchSize_commFo_enablePc_mAd_mAsAd_alphaSelectAs_asFanout_betaRecomp_enabAdapM
        sample_info = str.split(args.sampleInfo, ',')
        sample_num = [int(i) for i in str.split(sample_info[0], ':')]
        context.glContext.config['sample_num'] = sample_num
        context.glContext.config['batch_size'] = int(sample_info[1])
        context.glContext.config['sample_method'] = sample_info[2]

        ad_config = str.split(args.adConfig, ',')
        context.glContext.config['dim_itvs'] = [int(i) for i in str.split(ad_config[0], ':')]
        context.glContext.config['adcomp_num'] = int(ad_config[1])
        context.glContext.config['comm_fo'] = [int(i) for i in str.split(ad_config[2], ':')]
        context.glContext.config['enable_pc'] = True if ad_config[3] == 'true' else False
        context.glContext.config['m_ad'] = int(ad_config[4])
        context.glContext.config['enab_adap_m'] =True if ad_config[5] == 'true' else False
        context.glContext.config['num_threads'] =int(ad_config[6])

    context.glContext.ipInit(args.servers, args.workers)
    context.glContext.initGraph()
    printContext()


def printContext():
    print("*************************context info****************************")
    for id in context.glContext.config:
        print("{0} = {1}".format(id,context.glContext.config[id]))
    print("*************************context info****************************")
