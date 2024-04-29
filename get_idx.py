#from utils import get_dataset, get_net, get_net_lpl, get_net_waal, get_strategy
'''from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
								LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
								KMeansSampling, KMeansSamplingGPU, KCenterGreedy, KCenterGreedyPCA, BALDDropout,  \
								AdversarialBIM, AdversarialDeepFool, VarRatio, MeanSTD, BadgeSampling, CEALSampling, \
								LossPredictionLoss, VAAL, WAAL'''

from query_strategy import LeastConfidence, MarginSampling, EntropySampling,KMeansSampling,KCenterGreedyPCA, BALDDropout,VarRatio, MeanSTD, BadgeSampling
import numpy as np
import math

def get_idxs(dataset, net, args_input, args_task,NUM_QUERY,unlabeled_idxs,loader):
    #print("dataset",len(dataset))
    net1=net
    #st1=RandomSampling(dataset, net, args_input, args_task)
    st2=LeastConfidence(dataset=dataset, net=net, args_input=args_input, args_task=args_task,loader=loader)
    st3=MarginSampling(dataset, net, args_input, args_task,loader)
    st4=EntropySampling(dataset, net, args_input, args_task,loader)
    #st5=LeastConfidenceDropout(dataset, net, args_input, args_task)
    #st6=MarginSamplingDropout(dataset, net1, args_input, args_task)
    #st7=EntropySamplingDropout(dataset, net, args_input, args_task)
    st8=KMeansSampling(dataset, net, args_input, args_task,loader)
    #st9=KMeansSamplingGPU(dataset, net, args_input, args_task)
    #st10=KCenterGreedy(dataset, net, args_input, args_task)
    #st11=KCenterGreedyPCA(dataset, net, args_input, args_task,loader)
    st12=BALDDropout(dataset, net, args_input, args_task,loader)
    st13=VarRatio(dataset, net, args_input, args_task,loader)
    st14=MeanSTD(dataset, net, args_input, args_task,loader)
    #st15=BadgeSampling(dataset, net, args_input, args_task,loader)
    #st16=AdversarialBIM(dataset, net, args_input, args_task)
    #st17=AdversarialDeepFool(dataset, net, args_input, args_task)
    #st_list=[st1,st2,st3,st4,st5,st6,st7,st8,st10,st11,st12,st13,st14,st15]
    #st_list=[st2,st3,st4,st8,st12,st13,st14]
    st_list=[st2,st3]
    #print("loader length",len(loader.dataset))
    result_list=np.zeros((len(loader.dataset), len(st_list)))
    #print(NUM_QUERY)
    for i,strategy in enumerate(st_list):
        if len(loader.dataset)>NUM_QUERY:
            #print("yes")
            q_idx=strategy.query(1000)
        else:
            if math.ceil(len(loader.dataset)*0.02)==1:
                num_q=2
            else:
                num_q=math.ceil(len(loader.dataset)*0.02)
            q_idx=strategy.query(num_q)
        for t in q_idx:
            result_list[t][i]=1
        #for t_pos,t in enumerate(unlabeled_idxs):
            #if t in q_idx:
                #result_list[t_pos][i]=1
    #print(result_list)
    return result_list
