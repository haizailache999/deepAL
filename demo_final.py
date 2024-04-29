import argparse
import numpy as np
import warnings
import torch
from utils import get_dataset, get_net, get_net_lpl, get_net_waal, get_strategy
from pprint import pprint
from AL_model import AL_Net,Architect
torch.set_printoptions(profile='full')
from torch.utils.data import DataLoader, Subset
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import get_idx
from inter_dataset import myDataset
import sys
import os
import re
import random
import math
import datetime
from tqdm import tqdm
import arguments
from parameters import *
from utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.empty_cache()

# parameters
if __name__ == '__main__':
	warnings.filterwarnings('ignore')
	args_input = arguments.get_args()
	NUM_QUERY = args_input.batch
	NUM_INIT_LB = args_input.initseed
	NUM_ROUND = int(args_input.quota / args_input.batch)
	DATA_NAME = args_input.dataset_name
	STRATEGY_NAME = args_input.ALstrategy
	print(NUM_ROUND,STRATEGY_NAME)

	SEED = args_input.seed
	os.environ['TORCH_HOME']='./basicmodel'
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args_input.gpu)

	# fix random seed
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.backends.cudnn.enabled  = True
	torch.backends.cudnn.benchmark= True

	# device
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda")

	#recording
	#sys.stdout = Logger(os.path.abspath('') + '/logfile/' + DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_normal_log.txt')
	#warnings.filterwarnings('ignore')

	# start experiment

	iteration = args_input.iteration

	all_acc = []
	all_acc1=[]
	acq_time = []

	# repeate # iteration trials
	while (iteration > 0):
		#model1=AL_Net().to(device)
		#architect = Architect(model1) 
		iteration = iteration - 1
		# data, network, strategy
		args_task = args_pool[DATA_NAME]
		dataset = get_dataset(args_input.dataset_name, args_task)				# load dataset
		if args_input.ALstrategy == 'LossPredictionLoss':
			net = get_net_lpl(args_input.dataset_name, args_task, device)		# load network
		elif args_input.ALstrategy == 'WAAL':
			net = get_net_waal(args_input.dataset_name, args_task, device)		# load network
		else:
			net = get_net(args_input.dataset_name, args_task, device)			# load network
		strategy = get_strategy(args_input.ALstrategy, dataset, net, args_input, args_task)  # load strategy
		start = datetime.datetime.now()

        
		# generate initial labeled pool
		dataset.initialize_labels(args_input.initseed)       #random get 1000 samples

		#record acc performance
		acc = np.zeros(NUM_ROUND + 1)
		acc1 = np.zeros(20)
		# only for special cases that need additional data
		new_X = torch.empty(0)
		new_Y = torch.empty(0)
			
		# print info
		print(DATA_NAME)
		print('RANDOM SEED {}'.format(SEED))
		print(type(strategy).__name__)
		
		# round 0 accuracy
		if args_input.ALstrategy == 'WAAL':
			strategy.train(model_name = args_input.ALstrategy)
		else:
			strategy.train()
		preds = strategy.predict(dataset.get_test_data())
		acc[0] = dataset.cal_test_acc(preds)
		print('Round 0\ntesting accuracy {}'.format(acc[0]))
		print('\n')
		
		for rd in range(1, NUM_ROUND+1):
			print('Round {}'.format(rd))
			#unlabeled_idxs, unlabeled_data = dataset.get_unlabeled_data()
			labeled_idxs,labeled_data=dataset.get_labeled_data()
			num_train=len(labeled_data)
			split = int(np.floor(0.5 * num_train))

			# round 1 to rd
				#print("inter_round",inter_round)
			indices = np.random.permutation(len(labeled_data))
			train_queue = torch.utils.data.DataLoader(
				labeled_data, batch_size=NUM_QUERY,
				sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
				pin_memory=True, num_workers=2)

			valid_queue = torch.utils.data.DataLoader(
				labeled_data, batch_size=NUM_QUERY,
				sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
				pin_memory=True, num_workers=2)
				#print("this is valid queue",len(valid_queue.dataset))
			if rd==1:
				inter_round_total=3
			else:
				inter_round_total=1
			for inter_round in range(inter_round_total):
				for step, (input, target,idxs) in enumerate(tqdm(train_queue,file=sys.stdout)):
					input = Variable(input, requires_grad=False).cuda()
					target = Variable(target, requires_grad=False).cuda()
					input_search, target_search,idxs1 = next(iter(valid_queue))
					#print(idxs1)
					#input_search = Variable(input_search, requires_grad=False).cuda()
					#target_search = Variable(target_search, requires_grad=False).cuda()
					#subset = Subset(valid_queue.dataset, indices=range(step*input_search.size(0),(step+1)*input_search.size(0)))
					subset= TensorDataset(input_search,target_search,torch.arange(0, input_search.size(0), step=1))
					new_dataloader = DataLoader(subset, batch_size=NUM_QUERY, shuffle=True)
					#first_subset=Subset(train_queue.dataset, indices=range(step*input.size(0),(step+1)*input.size(0)))
					first_subset=TensorDataset(input,target,torch.arange(0, input.size(0), step=1))
					first_dataloader = DataLoader(first_subset, batch_size=NUM_QUERY, shuffle=True)
					#high_confident_idx = []
					#high_confident_pseudo_label = []
					# query  
					#unlabeled_data[0] tensor data, [1] label, [2] 0/1
					#unlabeled_idxs, unlabeled_data = dataset.get_unlabeled_data()
					#print(len(unlabeled_data))
					#loader = DataLoader(unlabeled_data, shuffle=False, batch_size=128,num_workers=0)
					#for batch_idx,(x, y, idxs) in enumerate(loader):
						#x=x.to(device)
					'''output=torch.empty(0,2)
					loader=tqdm(loader, file=sys.stdout)
					model1.train()
					for batch_idx,(x, y, idxs) in enumerate(loader):
						x=x.to(device)
						out=model1(x)     #(batch,17)
						output=torch.cat((output, out.to('cpu')), 0)
					output=strategy.train_1(loader)
					print(output.shape)
					output_list=output.to('cpu').detach().numpy()
					AL_numpy=get_idx.get_idxs(dataset, net, args_input, args_task,NUM_QUERY,unlabeled_idxs)
					print(AL_numpy.shape)
					score=np.sum(AL_numpy*output_list,axis=1)
					print("get score")
					ind_200 = np.argpartition(score, -200)[-200:]
					ind_100 = np.argpartition(score, -100)[-100:]
					unlabeled_200=np.zeros(200)
					for num_200_pos,num_200 in enumerate(ind_200):
						unlabeled_200[num_200_pos]=unlabeled_idxs[num_200]
					unlabeled_100=np.zeros(200)
					for num_100_pos,num_100 in enumerate(ind_100):
						unlabeled_100[num_100_pos]=unlabeled_idxs[num_100]
					mask = np.isin(unlabeled_200, unlabeled_100)
					filtered_array = unlabeled_200[~mask]
					#print("filt",len(filtered_array))
					#print(len(unlabeled_idxs))
					unlabeled_idxs_list=unlabeled_idxs.tolist()
					next_idxs=[]
					for element in filtered_array:
						if element not in unlabeled_idxs_list:
							print("empty",element)
						else:
							next_idxs.append(unlabeled_idxs_list.index(element))
					#next_idxs = [np.where(unlabeled_idxs == element) for element in filtered_array]
					mydataset=myDataset(next_idxs,unlabeled_data)
					myloader=DataLoader(mydataset, shuffle=False, batch_size=10,num_workers=0)
					q_idxs=ind_100'''
					'''if 'CEALSampling' in args_input.ALstrategy:
						q_idxs, new_data = strategy.query(NUM_QUERY, rd, option = args_input.ALstrategy[13:])
					else:
						q_idxs = strategy.query(NUM_QUERY)'''
					'''myloader=tqdm(myloader,file=sys.stdout)
					strategy.predict1(myloader)'''
					if rd==1 and inter_round==0:
						strategy.train_2_1(dataset,net,args_input,args_task,NUM_QUERY,labeled_idxs,new_dataloader)
					#for train_epochs in range(10):
						#print("train_epochs",train_epochs)
					strategy.train_1(first_dataloader)
					strategy.train_2(dataset,net,args_input,args_task,NUM_QUERY,labeled_idxs,new_dataloader)
						# update
						#strategy.update(q_idxs)
					#train
					'''if 'CEALSampling' in args_input.ALstrategy:
						strategy.train(new_data)
					elif args_input.ALstrategy == 'WAAL':
						strategy.train(model_name = args_input.ALstrategy)
					else:
						strategy.train()'''
					#strategy.train()
					#clf=net(dim = 32*32*3, pretrained = False, num_classes = 10).to(device)
					#print(dim)
					#clf = net(dim = 32*32*3, pretrained = self.params['pretrained'], num_classes = self.params['num_class']).to(self.device)
					#net=net.to(device)
					#clf.eval()
					'''for i_batch,batch_data in enumerate(myloader):
						test_next,test_next_label=batch_data
						test_next_valid=net(test_next.to(device))
						architect.step(test_next_valid,test_next_label,device)'''
				
					# round rd accuracy
					'''preds = strategy.predict(dataset.get_test_data())
					acc[rd] = dataset.cal_test_acc(preds)
					print('testing accuracy {}'.format(acc[rd]))
					print('\n')'''

					#torch.cuda.empty_cache()
			preds = strategy.predict1(valid_queue)
			#acc1[inter_round] = dataset.cal_test_acc(preds)
			acc1[inter_round]=(valid_queue.dataset.Y==preds).sum().item()/len(preds)
			print('validation1 accuracy {}'.format(acc1[inter_round]))
			print('\n')
			unlabeled_idxs,unlabeled_data=dataset.get_unlabeled_data()
			untrain_loader = torch.utils.data.DataLoader(
				unlabeled_data, batch_size=NUM_QUERY,
				pin_memory=True, num_workers=2,shuffle=True)
			q_idxs=strategy.predict2(untrain_loader,unlabeled_idxs,NUM_QUERY)
			strategy.update(q_idxs)
			#for train_epoch in range(10):
			strategy.train()
			preds = strategy.predict(dataset.get_test_data())
			acc[rd] = dataset.cal_test_acc(preds)
			print('testing accuracy {}'.format(acc[rd]))
			print('\n')
			

		# print results
		print('SEED {}'.format(SEED))
		print(type(strategy).__name__)
		print(acc)
		all_acc.append(acc)
		all_acc1.append(acc1)
		
		#save model
		timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
		model_path = './modelpara/'+timestamp + DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota)  +'.params'
		end = datetime.datetime.now()
		acq_time.append(round(float((end-start).seconds),3))
		torch.save(strategy.get_model().state_dict(), model_path)
		
	# cal mean & standard deviation
	acc_m = []
	acc_m1=[]
	file_name_res_tot = DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_normal_res_tot.txt'
	file_res_tot =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name_res_tot),'w')

	file_res_tot.writelines('dataset: {}'.format(DATA_NAME) + '\n')
	file_res_tot.writelines('AL strategy: {}'.format(STRATEGY_NAME) + '\n')
	file_res_tot.writelines('number of labeled pool: {}'.format(NUM_INIT_LB) + '\n')
	file_res_tot.writelines('number of unlabeled pool: {}'.format(dataset.n_pool - NUM_INIT_LB) + '\n')
	file_res_tot.writelines('number of testing pool: {}'.format(dataset.n_test) + '\n')
	file_res_tot.writelines('batch size: {}'.format(NUM_QUERY) + '\n')
	file_res_tot.writelines('quota: {}'.format(NUM_ROUND*NUM_QUERY)+ '\n')
	file_res_tot.writelines('time of repeat experiments: {}'.format(args_input.iteration)+ '\n')

	# result
	for i in range(len(all_acc1)):
		acc_m1.append(get_aubc(args_input.quota, NUM_QUERY, all_acc1[i]))
		print("inter_acc"+str(i)+': '+str(acc_m1[i]))
		file_res_tot.writelines(str(i)+': '+str(acc_m1[i])+'\n')
	for i in range(len(all_acc)):
		acc_m.append(get_aubc(args_input.quota, NUM_QUERY, all_acc[i]))
		print(str(i)+': '+str(acc_m[i]))
		file_res_tot.writelines(str(i)+': '+str(acc_m[i])+'\n')
	mean_acc,stddev_acc = get_mean_stddev(acc_m)
	mean_time, stddev_time = get_mean_stddev(acq_time)

	print('mean AUBC(acc): '+str(mean_acc)+'. std dev AUBC(acc): '+str(stddev_acc))
	print('mean time: '+str(mean_time)+'. std dev time: '+str(stddev_time))

	file_res_tot.writelines('mean acc: '+str(mean_acc)+'. std dev acc: '+str(stddev_acc)+'\n')
	file_res_tot.writelines('mean time: '+str(mean_time)+'. std dev acc: '+str(stddev_time)+'\n')

	# save result

	file_name_res = DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_normal_res.txt'
	file_res =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name_res),'w')


	file_res.writelines('dataset: {}'.format(DATA_NAME) + '\n')
	file_res.writelines('AL strategy: {}'.format(STRATEGY_NAME) + '\n')
	file_res.writelines('number of labeled pool: {}'.format(NUM_INIT_LB) + '\n')
	file_res.writelines('number of unlabeled pool: {}'.format(dataset.n_pool - NUM_INIT_LB) + '\n')
	file_res.writelines('number of testing pool: {}'.format(dataset.n_test) + '\n')
	file_res.writelines('batch size: {}'.format(NUM_QUERY) + '\n')
	file_res.writelines('quota: {}'.format(NUM_ROUND*NUM_QUERY)+ '\n')
	file_res.writelines('time of repeat experiments: {}'.format(args_input.iteration)+ '\n')
	avg_acc = np.mean(np.array(all_acc),axis=0)
	avg_acc1 = np.mean(np.array(all_acc1),axis=0)
	for i in range(len(avg_acc1)):
		tmp = 'Size of training set is ' + str(args_input.initseed) + ', ' + 'inter accuracy is ' + str(round(avg_acc1[i],4)) + '.' + '\n'
		file_res.writelines(tmp)
	for i in range(len(avg_acc)):
		tmp = 'Size of training set is ' + str(NUM_INIT_LB + i*NUM_QUERY) + ', ' + 'accuracy is ' + str(round(avg_acc[i],4)) + '.' + '\n'
		file_res.writelines(tmp)

	file_res.close()
	file_res_tot.close()
