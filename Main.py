import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, vgae_encoder, vgae_decoder, vgae, DenoisingNet
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import calcRegLoss, pairPredict
import os
from copy import deepcopy
import scipy.sparse as sp
import random

class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')

		recallMax = 0
		ndcgMax = 0
		bestEpoch = 0

		stloc = 0
		log('Model Initialized')

		for ep in range(stloc, args.epoch):
			temperature = max(0.05, args.init_temperature * pow(args.temperature_decay, ep))
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch(temperature)
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				reses = self.testEpoch()
				if (reses['Recall'] > recallMax):
					recallMax = reses['Recall']
					ndcgMax = reses['NDCG']
					bestEpoch = ep
				log(self.makePrint('Test', ep, reses, tstFlag))
			print()
		print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax)

	def prepareModel(self):
		self.model = Model().cuda()

		encoder = vgae_encoder().cuda()
		decoder = vgae_decoder().cuda()
		self.generator_1 = vgae(encoder, decoder).cuda()
		self.generator_2 = DenoisingNet(self.model.getGCN(), self.model.getEmbeds()).cuda()
		self.generator_2.set_fea_adj(args.user+args.item, deepcopy(self.handler.torchBiAdj).cuda())

		self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
		self.opt_gen_1 = torch.optim.Adam(self.generator_1.parameters(), lr=args.lr, weight_decay=0)
		self.opt_gen_2 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator_2.parameters()), lr=args.lr, weight_decay=0, eps=args.eps)

	def trainEpoch(self, temperature):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		generate_loss_1, generate_loss_2, bpr_loss, im_loss, ib_loss, reg_loss = 0, 0, 0, 0, 0, 0
		steps = trnLoader.dataset.__len__() // args.batch

		for i, tem in enumerate(trnLoader):
			data = deepcopy(self.handler.torchBiAdj).cuda()

			data1 = self.generator_generate(self.generator_1)

			self.opt.zero_grad()
			self.opt_gen_1.zero_grad()
			self.opt_gen_2.zero_grad()

			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()

			out1 = self.model.forward_graphcl(data1)
			out2 = self.model.forward_graphcl_(self.generator_2)

			loss = self.model.loss_graphcl(out1, out2, ancs, poss).mean() * args.ssl_reg
			im_loss += float(loss)
			loss.backward()

			self.opt.step()
			self.opt.zero_grad()

			# info bottleneck
			_out1 = self.model.forward_graphcl(data1)
			_out2 = self.model.forward_graphcl_(self.generator_2)

			loss_ib = self.model.loss_graphcl(_out1, out1.detach(), ancs, poss) + self.model.loss_graphcl(_out2, out2.detach(), ancs, poss)
			loss= loss_ib.mean() * args.ib_reg
			ib_loss += float(loss)
			loss.backward()

			self.opt.step()
			self.opt.zero_grad()

			# BPR
			usrEmbeds, itmEmbeds = self.model.forward_gcn(data)
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]
			scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
			bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
			regLoss = calcRegLoss(self.model) * args.reg
			loss = bprLoss + regLoss
			bpr_loss += float(bprLoss)
			reg_loss += float(regLoss)
			loss.backward()

			loss_1 = self.generator_1(deepcopy(self.handler.torchBiAdj).cuda(), ancs, poss, negs)
			loss_2 = self.generator_2(ancs, poss, negs, temperature)

			loss = loss_1 + loss_2
			generate_loss_1 += float(loss_1)
			generate_loss_2 += float(loss_2)
			loss.backward()

			self.opt.step()
			self.opt_gen_1.step()
			self.opt_gen_2.step()

			log('Step %d/%d: gen 1 : %.3f ; gen 2 : %.3f ; bpr : %.3f ; im : %.3f ; ib : %.3f ; reg : %.3f  ' % (
				i, 
				steps,
				generate_loss_1,
				generate_loss_2,
				bpr_loss,
                im_loss,
                ib_loss,
                reg_loss,
				), save=False, oneline=True)

		ret = dict()
		ret['Gen_1 Loss'] = generate_loss_1 / steps
		ret['Gen_2 Loss'] = generate_loss_2 / steps
		ret['BPR Loss'] = bpr_loss / steps
		ret['IM Loss'] = im_loss / steps
		ret['IB Loss'] = ib_loss / steps
		ret['Reg Loss'] = reg_loss / steps

		return ret

	def testEpoch(self):
		tstLoader = self.handler.tstLoader
		epRecall, epNdcg = [0] * 2
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat
		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()
			usrEmbeds, itmEmbeds = self.model.forward_gcn(self.handler.torchBiAdj)
			allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			_, topLocs = torch.topk(allPreds, args.topk)
			recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
			epRecall += recall
			epNdcg += ndcg
			log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		return ret

	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			allRecall += recall
			allNdcg += ndcg
		return allRecall, allNdcg

	def generator_generate(self, generator):
		edge_index = []
		edge_index.append([])
		edge_index.append([])
		adj = deepcopy(self.handler.torchBiAdj)
		idxs = adj._indices()

		with torch.no_grad():
			view = generator.generate(self.handler.torchBiAdj, idxs, adj)

		return view

def seed_it(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True 
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

if __name__ == '__main__':
	with torch.cuda.device(args.gpu):
		logger.saveDefault = True
		seed_it(args.seed)
		
		log('Start')
		handler = DataHandler()
		handler.LoadData()
		log('Load Data')

		coach = Coach(handler)
		coach.run()