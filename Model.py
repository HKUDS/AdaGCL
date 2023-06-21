from torch import nn
import torch.nn.functional as F
import torch
from Params import args
from copy import deepcopy
import numpy as np
import math
import scipy.sparse as sp
from Utils.Utils import contrastLoss, calcRegLoss, pairPredict
import time
import torch_sparse

init = nn.init.xavier_uniform_

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

	def forward_gcn(self, adj):
		iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

		embedsLst = [iniEmbeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		mainEmbeds = sum(embedsLst)

		return mainEmbeds[:args.user], mainEmbeds[args.user:]

	def forward_graphcl(self, adj):
		iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

		embedsLst = [iniEmbeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		mainEmbeds = sum(embedsLst)

		return mainEmbeds

	def forward_graphcl_(self, generator):
		iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

		embedsLst = [iniEmbeds]		
		count = 0
		for gcn in self.gcnLayers:
			with torch.no_grad():
				adj = generator.generate(x=embedsLst[-1], layer=count)
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
			count += 1
		mainEmbeds = sum(embedsLst)

		return mainEmbeds

	def loss_graphcl(self, x1, x2, users, items):
		T = args.temp
		user_embeddings1, item_embeddings1 = torch.split(x1, [args.user, args.item], dim=0)
		user_embeddings2, item_embeddings2 = torch.split(x2, [args.user, args.item], dim=0)

		user_embeddings1 = F.normalize(user_embeddings1, dim=1)
		item_embeddings1 = F.normalize(item_embeddings1, dim=1)
		user_embeddings2 = F.normalize(user_embeddings2, dim=1)
		item_embeddings2 = F.normalize(item_embeddings2, dim=1)

		user_embs1 = F.embedding(users, user_embeddings1)
		item_embs1 = F.embedding(items, item_embeddings1)
		user_embs2 = F.embedding(users, user_embeddings2)
		item_embs2 = F.embedding(items, item_embeddings2)

		all_embs1 = torch.cat([user_embs1, item_embs1], dim=0)
		all_embs2 = torch.cat([user_embs2, item_embs2], dim=0)

		all_embs1_abs = all_embs1.norm(dim=1)
		all_embs2_abs = all_embs2.norm(dim=1)
	
		sim_matrix = torch.einsum('ik,jk->ij', all_embs1, all_embs2) / torch.einsum('i,j->ij', all_embs1_abs, all_embs2_abs)
		sim_matrix = torch.exp(sim_matrix / T)
		pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]
		loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
		loss = - torch.log(loss)

		return loss

	def getEmbeds(self):
		self.unfreeze(self.gcnLayers)
		return torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

	def unfreeze(self, layer):
		for child in layer.children():
			for param in child.parameters():
				param.requires_grad = True

	def getGCN(self):
		return self.gcnLayers

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds, flag=True):
		if (flag):
			return torch.spmm(adj, embeds)
		else:
			return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)

class vgae_encoder(Model):
	def __init__(self):
		super(vgae_encoder, self).__init__()
		hidden = args.latdim
		self.encoder_mean = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
		self.encoder_std = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.Softplus())

	def forward(self, adj):
		x = self.forward_graphcl(adj)

		x_mean = self.encoder_mean(x)
		x_std = self.encoder_std(x)
		gaussian_noise = torch.randn(x_mean.shape).cuda()
		x = gaussian_noise * x_std + x_mean
		return x, x_mean, x_std

class vgae_decoder(nn.Module):
	def __init__(self, hidden=args.latdim):
		super(vgae_decoder, self).__init__()
		self.decoder = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
		self.sigmoid = nn.Sigmoid()
		self.bceloss = nn.BCELoss(reduction='none')

	def forward(self, x, x_mean, x_std, users, items, neg_items, encoder):
		x_user, x_item = torch.split(x, [args.user, args.item], dim=0)

		edge_pos_pred = self.sigmoid(self.decoder(x_user[users] * x_item[items]))
		edge_neg_pred = self.sigmoid(self.decoder(x_user[users] * x_item[neg_items]))

		loss_edge_pos = self.bceloss( edge_pos_pred, torch.ones(edge_pos_pred.shape).cuda() )
		loss_edge_neg = self.bceloss( edge_neg_pred, torch.zeros(edge_neg_pred.shape).cuda() )
		loss_rec = loss_edge_pos + loss_edge_neg

		kl_divergence = - 0.5 * (1 + 2 * torch.log(x_std) - x_mean**2 - x_std**2).sum(dim=1)

		ancEmbeds = x_user[users]
		posEmbeds = x_item[items]
		negEmbeds = x_item[neg_items]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
		regLoss = calcRegLoss(encoder) * args.reg
		
		beta = 0.1
		loss = (loss_rec + beta * kl_divergence.mean() + bprLoss + regLoss).mean()
		
		return loss

class vgae(nn.Module):
	def __init__(self, encoder, decoder):
		super(vgae, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, data, users, items, neg_items):
		x, x_mean, x_std = self.encoder(data)
		loss = self.decoder(x, x_mean, x_std, users, items, neg_items, self.encoder)
		return loss

	def generate(self, data, edge_index, adj):
		x, _, _ = self.encoder(data)

		edge_pred = self.decoder.sigmoid(self.decoder.decoder(x[edge_index[0]] * x[edge_index[1]]))

		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		edge_pred = edge_pred[:, 0]
		mask = ((edge_pred + 0.5).floor()).type(torch.bool)
		
		newVals = vals[mask]

		newVals = newVals / (newVals.shape[0] / edgeNum[0])
		newIdxs = idxs[:, mask]
		
		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)

class DenoisingNet(nn.Module):
	def __init__(self, gcnLayers, features):
		super(DenoisingNet, self).__init__()

		self.features = features

		self.gcnLayers = gcnLayers

		self.edge_weights = []
		self.nblayers = []
		self.selflayers = []

		self.attentions = []
		self.attentions.append([])
		self.attentions.append([])

		hidden = args.latdim

		self.nblayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.nblayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

		self.selflayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
		self.selflayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

		self.attentions_0 = nn.Sequential(nn.Linear( 2 * hidden, 1))
		self.attentions_1 = nn.Sequential(nn.Linear( 2 * hidden, 1))

	def freeze(self, layer):
		for child in layer.children():
			for param in child.parameters():
				param.requires_grad = False

	def get_attention(self, input1, input2, layer=0):
		if layer == 0:
			nb_layer = self.nblayers_0
			selflayer = self.selflayers_0
		if layer == 1:
			nb_layer = self.nblayers_1
			selflayer = self.selflayers_1

		input1 = nb_layer(input1)
		input2 = selflayer(input2)

		input10 = torch.concat([input1, input2], axis=1)

		if layer == 0:
			weight10 = self.attentions_0(input10)
		if layer == 1:
			weight10 = self.attentions_1(input10)
		
		return weight10

	def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
		gamma = args.gamma
		zeta = args.zeta

		if training:
			debug_var = 1e-7
			bias = 0.0
			np_random = np.random.uniform(low=debug_var, high=1.0-debug_var, size=np.shape(log_alpha.cpu().detach().numpy()))
			random_noise = bias + torch.tensor(np_random)
			gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
			gate_inputs = (gate_inputs.cuda() + log_alpha) / beta
			gate_inputs = torch.sigmoid(gate_inputs)
		else:
			gate_inputs = torch.sigmoid(log_alpha)

		stretched_values = gate_inputs * (zeta-gamma) +gamma
		cliped = torch.clamp(stretched_values, 0.0, 1.0)
		return cliped.float()

	def generate(self, x, layer=0):
		f1_features = x[self.row, :]
		f2_features = x[self.col, :]

		weight = self.get_attention(f1_features, f2_features, layer)

		mask = self.hard_concrete_sample(weight, training=False)

		mask = torch.squeeze(mask)
		adj = torch.sparse.FloatTensor(self.adj_mat._indices(), mask, self.adj_mat.shape)

		ind = deepcopy(adj._indices())
		row = ind[0, :]
		col = ind[1, :]

		rowsum = torch.sparse.sum(adj, dim=-1).to_dense()
		d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
		d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
		row_inv_sqrt = d_inv_sqrt[row]
		col_inv_sqrt = d_inv_sqrt[col]
		values = torch.mul(adj._values(), row_inv_sqrt)
		values = torch.mul(values, col_inv_sqrt)

		support = torch.sparse.FloatTensor(adj._indices(), values, adj.shape)

		return support

	def l0_norm(self, log_alpha, beta):
		gamma = args.gamma
		zeta = args.zeta
		gamma = torch.tensor(gamma)
		zeta = torch.tensor(zeta)
		reg_per_weight = torch.sigmoid(log_alpha - beta * torch.log(-gamma/zeta))

		return torch.mean(reg_per_weight)

	def set_fea_adj(self, nodes, adj):
		self.node_size = nodes
		self.adj_mat = adj

		ind = deepcopy(adj._indices())

		self.row = ind[0, :]
		self.col = ind[1, :]

	def call(self, inputs, training=None):
		if training:
			temperature = inputs
		else:
			temperature = 1.0

		self.maskes = []

		x = self.features.detach()
		layer_index = 0
		embedsLst = [self.features.detach()]

		for layer in self.gcnLayers:
			xs = []
			f1_features = x[self.row, :]
			f2_features = x[self.col, :]

			weight = self.get_attention(f1_features, f2_features, layer=layer_index)
			mask = self.hard_concrete_sample(weight, temperature, training)

			self.edge_weights.append(weight)
			self.maskes.append(mask)
			mask = torch.squeeze(mask)

			adj = torch.sparse.FloatTensor(self.adj_mat._indices(), mask, self.adj_mat.shape).coalesce()
			ind = deepcopy(adj._indices())
			row = ind[0, :]
			col = ind[1, :]

			rowsum = torch.sparse.sum(adj, dim=-1).to_dense() + 1e-6
			d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
			d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
			row_inv_sqrt = d_inv_sqrt[row]
			col_inv_sqrt = d_inv_sqrt[col]
			values = torch.mul(adj.values(), row_inv_sqrt)
			values = torch.mul(values, col_inv_sqrt)
			support = torch.sparse.FloatTensor(adj._indices(), values, adj.shape).coalesce()

			nextx = layer(support, x, False)
			xs.append(nextx)
			x = xs[0]
			embedsLst.append(x)
			layer_index += 1
		return sum(embedsLst)
	
	def lossl0(self, temperature):
		l0_loss = torch.zeros([]).cuda()
		for weight in self.edge_weights:
			l0_loss += self.l0_norm(weight, temperature)
		self.edge_weights = []
		return l0_loss

	def forward(self, users, items, neg_items, temperature):
		self.freeze(self.gcnLayers)
		x = self.call(temperature, True)
		x_user, x_item = torch.split(x, [args.user, args.item], dim=0)
		ancEmbeds = x_user[users]
		posEmbeds = x_item[items]
		negEmbeds = x_item[neg_items]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
		regLoss = calcRegLoss(self) * args.reg

		lossl0 = self.lossl0(temperature) * args.lambda0
		return bprLoss + regLoss + lossl0




		




