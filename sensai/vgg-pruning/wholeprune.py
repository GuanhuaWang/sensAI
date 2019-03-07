import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
from collections import Counter
#from cifar10class import *


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class ModifiedVGG16Model(torch.nn.Module):
	def __init__(self):
		super(ModifiedVGG16Model, self).__init__()

		model = models.vgg16(pretrained=True)
		self.features = model.features

		for param in self.features.parameters():
			param.requires_grad = False

		self.classifier = nn.Sequential(
		    nn.Dropout(),
		    nn.Linear(512, 512),
		    nn.ReLU(inplace=True),
		    nn.Dropout(),
		    nn.Linear(512, 512),
		    nn.ReLU(inplace=True),
		    nn.Linear(512, 2))

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

class FilterPrunner:
	def __init__(self, model):
		self.model = model
		self.reset()
	
	def reset(self):
		# self.activations = []
		# self.gradients = []
		# self.grad_index = 0
		# self.activation_to_layer = {}
		self.filter_ranks = {}

	def forward(self, x):
		self.activations = []
		self.gradients = []
		self.grad_index = 0
		self.activation_to_layer = {}

		activation_index = 0
		for layer, (name, module) in enumerate(self.model.features._modules.items()):
		    x = module(x)
		    if isinstance(module, torch.nn.modules.conv.Conv2d):
		    	x.register_hook(self.compute_rank)
		    	self.activations.append(x)
		    	self.activation_to_layer[activation_index] = layer
#		    	print("=LAYER=",layer)
		    	activation_index += 1

		return self.model.classifier(x.view(x.size(0), -1))

	def compute_rank(self, grad):
		activation_index = len(self.activations) - self.grad_index - 1
		activation = self.activations[activation_index]
		values = \
			torch.sum((activation * grad), dim = 0, keepdim = True).\
				sum(dim=2,keepdim = True).sum(dim=3,keepdim = True)[0, :, 0, 0].data
		
		# Normalize the rank by the filter dimensions
		values = \
			values / (activation.size(0) * activation.size(2) * activation.size(3))

		if activation_index not in self.filter_ranks:
			self.filter_ranks[activation_index] = \
				torch.FloatTensor(activation.size(1)).zero_().cuda()

		self.filter_ranks[activation_index] += values
		self.grad_index += 1

	def lowest_ranking_filters(self, num):
		data = []
		for i in sorted(self.filter_ranks.keys()):
			for j in range(self.filter_ranks[i].size(0)):
				data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

		return nsmallest(num, data, itemgetter(2))

	def normalize_ranks_per_layer(self):
		for i in self.filter_ranks:
			v = torch.abs(self.filter_ranks[i])
			v = v / np.sqrt(torch.sum(v * v)).cuda()
			self.filter_ranks[i] = v.cpu()

	def get_prunning_plan(self, num_filters_to_prune):
		filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

		# After each of the k filters are prunned,
		# the filter index of the next filters change since the model is smaller.
		filters_to_prune_per_layer = {}
		for (l, f, _) in filters_to_prune:
			if l not in filters_to_prune_per_layer:
				filters_to_prune_per_layer[l] = []
			filters_to_prune_per_layer[l].append(f)

		for l in filters_to_prune_per_layer:
			filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
			for i in range(len(filters_to_prune_per_layer[l])):
				filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

		filters_to_prune = []
		for l in filters_to_prune_per_layer:
			for i in filters_to_prune_per_layer[l]:
				filters_to_prune.append((l, i))

		return filters_to_prune				

class PrunningFineTuner_VGG16:
	def __init__(self, train_path, test_path, model):

		self.train_data_loader = dataset.loader(train_path)
		self.test_data_loader = dataset.test_loader(test_path)

		self.model = model
		self.criterion = torch.nn.CrossEntropyLoss()
		self.prunner = FilterPrunner(self.model) 
		self.model.train()

	def test(self):
		self.model.eval()
		correct = 0
		total = 0

		for i, (batch, label) in enumerate(self.test_data_loader):
			batch = batch.cuda()
			output = model(Variable(batch))
			print("output ", output.data)
			pred = output.data.max(1)[1]
			print("pred ", pred)
			print("label is ", label)
			correct += pred.cpu().eq(label).sum()
			total += label.size(0)
#		print("label size: ",label.size(0))	 	
		print("Accuracy : ", float(correct) / total)
		self.model.train()

	def train(self, optimizer = None, epoches = 10):
		if optimizer is None:
			optimizer = \
				optim.SGD(model.classifier.parameters(), 
					lr=0.01, momentum=0.9)

		for i in range(epoches):
			print("Epoch: ", i)
			self.train_epoch(optimizer)
			self.test()
		print("Finished fine tuning.")
		

	def train_batch(self, optimizer, batch, label, rank_filters):
		self.model.zero_grad()
		input = Variable(batch)

		if rank_filters:
			output = self.prunner.forward(input)
			self.criterion(output, Variable(label)).backward()
		else:
			self.criterion(self.model(input), Variable(label)).backward()
			optimizer.step()

	def train_epoch(self, optimizer = None, rank_filters = False):
		for batch, label in self.train_data_loader:
			self.train_batch(optimizer, batch.cuda(), label.cuda(), rank_filters)
#			print("train label: ", label)

	def get_candidates_to_prune(self, num_filters_to_prune):
		self.prunner.reset()

		self.train_epoch(rank_filters = True)
		
		self.prunner.normalize_ranks_per_layer()

		return self.prunner.get_prunning_plan(num_filters_to_prune)
		
	def total_num_filters(self):
		filters = 0
		for name, module in self.model.features._modules.items():
			if isinstance(module, torch.nn.modules.conv.Conv2d):
				filters = filters + module.out_channels
		return filters

	def prune(self):
		#Get the accuracy before prunning
		self.test()

		self.model.train()

		#Make sure all the layers are trainable
		for param in self.model.features.parameters():
			param.requires_grad = True

		number_of_filters = self.total_num_filters()
		num_filters_to_prune_per_iteration = 512
		iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

		iterations = int(iterations * 2.0 / 3)

		print("Number of prunning iterations to reduce 67% filters", iterations)

		for _ in range(iterations):
			print("Ranking filters.. ")
			prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
			layers_prunned = {}
			for layer_index, filter_index in prune_targets:
				if layer_index not in layers_prunned:
					layers_prunned[layer_index] = 0
				layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

			print("Layers that will be prunned", layers_prunned)
			print("Prunning filters.. ")
			model = self.model.cpu()
			for layer_index, filter_index in prune_targets:
				model = prune_vgg16_conv_layer(model, layer_index, filter_index)

			self.model = model.cuda()

			message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
			print("Filters prunned", str(message))
			self.test()
			print("Fine tuning to recover from prunning iteration.")
			optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
			self.train(optimizer, epoches = 10)


		print("Finished. Going to fine tune the model a bit more")
		self.train(optimizer, epoches = 15)
		torch.save(model.state_dict(), "model_prunned")


def prune_candidate(class_index):
	layers_channels = np.load(open("lc_logs", "rb"))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test1")
    parser.add_argument('--class-index', default=0, type=int,
                    help='class index for pruning')
    parser.add_argument("--prune_trained", dest="prune_trained", action="store_true")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    # Architecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    parser.set_defaults(prune_trained=True)
    args = parser.parse_args()
    return args


"""
	Returns a list of candidates for each layer of filters to prune
	output: [candidate_filters_layer1, ... ,candidate_filters_layer-n]
"""
def pruning_candidates(class_index):
	layers_channels = np.load(open("lc_logs/class_{}_lc.npy".format(class_index), "rb")) 	
	print(sum([l.shape[0] for l in layers_channels]))
        # Find channels to prune across each layer, list of layers, with list of channels to prune
	candidates = []
	count = 0
		## Local information
	for index, layer in enumerate(layers_channels):
		percentile_bins = np.percentile(layer, [20, 40, 60, 80])
		print("Percentile bins: ", percentile_bins)
                # indices is an array where each element of layer is mapped to a percentile bin index
		indices = np.digitize(layer, percentile_bins)
		mean = np.mean(layer)
		std = np.std(layer)
		count += Counter(indices)[1] 
		print(type(indices[0]))
		# Mark as candidate if its filter is below a particular percentile for a particular layer
		candidates.append( [i for i in range(len(layer)) if indices[i] < 3])
	"""
	print(sum([len(c) for c in candidates]))
       
	# global information
	idx, channels = enumerate(layers_channels)
	all_channels = zip(ixd, channels)
	all_channels = np.flatten(all_channels))
	percentile_bins = np.percentile(np.flatten(layers_channels), [20, 40, 60, 80])
	print("Percentile bins: ", percentile_bins)
	indices = np.digitize(all_channels, percentile_bins)
	global_candidates = [i for i in range(len(all_channels)) if indices[i] < 3]
	"""
		
	return candidates
    #print("Total pruning candidates: {}".format(sum([len(l) for l in candidates])))
		 
	
"""
	python3 whole-prune.py -a vgg19_bn --resume checkpoints/cifar10/vgg19_bn/checkpoint.pth.tar --class-index 0 
"""

if __name__ == '__main__':
	args = get_args()

	if args.train:
		model = ModifiedVGG16Model().cuda()
	elif args.prune:
		model = torch.load("model").cuda()
	# elif args.prune_trained:	
		## model = torch.nn.DataParallel(model).cuda()
		## model.eval()        pass

	pruning_candidates(args.class_index)
		
	fine_tuner = PrunningFineTuner_VGG16(args.train_path, args.test_path, model)

	if args.train:
		fine_tuner.train(epoches = 20)
		torch.save(model, "model")

	elif args.prune:
		fine_tuner.prune()
