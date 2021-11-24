import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset
from functools import reduce

def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""

	df = pd.read_csv(path)

	df_train = df.drop('y', axis = 1).values.astype(np.float32)
	df_target = (df['y'] - 1).values

	if model_type == 'MLP':
		data = torch.tensor(df_train)
		target = torch.tensor(df_target)
		dataset = TensorDataset(data, target)
	elif model_type == 'CNN':
		data = df.loc[:, 'X1':'X178'].values
		target = torch.tensor(df_target)
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(1), target)
	elif model_type == 'RNN':
		data = df.loc[:, 'X1':'X178'].values
		target = torch.tensor(df_target)
		dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(2), target)
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset


def calculate_num_features(seqs):
	"""
	:param seqs:
	:return: the calculated number of features
	"""
	features = reduce(lambda a, b: a + b, seqs)
	features = reduce(lambda a, b: a + b, features)
	n_features = int(max(features) + 1)
	return n_features


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels

		self.seqs = [i for i in range(len(labels))]
		for i in range(len(seqs)):
			num_visit = seqs[i].__len__()
			tmp = np.zeros(shape=(num_visit, num_features),dtype='int16')
			for n in range(num_visit):
				tmp[n][seqs[i][n]] = 1
			self.seqs[i] = tmp

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq_1, label_1), (seq_2, label_2), ... , (seq_N, label_N)]
	where N is minibatch size, seq_i is a Numpy (or Scipy Sparse) array, and label is an int value

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	n_visit     = [x[0].__len__() for x in batch]
	order_visit = np.argsort(np.array(n_visit) * -1)
	order_Batch = [batch[i] for i in order_visit]
	lab_list    = [x[1] for x in order_Batch]
	seq_        = [x[0] for x in order_Batch]
	len_list    = [x[0].__len__() for x in order_Batch]
	max_len     = np.max(len_list)
	seq_list    = []

	for i in range(len(len_list)):
		length = seq_[i].shape[0]
		num_features = seq_[i].shape[1]

		if length < max_len:
			padded = np.concatenate(
				(seq_[i], np.zeros((max_len - length, num_features), dtype=np.float32)), axis=0)
		else:
			padded = seq_[i]

		seq_list.append(padded)
		# seq_list[i] = np.pad(seq_[i], ((max_len - len_list[i], 0), (0, 0)), 'constant', constant_values=(0, 0))

	seqs_tensor = torch.from_numpy(np.stack(seq_list, axis=0)).float()
	lengths_tensor = torch.LongTensor(np.asarray(len_list, dtype='long'))
	labels_tensor = torch.LongTensor(np.asarray(lab_list, dtype='long'))

	return (seqs_tensor, lengths_tensor), labels_tensor
