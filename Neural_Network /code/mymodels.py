import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(178, 16),
			nn.Sigmoid(),
			nn.Linear(16, 5)
		)

		self.my_model = nn.Sequential(
			nn.BatchNorm1d(178),
			nn.Linear(178,89),
			nn.Dropout(p=0.3),
			nn.ReLU(),
			nn.Linear(89, 16),
			nn.Dropout(p=0.3),
			nn.ReLU(),
			nn.Linear(16, 5)
		)

	def forward(self, x):
		x = self.my_model(x)
		return x


class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.input = nn.Sequential(
			nn.Conv1d(in_channels = 1, out_channels = 6, kernel_size = 5),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size = 2, stride = 2),
			nn.Conv1d(in_channels = 6, out_channels = 16, kernel_size = 5),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size = 2, stride = 2),
		)

		self.output = nn.Sequential(
			nn.Linear(in_features = 16 * 41, out_features=128),
			nn.ReLU(),
			nn.Linear(128, 5),
		)

		self.my_input = nn.Sequential(
			nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2, stride=2),
			nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2, stride=2),
			nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2, stride=2),
		)

		self.my_output = nn.Sequential(
			nn.Linear(in_features=32 * 18, out_features=128),
			nn.Dropout(p=0.3),
			nn.ReLU(),
			nn.Linear(128, 5),
		)

	def forward(self, x):
		#x = self.input(x)
		#x = x.view(-1, 16 * 41)
		x = self.my_input(x)
		x = x.view(-1, 32 * 18)
		#x = self.output(x)
		x = self.my_output(x)
		return x


class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.input = nn.Sequential(
			nn.GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True),
		)

		self.output = nn.Sequential(
			nn.Linear(16, 5)
		)

		self.my_input = nn.Sequential(
			nn.LSTM(input_size=1, hidden_size=16, num_layers=2, batch_first=True, dropout=0.1),
		)

		self.my_output = nn.Sequential(
			nn.ReLU(),
			nn.Linear(16, 5)
		)

	def forward(self, x):
		# x, _ = self.input(x)
		# x = self.output(x[:,-1,:])
		x, _ = self.my_input(x)
		x = self.my_output(x[:, -1, :])
		return x


class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
		self.fc1 = nn.Sequential(
				nn.Dropout(p=0.3),
				nn.Linear(dim_input,32),
				nn.Tanh(),
		)
		self.rnn = nn.GRU(input_size=32, hidden_size=16, num_layers=1,dropout=0.3, batch_first=True)
		self.fc2 = nn.Linear(16,2)

	def forward(self, input_tuple):
		seqs, lengths = input_tuple
		seqs = self.fc1(seqs)
	#	seqs = pack_padded_sequence(seqs, lengths, batch_first=True, enforce_sorted=False)
		seqs,_ = self.rnn(seqs)
		seqs = seqs[:, -1, :]
		seqs = self.fc2(seqs)

		return seqs
