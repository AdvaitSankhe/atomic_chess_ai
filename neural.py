import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
from torch import nn,optim
import chess
import chess.variant
import chess.svg
from data import *
import random
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import TensorDataset, DataLoader
pgn = []
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open('my_pgn.pgn','r') as file:
  
    # reading each line   
    for line in file:
  
        # reading each word       
        for word in line.split():
  
            # displaying the words          
            #print(word)
            pgn.append(word)

def board_vectorize(pgn):
	final_boards = []
	for _ in range(100000):
		game = chess.pgn.read_game(pgn)
		board = game.board()
		if game.headers['Termination'] == 'Time forfeit' or game.headers['Result'] == '1/2-1/2' or int(game.headers['WhiteElo']) < 2000:
			continue
		#print(game.mainline_moves())
		#print(game.headers)
		move_count = 0
		for move in game.mainline_moves():
			move_count+=1
		if move_count < 15:
			continue
		total_moves = move_count
		move_count = 0
		#print(move_count)	
		board_matrices = []
		for move in game.mainline_moves():
			board.push(move)
			board_matrices.append(fen_to_board(str(board.fen)))
			winner = 1
			if game.headers['Result'] == '0-1':
				winner = -1
			final_boards.append((board_matrices[move_count],float(winner)*move_count/total_moves))
			move_count += 1
	return final_boards


def label_to_num(board_matrix):
	mapping = {'p' : -1,'P' :1, 'n' : -2,'N':2,'q': -5,'Q': 5,'b': -3,'B':3,'k':-10,'K': 10,'r': -4,'R':4,'-' :0}
	for i in range(8):
		for j in range(8):
			board_matrix[i][j] = mapping[board_matrix[i][j]]
	x = torch.FloatTensor(board_matrix)
	#print(x)
	return x
	#return F.one_hot(x)
'''
X = []
Y = []
pgn = open('lichess_db_atomic_rated_2023-02.pgn')
for board_matrix in board_vectorize(pgn):
	X.append(label_to_num(board_matrix[0]).flatten())
	Y.append(torch.tensor(board_matrix[1]))
	#print(X)
	#print(X.shape)
	#X =	X.flatten()
	#print(X)
	#print(X.shape)
	#break
X = torch.stack(X)
Y = torch.stack(Y)
print(X)
print(X.shape)
print("HEY")
print(Y)

X_test = []
Y_test = []
pgn = open('lichess_db_atomic_rated_2023-01.pgn')
for board_matrix in board_vectorize(pgn):
	X_test.append(label_to_num(board_matrix[0]).flatten())
	Y_test.append(torch.tensor(board_matrix[1]))
	#print(X)
	#print(X.shape)
	#X =	X.flatten()
	#print(X)
	#print(X.shape)
	#break
X_test = torch.stack(X_test)
Y_test = torch.stack(Y_test)
print(X_test)
print(X_test.shape)
print("HEY_TEST")
print(Y_test)
'''
class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork,self).__init__()
		self.input_layer = nn.Linear(64,512)
		self.h1 = nn.Linear(512,512)
		self.h2 = nn.Linear(512,64)
		self.output = nn.Linear(64,1)
	def forward(self,x):
		x = F.relu(self.input_layer(x))
		x = F.relu(self.h1(x))
		x = F.relu(self.h2(x))
		x = F.tanh(self.output(x))

		return x

model = NeuralNetwork().to(DEVICE)
print(model)



class Data(TensorDataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
   
    def __len__(self):
        return self.len
   
batch_size = 8
'''
train_data = Data(X,Y)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
with open('train_data.txt','wb') as f:
	pickle.dump(train_dataloader,f)
'''
with open('train_data.txt','rb') as f:
	train_dataloader = pickle.load(f)
'''
test_data = Data(X_test,Y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

with open('test_data.txt','wb') as f:
	pickle.dump(test_dataloader,f)
'''

with open('test_data.txt','rb') as f:
	test_dataloader = pickle.load(f)


#with open('nn4_512layers.txt','rb') as f:
	#model = pickle.load(f)


learning_rate = 0.1

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


num_epochs = 0
loss_values = []

for epoch in range(num_epochs):
    for X, Y in train_dataloader:
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)
        # zero the parameter gradients
        optimizer.zero_grad()
       
        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, Y.unsqueeze(-1))
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
    print(loss)
    print(epoch)

'''
#print(X.shape)
#print("HERE")
#print(X)
#print(train_data.X.shape)
'''
#with open('nn4_512layers.txt','wb') as f:
#	pickle.dump(model,f)
print(model)
'''

'''
#50 => 1273550
#10 => 254710
'''
step = np.linspace(0, 100,1273550)

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
'''


y_pred = []
y_test = []
total,correct = 0,0
with torch.no_grad():
    for X, y in test_dataloader:
        X = X.to(DEVICE)
        #y = y.to(DEVICE)
        outputs = model(X)
        outputs = outputs.to('cpu')
        #print(outputs)
        predicted = np.where(outputs < 0, -1, 1)
        #print(predicted)
        predicted = list(itertools.chain(*predicted))
        #print(predicted)
        y_pred.append(predicted)
        y_test.append(y)
        total += y.size(0)
        correct += (predicted*y.numpy() > 0).sum().item()
    print(model)
    print(y_pred[0])
    print(y_test[0])
print("DONE")
print(f'Accuracy of the network on the {total} test instances: {100 * correct / total}%')