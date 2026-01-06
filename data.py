import chess.pgn
import random


pgn = open('lichess_db_atomic_rated_2023-02.pgn')

class player():
	def __init__(self,piece_values,pos_values):
		self.piece_values = piece_values
		self.pos_values = pos_values
		self.fitness = 0

#print(len('<bound method Board.fen of AtomicBoard('))

def fen_to_board(fen):
    board = []
    fen = fen[40:]
    for row in fen.split('/'):
        brow = []
        for c in row:
            if c == ' ' or c == '<':
                break
            elif c in '12345678':
                brow.extend( ['-'] * int(c) )
            else:
            	brow.append(c)
            '''	
            elif c == 'p':
                brow.append( 'bp' )
            elif c == 'P':
                brow.append( 'wp' )
            elif c > 'Z':
                brow.append( 'b'+c.upper() )
            else:
                brow.append( 'w'+c )
			'''
        board.append( brow )
    return board[:]
def evaluation(board_matrices,piece_values,pos_values,mate_val):
	value,weight = 0,0
	for board_matrix in board_matrices:
		for i in range(len(board_matrix)):
			for j in range(len(board_matrix[i])):
				value += weight*piece_values[board_matrix[i][j]]*pos_values[i][j]
			#print(value)
		weight += 1
		value += mate_val
		#print(board_matrix)
		#print(value)
	return value/weight


def calc_fitness(pgn,player):
	games_count = 0
	fitness = 0
	pgn = open('lichess_db_atomic_rated_2023-02.pgn')
	for _ in range(10000):

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
		#print(move_count)
		games_count +=1	
		curr_count = 0
		board_matrices = []
		mate_val = 0
		for move in game.mainline_moves():
			board.push(move)
			if board.is_checkmate():
				if curr_count%2 == 0:
					mate_val = 100
				else:
					mate_val = -100
			if curr_count%6 == 0:
				board_matrix = fen_to_board(str(board.fen))
				board_matrices.append(board_matrix)
			curr_count += 1
		#print(board)
		#print(board.fen)

		#print(board_matrix)
		winner = 1
		if game.headers['Result'] == '0-1':
			winner = -1
		fitness  += evaluation(board_matrices,player.piece_values,player.pos_values,mate_val)*winner

	return fitness

piece_values = {'p' : -1,'P' :1, 'n' : -1,'N':1,'q': -1,'Q': 1,'b': -1,'B':1,'k':-100,'K': 100,'r': -1,'R':1,'-' :0}
pos_values = [[1 for x in range(8)] for y in range(8)]

bad_piece_values = {'p' : 1,'P' :-1, 'n' : 1,'N':-1,'q': 1,'Q': -1,'b': 1,'B':-1,'k':100,'K': -100,'r': 1,'R':-1,'-' :0}
first_player = player(piece_values,pos_values)
second_player = player(bad_piece_values,pos_values)


def crossover(player1,player2):
	child_piece_values = {key: value for key, value in player1.piece_values.items()}
	for key,value in player2.piece_values.items():
		if random.random() > 0.5:
			child_piece_values[key] = value
			print(key)
	child = player(child_piece_values,player1.pos_values)
	return child
def GA(size,num_gens):
	current_population = []
	for gen in range(num_gens):
		if gen == 0:
			for i in range(size):
				pop_piece_values = {'p' : random.uniform(-10,0),'P' :random.uniform(0,10), 'n' : random.uniform(-10,0),'N':random.uniform(0,10),
								'q': random.uniform(-10,0),'Q': random.uniform(0,10),'b': random.uniform(-10,0),
								'B':random.uniform(0,10),'k':0,'K': 0,'r': random.uniform(-10,0),'R':random.uniform(0,10),'-' :0}
				pop_pos_values = [[random.uniform(0,1) for x in range(8)] for y in range(8)]
				#print(pop_pos_values)
				pop_player = player(pop_piece_values,pop_pos_values)
				current_population.append(pop_player)
				#print(i)
		for member in current_population:
			member.fitness = calc_fitness(pgn,member)
		current_population.sort(key=lambda x: x.fitness, reverse=True)
		print(gen, current_population[0].fitness)
		print(gen, current_population[-1].fitness)
		current_population = current_population[:int(size/2)]
		for i in range(int(size/2)):
			for j in range(i+1,int(size/2)):
				current_population.append(crossover(current_population[i],current_population[j]))
		print("GEN : ",gen)
	return current_population


if __name__ == '__main__':
	final_population =GA(40,10)
#for member in final_population:
#	print(member.piece_values,member.pos_values,member.fitness)
#best_player = final_population[0]
#print(best_player.piece_values,best_player.pos_values,best_player.fitness)
#print(calc_fitness(pgn,first_player))
#print(calc_fitness(pgn,second_player))


