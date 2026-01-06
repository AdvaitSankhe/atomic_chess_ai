import chess
import chess.variant
import chess.svg
import random
import time
import pickle
from data import *
pgn = []
with open('my_pgn.pgn','r') as file:
  
    # reading each line   
    for line in file:
  
        # reading each word       
        for word in line.split():
  
            # displaying the words          
            #print(word)
            pgn.append(word)
pgn = pgn[6:]
#print(pgn)

def eval_position(board):
	piece_values = {
	    chess.PAWN: 100,
	    chess.ROOK: 500,
	    chess.KNIGHT: 320,
	    chess.BISHOP: 330,
	    chess.QUEEN: 900,
	    chess.KING: 200000
	}

	#board = chess.Board(chess.STARTING_FEN)
	white_material = 0
	black_material = 0

	for square in chess.SQUARES:
	    piece = board.piece_at(square)
	    if not piece:
	        continue
	    if piece.color == chess.WHITE:
	        white_material += piece_values[piece.piece_type]
	    else:
	        black_material += piece_values[piece.piece_type]
	return white_material - black_material 

final_population =GA(50,5)
best_player = final_population[0]
with open('first_crossover.txt','wb') as f:
	pickle.dump(best_player,f)

with open('first_crossover.txt','rb') as f:
	best_player = pickle.load(f)


print(best_player.fitness,best_player.piece_values,best_player.pos_values)
def new_eval_position(board,best_player):
	board_matrix = fen_to_board(str(board.fen))
	value = 0
	piece_values = best_player.piece_values
	pos_values = best_player.pos_values
	for i in range(len(board_matrix)):
		for j in range(len(board_matrix[i])):
			value += piece_values[board_matrix[i][j]]*pos_values[i][j]
	return value


def pgn_to_notation(pgn):
	move_list = []
	for i in range(len(pgn)):
		if i%3 == 0:
			continue
		move_list.append(pgn[i])
	return move_list


board = chess.variant.AtomicBoard()
move_list = pgn_to_notation(pgn)
#board_set = []
counts = 0
'''
def new_minimax(board,depth,maximizing_player,max_depth,flag):
	global counts
	counts += 1
	value = eval_position(board)
	if(value < -10000):
		#print(depth,value,maximizing_player)
		return value
	if depth == 0 :	
		return eval_position(board)
	if maximizing_player:
		ideal_move = null
		value = -float('inf')
		my_list = ["depth = " + str(depth)]
		old_val = value
		for move in board.legal_moves:
			board.push(move)
			x = new_minimax(board,depth-1,False,max_depth,flag)
			value = max(value, x)
			#if flag and x !=0:
				#print(move,x,depth,"WHITE")
			flag = False
			board.pop()
			if value > old_val or ideal_move == null:
				ideal_move = move
				old_val = value
			#if(depth == max_depth):
				#print(move,x)
				#print("IDEAL MOVE IS ",ideal_move)
		#print(my_list)
		if(depth == max_depth):
			#print("WHITE", value)
			return str(ideal_move)
	else:
		value = float('inf')
		old_val = value
		my_list = ["depth = " + str(depth)]
		for move in board.legal_moves:
			board.push(move)
			x = new_minimax(board,depth-1,True,max_depth,flag)
			value = min(value,x)
			my_list.append(value)
			#if flag and x !=0:
				#print(move,x,depth, "BLACK")
			board.pop()
			if value < old_val:
				ideal_move = move
				old_val = value
		#print(my_list)
		if depth == max_depth:
			#print("BLACK", value)
			return str(ideal_move)
	return value
'''
visited_boards = dict()
def abpruning(board,depth,maximizing_player,max_depth,alpha,beta,movelist):
	global counts
	global visited_boards
	
	if depth != max_depth and (str(board),depth,maximizing_player) in visited_boards:
		#print('trigger')
		return visited_boards[(str(board),depth,maximizing_player)]
	
	counts += 1
	if board.is_variant_win():
		if maximizing_player:
			return 1000000 - depth
		else:
			return -1000000 + depth
	#value = eval_position(board)
	value = new_eval_position(board,best_player)
	#if(value < -10000):
		#print(depth,value,maximizing_player)
		#print(movelist, depth)
	#	return value
	if depth == 0 :	
		return value
	if maximizing_player:
		value = -float('inf')
		moves = []
		my_list = ["depth = " + str(depth)]
		old_val = value
		for move in board.legal_moves:
			#if ideal_moves == []:
			#	ideal_moves = [move]
			board.push(move)
			current_val = abpruning(board,depth-1,False,max_depth,alpha,beta,movelist + ' ' + str(move))
			value = max(value, current_val)
			'''
			if depth == max_depth:
				if value == old_val:
					ideal_moves.append([move,value])
				if value > old_val:
					ideal_moves = [[move,value]]
					old_val = value
			'''
			moves.append([move,current_val])
			#print(value,alpha)
			
			if beta < value:
				#print("BROKE HERE MAX" ,movelist, alpha, value)
				board.pop()
				break
			alpha = max(alpha,value)
			#if flag and x !=0:
				#print(move,x,depth,"WHITE")
			#flag = False
			board.pop()
			#if(depth == max_depth):
				#print(move,x)
				#print("IDEAL MOVE IS ",ideal_move)
		#print(my_list)
		if(depth == max_depth):
			visited_boards = dict()
			#print("WHITE", value)
			#print(moves)
			ideal_moves = []
			#ideal_value = max(x[1] for x in moves)
			for x in moves:
				if x[1] == value:
					ideal_moves.append(x)
			ideal_move = random.choice(ideal_moves)
			return (str(ideal_move[0]),ideal_move[1])
	else:
		value = float('inf')
		old_val = value
		my_list = ["depth = " + str(depth)]
		moves = []
		for move in board.legal_moves:
			board.push(move)
			#if ideal_moves == []:
			#	ideal_moves = [move]
			current_val = abpruning(board,depth-1,True,max_depth,alpha,beta, movelist + ' ' + str(move))
			value = min(value,current_val)
			moves.append([move,current_val])
			'''
			if depth == max_depth:
				if value == old_val:
					ideal_moves.append(move)
				if value < old_val:
					ideal_moves = [move]
					old_val = value
			'''

			if value < alpha:
				#print("BROKE HERE MINNN", movelist + ' ' + str(move), alpha,value)
				board.pop()
				break
			beta = min(beta,value)
			#my_list.append(value)
			#if flag and x !=0:
				#print(move,x,depth, "BLACK")
			board.pop()
		#print(my_list)
		if depth == max_depth:
			visited_boards = dict()
			#print("BLACK", value)
			ideal_moves = []
			for x in moves:
				if x[1] == value:
					ideal_moves.append(x)
			ideal_move = random.choice(ideal_moves)
			return (str(ideal_move[0]),ideal_move[1])
	visited_boards[(str(board),depth,maximizing_player)] = value
	#print("hey,",visited_boards)
	return value

'''
def minimax(board, depth, maximizing_player,movelist,max_depth):
    #print(board.is_game_over())
    global counts
    counts += 1
    value = eval_position(board)
    if board.is_game_over():
    	return -10001
    if depth == 0 or value < -10000:
        #value = eval_position(board)S
        #if True or value != 0:
        	#my_img = chess.svg.board(board)
        	#f = open("moves/" + movelist + str(value)  + ".svg", "w")
        	#f.write(my_img)
        return value
    if maximizing_player:
        value = -float('inf')
        old_val = value
        for move in board.legal_moves:
            board.push(move)
            #if(board in board_set):
            #	continue
            
            value = max(value, minimax(board, depth - 1, False,movelist+str(move),max_depth))
            if value >= old_val:
            	ideal_move = move
            	old_val = value
            board.pop()
        if depth == max_depth:
            #my_img = chess.svg.board(board)
            #f = open("moves/" + movelist + str(value) + ".svg", "w")
            #f.write(my_img)
            #rint("here" + str(move))
            print("WHITE",value)
            return str(ideal_move)
            
    else:
        value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            #if board in board_set:
            #	continue
            old_val = value
            value = min(value, minimax(board, depth - 1, True,movelist + str(move),max_depth))
            if value <= old_val:
            	ideal_move = move

            board.pop()
        if depth == max_depth:
          	#rint("HEERE" + str(move))
            print("BLACK",value)
            return str(ideal_move)
            
    return value
'''
def make_moves(move_list,board):
	for idx,move in enumerate(move_list):
		move = ''.join(ch for ch in move if not ch.isupper())
		move = chess.Move.from_uci(move)
		board.push(move)
		eval_position(board)
		my_img = chess.svg.board(board)
		f = open("moves/my_img" + str(idx) + ".svg", "w")
		f.write(my_img)
	return board

movelist = ["a2a4","d7d5","b2b4","e7e5","c2c4","c8f5","d2d4","d8e7","e2e4","h7h5"]

def minimax_moves(board):
	max_depth = 5
	i = 0
	total_time = 0
	MINIMUM,MAXIMUM = -float('inf'), float('inf')
	global counts
	while(i<300):
		#print(i)
		start_time = time.time()
		if(i%2 == 1):
			move = input()
			#move,value = abpruning(board,max_depth,False,max_depth,MINIMUM,MAXIMUM,'')
		else:
		    move,value = abpruning(board,max_depth,True,max_depth,MINIMUM,MAXIMUM,'')
		print("--- %s seconds ---" % (time.time() - start_time))
		total_time += float(time.time()) - float(start_time)
		print(move, value, i )
		#print(board.fen)
		print(counts)
		#chess_move = chess.Move.from_uci(move)
		#board.push(chess_move)
		
		
		try:
			chess_move = chess.Move.from_uci(move)
			board.push(chess_move)
			#print("SCUEES")
		except:
			print("WRONG MOVE!\n")
			board.pop()
			continue


		#rint(i)
		#rint(board)
		
		my_img = chess.svg.board(board)
		f= open("moves/" + str(i) + '.svg', "w")
		f.write(my_img)
		f.close()
		counts = 0
		i += 1
		print(total_time/i)


#board = make_moves(movelist,board)
minimax_moves(board)
#print(minimax(board,4,True,''))
#board = make_moves(move_list,board)


#21.937 ==>15