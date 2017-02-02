from search import *
from random import *
import itertools
import copy
import time

from operator import add
from itertools import chain, combinations
from functools import reduce

import numpy as np
from scipy import ndimage

#221 works well 
seed(2123)

class lightsout(StateSpace):
	def __init__(self, action, gval, board, board_size, presses, count, press=None, parent = None):
		"""Initialize a lightsout search state object."""
		StateSpace.__init__(self, action, gval, parent)

		if count == -1:
			self.count = 0
			for row in range(board_size):
				for col in range(board_size):
					self.count += (1 if board[row][col] else 0)
		else:
			self.count = count
			
		if presses == -1:
			self.presses = []
			row_copy =  [False] * board_size
			for row in range(board_size):
				self.presses.append(list(row_copy))
		else:
			self.presses = presses

		self.board_size = board_size
		self.board = board
		self.press = press

	def successors(self):
		'''Return list of lightsout objects that are the successors of the current object'''
		states = []


		for row in range(self.board_size):
			for col in range(self.board_size):
				
				if (not self.presses[row][col]):
				
					new_count =  self.count
					new_board = copy.deepcopy(self.board)
					new_presses = copy.deepcopy(self.presses)
					new_presses[row][col] = True
					toggle = [(row, col), (row - 1, col), (row + 1, col ), (row, col + 1), (row, col - 1)]
					toggle = [(x, y) for x, y in toggle if x >= 0 and  x < self.board_size and y >=  0  and y <  self.board_size]
	
					for x, y in toggle:
						if new_board[x][y]:
							new_count -= 1
						else:
							new_count += 1
							
						new_board[x][y] =  not new_board[x][y]
				
					states.append(lightsout('toggle(' + str(row) +', ' +  str(col) + ' )', self.gval + 1, new_board, self.board_size, new_presses, new_count, (row, col), self))

		return states

	def hashable_state(self):
		'''Return a data item that can be used as a dictionary key to UNIQUELY represent the state.'''

		#return 	tuple([self.board[x][y] for x, y in itertools.product(range(self.board_size), range(self.board_size))])
		return 	tuple([self.presses[x][y] for x, y in itertools.product(range(self.board_size), range(self.board_size))])
		
	def print_state(self):
		#DO NOT CHANGE THIS FUNCTION---it will be used in auto marking
		#and in generating sample trace output.
		#Note that if you implement the "get" routines
		#(rushhour.get_vehicle_statuses() and rushhour.get_board_size())
		#properly, this function should work irrespective of how you represent
		#your state.
	
		print('')
		if self.press:
			print("Toggle:", str(self.press))
		else:
			print("START")
		for i in range(self.board_size):
			line = ''
			for j in range(self.board_size):
				line += '|' + str(1 if self.board[i][j] else 0)
			line += '|'
			
			print(line)

		
def heur_zero(state):
	'''Zero Heuristic use to make A* search perform uniform cost search'''
	return 0


def heur_min_moves(state):

	#return state.count / 5.
	return state.count

def lightsout_goal_fn(state):

	return state.count == 0

def make_init_state(board_size, board):


	return lightsout("START", 
					0, 
					board, 
					board_size, -1, -1)

def lights_out(n):
	
	B = [0] * (n * n)
	A = []
	for i in range(n*n):
		A.append(list(B))
	for i in range(n): 
		for j in range(n): 
			m = n*i+j 
			A[m][m] = 1
			if i > 0 : A[m][m-n] = 1
			if i < n-1 : A[m][m+n] = 1
			if j > 0 : A[m][m-1] = 1
			if j < n-1 : A[m][m+1] = 1
	return A


# Linear Algebra Solution in python taken from:
# https://github.com/pmneila/Lights-Out/blob/master/lightsout.py

class GF2(object):
	"""Galois field GF(2)."""
	
	def __init__(self, a=0):
		self.value = int(a) & 1
	
	def __add__(self, rhs):
		return GF2(self.value + GF2(rhs).value)
	
	def __mul__(self, rhs):
		return GF2(self.value * GF2(rhs).value)
	
	def __sub__(self, rhs):
		return GF2(self.value - GF2(rhs).value)
	
	def __truediv__(self, rhs):
		return GF2(self.value / GF2(rhs).value)
	
	def __repr__(self):
		return str(self.value)
	
	def __eq__(self, rhs):
		if isinstance(rhs, GF2):
			return self.value == rhs.value
		return self.value == rhs
	
	def __le__(self, rhs):
		if isinstance(rhs, GF2):
			return self.value <= rhs.value
		return self.value <= rhs
	
	def __lt__(self, rhs):
		if isinstance(rhs, GF2):
			return self.value < rhs.value
		return self.value < rhs
	
	def __int__(self):
		return self.value
	
	def __long__(self):
		return self.value
	
	
GF2array = np.vectorize(GF2)
	
def gjel(A):
	"""Gauss-Jordan elimination."""
	nulldim = 0
	for i in range(len(A)):
		pivot = A[i:,i].argmax() + i
		if A[pivot,i] == 0:
			nulldim = len(A) - i
			break
		row = A[pivot] / A[pivot,i]
		A[pivot] = A[i]
		A[i] = row
		
		for j in range(len(A)):
			if j == i:
				continue
			A[j] -= row*A[j,i]
	return A, nulldim
	
def GF2inv(A):
	"""Inversion and eigenvectors of the null-space of a GF2 matrix."""
	n = len(A)
	assert n == A.shape[1], "Matrix must be square"
	
	A = np.hstack([A, np.eye(n)])
	B, nulldim = gjel(GF2array(A))
	
	inverse = np.int_(B[-n:, -n:])
	E = B[:n, :n]
	null_vectors = []
	if nulldim > 0:
		null_vectors = E[:, -nulldim:]
		null_vectors[-nulldim:, :] = GF2array(np.eye(nulldim))
		null_vectors = np.int_(null_vectors.T)
	
	return inverse, null_vectors
	
def lightsoutbase(n):
	"""Base of the LightsOut problem of size (n,n)"""
	a = np.eye(n*n)
	a = np.reshape(a, (n*n,n,n))
	a = np.array(list(map(ndimage.binary_dilation, a)))
	return np.reshape(a, (n*n, n*n))
	
def powerset(iterable):
	"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
	
class LightsOut(object):
	"""Lights-Out solver."""
	
	def __init__(self, size=5):
		self.n = size
		self.base = lightsoutbase(self.n)
		self.invbase, self.null_vectors = GF2inv(self.base)
	
	def solve(self, b):
		b = np.asarray(b)
		assert b.shape[0] == b.shape[1] == self.n, "incompatible shape"
		
		if not self.issolvable(b):
			raise "The given setup is not solvable"
		
		# Find the base solution.
		first = np.dot(self.invbase, b.ravel()) & 1
		
		# Given a solution, we can find more valid solutions
		# adding any combination of the null vectors.
		# Find the solution with the minimum number of 1's.
		solutions = [(first + reduce(add, nvs, 0))&1 for nvs in powerset(self.null_vectors)]
		final = min(solutions, key=lambda x: x.sum())
		return np.reshape(final, (self.n,self.n))
	
	def issolvable(self, b):
		"""Determine if the given configuration is solvable.
		
		A configuration is solvable if it is orthogonal to
		the null vectors of the base.
		"""
		b = np.asarray(b)
		assert b.shape[0] == b.shape[1] == self.n, "incompatible shape"
		b = b.ravel()
		p = map(lambda x: np.dot(x,b)&1, self.null_vectors)

		return not any(p)
# Used to generate a random init state
# of a board of size "size" with "tog" lights turned on.
def make_rand_state(tog, size):

	l = list(itertools.product(range(size), range(size)))
	row =  [False] * size
	toggle = lights_out(size)
	good = False
	lo = LightsOut(size)
	
	while not good:
		
		shuffle(l)
		count = 0
		board = []

		for i in range(size):
			board.append(list(row))
		
		for x, y in l:
	
			if count < tog:
				board[x][y] = True
				count+=1
			else:
				break;
				
		if (lo.issolvable(board)):
			print(board)
			print(lo.solve(board))
			good = True
			return make_init_state(size, board)
			
s0 = make_rand_state(4, 3)		
s0.print_state()		
se = SearchEngine('best_first', 'full')
se.trace_on(1)
final = se.search(s0, lightsout_goal_fn, heur_min_moves)