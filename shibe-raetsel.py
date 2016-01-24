#!/usr/bin/env python3

# Imports
import sys
import math
import pyglet  # INSTALL
import pyglet.gl
from pyglet.window import key
from queue import Queue, PriorityQueue  # ,LifoQueue
import random
from timeit import default_timer as timer
import numpy as np

import cProfile


# ---- globals

# will be initialized elsewhere
searches = []
curSearch = None
heuristics = []
curHeur = None
keys = {}

# Flags
flag_debug = False
flag_hint = False

# Performance Data
heuristic_calls = 0

# ui
font_large = 32
font_small = 13
font_tile = 20

window = pyglet.window.Window(resizable=True, caption='15-Puzzle')
maxdimension = min(window.width, window.height)

bgimg = None
# bgimg = pyglet.resource.image('data/img/img.png')

pyglet.gl.glClearColor(0.1, 0.1, 0.1, 1)


# Objects of class Puzzle represent a singe game
class Puzzle(object):

    # Initialize with dimension of game
    def __init__(self, dimX, dimY):
        self.dim = (dimX, dimY)

        self.initstate = list(range(self.dim[0] * self.dim[1]))[1:]
        self.initstate.append(0)

        self.twisted = False

        self.reset()

        self.solve('')

        self.columnlist = []
        for x in range(self.dim[0]):
            for y in range(self.dim[1]):
                self.columnlist.append([])
                self.columnlist[x].append(self.boardcopy()[y*self.dim[0] + x])

        self.rowlist = []
        for y in range(self.dim[1]):
            for x in range(self.dim[0]):
                self.rowlist.append([])
                self.rowlist[y].append(self.boardcopy()[y*self.dim[0] + x])

    def twistmoves(self):
        self.twisted = not self.twisted

    def debugsolution(self):
        for element in self.solution:
            print(element)

    def heuristic(self, _heuristic=curHeur):
        return _heuristic.run(self.state(), self.dim)

    def debugheuristic(self):
        print("Heuristic cost: " + str(self.heuristic()))

    def calchint(self):
        if not self.solved:
            if self.solution is not '':
                hints = ['←', '↑', '↓', '→']
                if puzzle.twisted:
                    hints.reverse()

                self.hint = hints[int(self.solution[0])]
            else:
                # calc a new hint
                # puzzle.solve(puzzle.search(searches[2],
                #                           heuristics[len(heuristics) - 1],
                #                           False))
                self.hint = None
        else:
            self.hint = None
        return None

    # Set the game to solved state
    def reset(self):
        self.board = self.initcopy()

        self.solved = True
        self.solvable = True
        self.solve('')

    def solve(self, solution):
        if isinstance(solution, tuple):
            # convert to String
            self.update(solution[1], _sol=solution[0])
            # we are done
        elif isinstance(solution, str):
            self.solution = solution
        else:
            raise(ValueError("The solution in solve() must be str or tuple"))

        self.calchint()

        return None

    # If there is a solution go one step further
    # doesnt work with ida solutions yet
    def step(self):
        if isinstance(self.solution, tuple):
            self.solution = self.solution[0]
            self.step()
        elif isinstance(self.solution, str):
            if len(self.solution) < 1:
                return None  # we are done here

            move = self.solution[0]
            rest = self.solution[1:]
            # TODO try-catch to filter out illegal moves
            self.update(getNeighborStates(
                                self.board,
                                self.dim)[int(move)], _sol=rest)

        if self.solution == '':
            if not self.solved:
                # the solution was wrong
                print("the solution was wrong")

    # return a copy of solved game state
    def initcopy(self):
        return self.initstate[:]

    # return a copy of actual game state
    def boardcopy(self):
        return self.board[:]

    # return index of given number in game
    def index(self, element):
        return self.board.index(element) % self.dim[0],\
               self.board.index(element) // self.dim[0]

    # return element at given coords
    def tile(self, x, y):
        return self.board[y*self.dim[0] + x]

    # set the game state and check for solvability
    def update(self, newfield, _paritycheck=True, _sol=''):
        self.board = newfield[:]
        if _paritycheck:
            self.checkparity()
        self.checksolved()

        self.solve(_sol)

    # ...
    def randomize(self, _bound=0, _heuristic=None):
        iter_max = 10000
        if _bound <= 0:
            iter_max = 1

        min_heur = 2147483647
        min_board = None

        for i in range(iter_max):
            board = self.boardcopy()
            self.solvable = False
            while not self.solvable:
                random.shuffle(board)
                self.update(board)

            heur = _heuristic.run(self.state(), self.dim)
            if heur < min_heur:
                min_heur = heur
                min_board = board
                if heur < _bound:
                    break

        self.update(min_board)

    def checksolved(self):
        self.solved = self.board == self.initcopy()

    # is the puzzle solvable
    def checkparity(self):
        inversions = 0
        for num in self.board:
            if num == 0:
                continue
            for other in self.board:
                if other == 0:
                    continue

                numpos = self.board.index(num)
                otherpos = self.board.index(other)
                if (numpos < otherpos) and (other < num):
                    inversions += 1

        if (self.dim[0] % 2) != 0:
            self.solvable = (inversions % 2) == 0
        else:
            if ((self.dim[1] - self.index(0)[1]) % 2) != 0:
                self.solvable = (inversions % 2) == 0
            else:
                self.solvable = (inversions % 2) != 0

        return None

    # swap the empty tile with left neighbor
    def move(self, direction):
        neighbors = list(getNeighborStates(self.board, self.dim))

        if self.twisted:
            direction = [0, 1, 2, 3][::-1][direction]

        new = neighbors[direction]

        if new is None:
            if flag_debug:
                print("This move is not possible (" + str(direction) + ")")
            return None

        if self.solution != '' and str(direction) == self.solution[0]:
            self.update(new, _paritycheck=False, _sol=self.solution[1:])
        else:
            self.update(new, _paritycheck=False)
            # The user moved in other dir than hint

        return None

    def search(self, searchObject, heuristicObject, _debug=False):
        start = self.boardcopy()
        goal = self.initcopy()

        return searchObject.run(start, goal, self.dim,
                                heuristicObject, _debug, False)

    def state(self):
        return '', self.boardcopy()


class Search(object):
    def __init__(self, name, _frontier=None):
        self.name = name
        self.frontier = _frontier  # ida is None,
        return None

    def run(self, start, goal, dim, _heuristic=None, _debug=False,
            _profile=False):
        _profile = _debug  # for later

        if _heuristic is None:
            _heuristic = Heuristic("Zero", lambda p, d: 0)

        print("\nSearching with " + self.name +
              "\n    Heuristic function: " + _heuristic.name +
              "\n    Debug is " + str(_debug))

        solution = ('', [])

        if _profile:
            solution = self.runProfile(start, goal, dim, _heuristic)
        else:
            heurf = _heuristic.function
            frontier = self.frontier

            tstart = timer()

            if frontier is None:  # this is an ID search
                solution = idaSearch(start, goal, heurf, False)
                solution = self.pathConv(solution, dim)
            else:                  # this is a normal search
                solution = genericSearch(start, goal, heurf, frontier, False)

            tend = timer()
            elapsed_time = tend - tstart

            print("\n" + self.name + " is complete." +
                  "\n    It took " + str(elapsed_time) + "s." +
                  "\n    Solution has " + str(len(solution[0])) + " steps.")

        return solution

    def runProfile(self, start, goal, dim, heuristic):
        heurf = heuristic.function
        frontier = self.frontier
        solution = ('', [])

        ref = [None]  # cProfile: need to pass a mutable object
        if frontier is None:      # this is an ID search
            cProfile.runctx('ref[0] = idaSearch(start, goal, heurf, True)',
                            globals(), locals())
            solution = self.pathConv(ref[0], dim)
        else:              # this is a normal search
            cProfile.runctx('ref[0] = genericSearch(start, goal, heurf,' +
                            'frontier, True)', globals(), locals())
            solution = ref[0]

        print("\n" + self.name + " is complete." +
              "\n    Solution has " + str(len(solution[0])) + " steps.")

        return solution

    # converts a given path in ida format
    # to a string of moves plus start
    def pathConv(self, path, dim):
        #genpath = ""
        #idapath = path[:]
        #start = idapath[0]

        #while len(idapath) > 1:
        #    possible = getNeighborStates(idapath.pop(0), dim)
        #    current = idapath[0]
        #    for i in range(4):
        #        if current == possible[i]:
        #            genpath = genpath + str(i)

        #return genpath, start
        return path[0], path[-1]


class Heuristic(object):
    def __init__(self, name, function):
        self.name = name
        self.function = function
        return None

    def run(self, state, dim):
        return self.function(state, dim)

# ######################## heuristic functions

# highly used function!
#
# for a given path, calc the heuristic costs
#
def hCostInvertDistance(path, dim):
    length = dim[0]*dim[1]
    state = path[-1]
    # Use of numpy to reformat state as columns
    #array = np.asarray(state)
    #matrix = array.reshape(dim[1],dim[0])
    #matrixT = matrix.transpose()
    #matrixL = matrixT.reshape(length)
    #cols = np.reshape(np.transpose(np.reshape(np.asarray(state),dim[1],dim[0])),length)
    #print(cols)
    #print(matrix)
    #print(matrixT)
    #print(matrixL)
    vertical = 0
    horizontal = 0
    inversions = 0
    length = dim[0]*dim[1]
    # calc number of vertical moves
    for i in range(length):
        ival = state[i]
        if ival == 0: continue
        for other in range(length):
            otherval = state[other]
            if otherval == 0: continue
            if (i < other) and (otherval < ival):
                #print(i,other,ival,otherval)
                inversions += 1
    vertical = inversions // 3 + inversions % 3
    inversions = 0
    # calc number of horizontal moves
    for x in range(dim[0]):
        for y in range(dim[1]):
            xy = x*dim[1]+y
            i = y*dim[0]+x
            ival = state[i]
            #print(xy, ival)
            #print("---")
            if ival == 0: continue
            for xo in range(dim[0]):
                for yo in range(dim[1]):
                    xoyo = xo*dim[1]+yo
                    o = yo*dim[0]+xo
                    otherval = state[o]
                    #print(xoyo, otherval)
                    if otherval == 0: continue
                    if (xy < xoyo) and (otherval < ival):
                    #if (xy < xoyo) and (otherval < ival):
                        #print("inv")
                        inversions += 1
    horizontal = inversions // 3 + inversions % 3

    #return inversions
    return vertical + horizontal



# ######################## heuristic functions

# highly used function!
#
# for a given path, calc the heuristic costs
#
def hCostLinearConflict(path, dim):
    state = path[-1]
    cost = 0

    for row in range(dim[1]):
        rowtimesdimzero = row * dim[0]
        for col in range(dim[0]):
            index = rowtimesdimzero + col
            num = state[index]
            if num == 0: continue
            should_row, should_col = divmod(num - 1, dim[0])
            if should_col == col:  # col num should
                for i in range(row):
                    pre = state[index - (i+1) * dim[0]]
                    if pre < num: continue # pre != 0 is checked implicitly
                    if (pre-1) % dim[0] != col: continue  # col pre should
                    cost += 2

            if should_row == row:  #row num should
                for x in range(rowtimesdimzero, index):
                    pre = state[x]
                    if pre < num: continue # pre != 0 is checked implicitly
                    if (pre-1) // dim[0] != row: continue  # row pre should
                    cost += 2

            if should_row > row: cost += should_row - row
            else: cost += row - should_row

            if should_col > col: cost += should_col - col
            else: cost += col - should_col
    return cost

def toString(state):
    hashint = 0
    for y in range(dim[1]):
        for x in range(dim[0]):
            state[i]
    return hashint

# highly used function!
#
# for a given path, calc the heuristic costs
#
def hCostManhattan(path, dim):
    state = path[-1]
    cost = 0
    for row in range(dim[1]):
        for col in range(dim[0]):
            num = state[row * dim[0] + col]
            if num is 0: continue
            should_row, should_col = divmod(num-1, dim[0])

            if should_row > row: cost += should_row - row
            else: cost += row - should_row

            if should_col > col: cost += should_col - col
            else: cost += col - should_col
    return cost


# highly used function!
#
# for a given path, calc the heuristic costs
# Just for fun, calc manhattan times 3
def hCostMhtn3x(path, dim):
    return hCostManhattan(path, dim) * 3


# highly used function!
#
# for a given path, calc the heuristic costs
# Just for fun, calc manhattan times 3
def hCostMhtn2x(path, dim):
    return hCostManhattan(path, dim) * 2


# highly used function!
#
# for a given path, calc the heuristic costs
# Just for fun, calc manhattan times 3
def hCostMhtn1_5x(path, dim):
    return int(hCostManhattan(path, dim) * 1.5)


# highly used function!
#
# for a given path, calc the heuristic costs
# heuristic function: Toorac = tiles out of row and column
def hCostToorac(path, dim):
    state = path[-1]
    cost = 0
    for row in range(dim[1]):
        for col in range(dim[0]):
            num = state[row * dim[0] + col]
            if num is 0: continue
            should_row = (num-1) // dim[0]
            should_col = (num-1) % dim[0]
            if row != should_row:
                cost += 1
            if col != should_col:
                cost += 1
    return cost


# highly used function!
#
# for a given path, calc the heuristic costs
# heuristic funktion: Mpt = Misplaced Tiles
def hCostMpt(path, dim):
    state = path[-1]
    cost = 0
    for i, num in enumerate(state):
        exp = i + 1
        if exp !=  num and exp != 16:
            cost += 1
    return cost


# highly used function!
#
# for a element give coords (in state)
def getStatePosition(state, dim, element):
    index = state.index(element)
    return divmod(index, dim[0])


# highly used function!
#
# for a given state give possible next states
def getNeighborStates(state, dim):
    # precalc
    izero = state.index(0)
    izero_fdiv = izero // dim[0]
    izero_mod = izero % dim[0]

    # left:
    iswap = izero - 1
    if izero_fdiv == iswap // dim[0]:
        left = state[:]
        left[izero] = left[iswap]
        left[iswap] = 0
    else:
        left = None

    # up:
    iswap = izero + dim[0]
    if izero_mod == iswap % dim[0] and iswap < len(state):
        up = state[:]
        up[izero] = up[iswap]
        up[iswap] = 0
    else:
        up = None

    # down:
    iswap = izero - dim[0]
    if izero_mod == iswap % dim[0] and iswap >= 0:
        down = state[:]
        down[izero] = down[iswap]
        down[iswap] = 0
    else:
        down = None

    # right:
    iswap = izero + 1
    if izero_fdiv == iswap // dim[0]:
        right = state[:]
        right[izero] = right[iswap]
        right[iswap] = 0
    else:
        right = None

    return (left, up, down, right)


# Do a search without ID
def genericSearch(start_pos, end_state, _heurf=lambda p, d: 0,
                  _dataStructure=Queue, _debug=False):

    visited = set()
    frontier = _dataStructure()
    # heap = []

    max_frontier = 0

    global heuristic_calls
    heuristic_calls = 0

    item = (_heurf(('', start_pos), puzzle.dim), 1, (' ', start_pos))
    frontier.put(item)

    while not frontier.empty():
        max_frontier = max(frontier.qsize(), max_frontier)

        # hcosts, plength, path = heappop(heap)
        hcosts, plength, path = frontier.get()
        plength += 1

        head = path[1]

        if str(head) in visited:
            # omitted += 1
            continue
        else:
            visited.add(str(head))

            if head == end_state:
                return (path[0][1:], start_pos)

            if _debug and len(visited) % 10000 == 0:
                print("----------\n" +
                      "Heur. calls:   " + str(heuristic_calls) + "\n" +
                      "Visited nodes: " + str(len(visited)) + "\n" +
                      "Max. frontier: " + str(max_frontier) + "\n" +
                      "Cur Distance:  " + str(hcosts) + " | " +
                      str(hcosts-plength+1) + "h, " + str(plength - 1) + "p")

            left, up, down, right = getNeighborStates(head, puzzle.dim)

            if left is not None and path[0][-1] != '3' and str(left) not in visited:
                new_path = (path[0] + '0', left)
                frontier.put((_heurf(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            if up is not None and path[0][-1] != '2' and str(up) not in visited:
                new_path = (path[0] + '1', up)
                frontier.put((_heurf(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            if down is not None and path[0][-1] != '1' and str(down) not in visited:
                new_path = (path[0] + '2', down)
                frontier.put((_heurf(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            if right is not None and path[0][-1] != '0' and str(right) not in visited:
                new_path = (path[0] + '3', right)
                frontier.put((_heurf(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            # frontier elements: (hcost+plength, plength, (string, state))

    return None


# Do a search with IDA
def idaSearch(startPos, endPos, heurf,
              _dataStructure=Queue, _debug=False):
    #bound = heurf([startPos], puzzle.dim)

    global global_added_nodes

    global_added_nodes = 0
    
    # for increasing bound by 2 you need to find the right start bound
    # that is 1 the MD of the blank tile to its final position is odd, 0 else
    y, x = getStatePosition(startPos,puzzle.dim,0)
    dist = abs(x-puzzle.dim[0]) + abs(y-puzzle.dim[1])
    bound = (dist%2)
    if True:
    #if _debug:
        tstart = timer()
        prev_elapsed = 0

    while True:
        path = idaIteration(["x",startPos], 1, bound, endPos, heurf)

        if path is not None:
            #remove x in moves before returning
            moves = path[0]
            node = path[-1]
            return [moves[1:],startPos]

        if True:
        #if _debug:
            tnow = timer()
            elapsed_time = tnow - tstart
            diff = elapsed_time - prev_elapsed
            prev_elapsed = elapsed_time
            print("Iteration " + str(bound) + " done in " + str(elapsed_time) +
                  " (cumulated)" + " add.: " + str(diff))
        bound += 2


# Used by IDA to search until a given bound
def idaIteration(path, lenpath, bound, endPos, heur):
    global global_added_nodes
    visited = set()
    visited.add(str(path[-1]))
    visited_dict = {}
    visited_dict[str(path[-1])] = 0
    frontier = []
    frontier.append(path)
    added_nodes = 0
    startTime = timer()
    start = timer()
    stop = timer()
    while frontier:
        #print(frontier)
        path = frontier.pop()
        #print(path)
        moves = path[0]
        node = path[-1]
        if True:
        #if str(node) not in visited:
            #visited.add(str(node))
            if node == endPos:
                print("Visited: " + str(len(visited)))
                return path
            #print(moves)

            # moves includes start-symbol x, therefore subtract 1
            movelen = len(moves) - 1

            left, up, down, right = getNeighborStates(node, puzzle.dim)

            estlen = 0
            string = ""
            if left is not None and moves[-1] != '3':
                estlen = movelen + heur([left], puzzle.dim)
                if estlen <= bound:
                    #continue
                    string = str(left)
                    if string not in visited or visited_dict[string] > estlen:
                        added_nodes += 1
                        global_added_nodes += 1
                        if added_nodes % 10000 == 0:
                            stop = timer()
                            delta = (stop - start) / 1000
                            deltaSecs = (stop - start)
                            start = timer()
                            # Set True to enable debugging
                            if True:
                                print("\nCurrent State: ")
                                print(right)
                                print("\nPath: ")
                                print(moves)
                                print("\nLength Path: ", movelen)
                                print("Bound: ", bound)
                                print("Heuristic: ", heur([left], puzzle.dim))
                                print("Length+Heuristic: ", estlen)
                                print("Added nodes: ", added_nodes)
                                print("Global added nodes: ", global_added_nodes)
                                print("Closed nodes: ", len(visited))
                                #print("Ommited Nodes: ", omitted_nodes)
                                print("Stack Length: ", len(frontier))
                                #print("Overall computed Heuristics: ", heuristic_calls)
                                print('')
                                print("Used Time: {}s".format(deltaSecs))      
                        #if string in visited:
                        #    if visited_dict[string] > estlen:
                        #        print("update")
                        visited_dict[string] = estlen
                        #print("ol", visited_dict[string])
                        #print(visited_dict)
                        visited.add(string)
                        frontier.append((moves + '0', left))

            if up is not None and moves[-1] != '2':
                estlen = movelen + heur([up], puzzle.dim)
                if estlen <= bound:
                    #print("fuckup")
                    #continue
                    string = str(up)
                    if string not in visited or visited_dict[string] > estlen:
                        added_nodes += 1
                        global_added_nodes += 1
                        if added_nodes % 10000 == 0:
                            stop = timer()
                            delta = (stop - start) / 1000
                            deltaSecs = (stop - start)
                            start = timer()
                            # Set True to enable debugging
                            if True:
                                print("\nCurrent State: ")
                                print(right)
                                print("\nPath: ")
                                print(moves)
                                print("\nLength Path: ", movelen)
                                print("Bound: ", bound)
                                print("Heuristic: ", heur([up], puzzle.dim))
                                print("Length+Heuristic: ", estlen)
                                print("Added nodes: ", added_nodes)
                                print("Global added nodes: ", global_added_nodes)
                                print("Closed nodes: ", len(visited))
                                #print("Ommited Nodes: ", omitted_nodes)
                                print("Stack Length: ", len(frontier))
                                #print("Overall computed Heuristics: ", heuristic_calls)
                                print('')
                                print("Used Time: {}s".format(deltaSecs))      
                        visited_dict[string] = estlen
                        visited.add(string)
                        frontier.append((moves + '1', up))

            if down is not None and moves[-1] != '1':
                estlen = movelen + heur([down], puzzle.dim)
                if estlen <= bound:
                    #print("fuckdown")
                    #continue
                    string = str(down)
                    if string not in visited or visited_dict[string] > estlen:
                        added_nodes += 1
                        global_added_nodes += 1
                        if added_nodes % 10000 == 0:
                            stop = timer()
                            delta = (stop - start) / 1000
                            deltaSecs = (stop - start)
                            start = timer()
                            # Set True to enable debugging
                            if True:
                                print("\nCurrent State: ")
                                print(right)
                                print("\nPath: ")
                                print(moves)
                                print("\nLength Path: ", movelen)
                                print("Bound: ", bound)
                                print("Heuristic: ", heur([down], puzzle.dim))
                                print("Length+Heuristic: ", estlen)
                                print("Added nodes: ", added_nodes)
                                print("Global added nodes: ", global_added_nodes)
                                print("Closed nodes: ", len(visited))
                                #print("Ommited Nodes: ", omitted_nodes)
                                print("Stack Length: ", len(frontier))
                                #print("Overall computed Heuristics: ", heuristic_calls)
                                print('')
                                print("Used Time: {}s".format(deltaSecs))     
                        visited_dict[string] =  estlen
                        visited.add(string)
                        frontier.append((moves + '2', down))

            if right is not None and moves[-1] != '0':
                estlen = movelen + heur([right], puzzle.dim)
                if estlen <= bound:
                    #print("fuckright")
                    #continue
                    string = str(right)
                    if string not in visited or visited_dict[string] > estlen:
                        added_nodes += 1
                        global_added_nodes += 1
                        if added_nodes % 10000 == 0:
                            stop = timer()
                            delta = (stop - start) / 1000
                            deltaSecs = (stop - start)
                            start = timer()
                            # Set True to enable debugging
                            if True:
                                print("\nCurrent State: ")
                                print(right)
                                print("\nPath: ")
                                print(moves)
                                print("\nLength Path: ", movelen)
                                print("Bound: ", bound)
                                print("Heuristic: ", heur([right], puzzle.dim))
                                print("Length+Heuristic: ", estlen)
                                print("Added nodes: ", added_nodes)
                                print("Global added nodes: ", global_added_nodes)
                                print("Closed nodes: ", len(visited))
                                #print("Ommited Nodes: ", omitted_nodes)
                                print("Stack Length: ", len(frontier))
                                #print("Overall computed Heuristics: ", heuristic_calls)
                                print('')
                                print("Used Time: {}s".format(deltaSecs))      
                        visited_dict[string] = estlen
                        visited.add(string)
                        frontier.append((moves + '3', right))

            
            # frontier elements: (moves, state)


    print("Visited: " + str(len(visited)))
    return None


@window.event
def on_resize(width, height):
    global maxdimension
    maxdimension = min(width, height)
    if bgimg is not None:
        bgimg.width = maxdimension
        bgimg.height = maxdimension


@window.event
def on_draw():
    global puzzle
    global curHeur

    # ---- Background respond to solved state
    if puzzle.solved:
        pyglet.gl.glClearColor(0.1, 0.3, 0.1, 1)
    else:
        pyglet.gl.glClearColor(0.1, 0.1, 0.1, 1)

    offsetx, offsety = ((window.width-maxdimension)/2,
                        (window.height-maxdimension)/2)
    window.clear()

    # ---- Use background image
    if bgimg is not None:
        bgimg.blit(offsetx, offsety)

    # ---- Draw puzzle
    for y in range(puzzle.dim[1]):
        for x in range(puzzle.dim[0]):
            tile = str(puzzle.tile(x, y))
            size = font_tile
            if tile == '0':
                tile = '⋅'
                size = int(size * 2)
                if flag_hint and puzzle.hint is not None:
                    tile = str(puzzle.hint)

            number = pyglet.text.Label(tile, font_size=size,
                x=offsetx+(x+1)*(maxdimension/(puzzle.dim[0]+1)),
                y=window.height-offsety-(y+1)*(maxdimension/(puzzle.dim[1]+1)),
                anchor_x='center', anchor_y='center')
            number.draw()

    # ---- Construct labels
    top = window.height - font_large
    labels = [("Current heuristic function: ", 10, top)]
    for h in heuristics:
        prefix = " "
        if h is curHeur:
            prefix = "*"
        y = top - len(labels) * round(1.5 * font_small)
        text = prefix + " " + str(puzzle.heuristic(h)) + ' ' + h.name

        labels.append((text, 16, y))

    right = window.width - 180
    labels.append(("Hint: " + str(flag_hint), right, top))
    labels.append(("Debug: " + str(flag_debug), right, top - 1.5*font_small))
    labels.append(("Solution: " + str(len(puzzle.solution)) + " steps",
                   right, top - 4.5*font_small))

    # ---- Draw controls
    x = line = round(1.5*font_small)
    for char, desc, func in list(keys.values())[::-1]:
        if char is not None:
            labels.append((char+' - '+desc, 20, x))
            x += line

    labels.append(("Controls:", 10, x))

    font = 'Monospace'
    for text, posx, posy in labels:
        pyglet.text.Label(text, font_name=font, font_size=font_small, x=posx,
                          y=posy, anchor_x='left', anchor_y='center').draw()


@window.event
def on_key_press(symbol, modifiers):
    if symbol in keys.keys():
        keys[symbol][2]()


def toggleHeuristic():
    global curHeur, heuristics
    new_index = (heuristics.index(curHeur)+1) % len(heuristics)
    curHeur = heuristics[new_index]


def toggleDebug():
    global flag_debug
    flag_debug = not flag_debug


def toggleHint():
    global flag_hint
    flag_hint = not flag_hint

def convertPuzzle(state):
    for i in range(len(state)):
        if state[i] == "":
            state[i] = 0
        else:
            state[i] += 1
    return state

def main():
    global puzzle, searches, curSearch, heuristics, curHeur, keys

    if len(sys.argv) == 2:
        board = []
        given = sys.argv[1].replace(' ', '').split(',')
        for tile in given:
            tile = int(tile)
            if tile in board or tile < 0 or tile > len(given):
                print("Error reading input!")
                sys.exit(0)
            board.append(tile)
        if 0 not in board:
            print("Error reading input! no zero")
            sys.exit(0)
        y = int(math.sqrt(len(board)))
        while y > 0:
            if len(board) % y != 0:
                y -= 1
            else:
                break
        puzzle = Puzzle(len(board)//y, y)
        puzzle.update(board)
    elif len(sys.argv) == 1:
        puzzle = Puzzle(4, 4)
    else:
        print("Unable to parse given data")

    heuristics = [Heuristic("Manhattan distance", hCostManhattan),
                  Heuristic("Misplaced tiles", hCostMpt),
                  Heuristic("Tiles out of row & column", hCostToorac),
                  Heuristic("Linear conflicts", hCostLinearConflict),
                  Heuristic("Manhattan * 1.5", hCostMhtn1_5x),
                  Heuristic("Manhattan * 2", hCostMhtn2x),
                  Heuristic("Manhattan * 3", hCostMhtn3x),
                  Heuristic("Invert Distance", hCostInvertDistance)]
    curHeur = heuristics[0]

    searches = [Search("BFS", Queue),
                Search("A*", PriorityQueue),
                Search("IDA*", None)]
    curSearch = searches[0]

    #testpuzzle = [10, 2,  5,  4, 0,  11, 13, 8, 3,  7,  6,  12, 14, 1,  9,  15]
    testpuzzle = convertPuzzle([12, 4, 2, 7, 6, 1, 9, 3, 14, 5, 8, '', 0, 10, 11, 13])
    #61 steps
    #testpuzzle = [14,12,15,13,6,1,8,9,10,11,4,7,0,2,5,3]
    print(testpuzzle)

    keys = {
        key.B:     ('b',  "search BFS",
                    lambda: puzzle.solve(puzzle.search(searches[0], curHeur, flag_debug))),
        key.A:     ('a',  "search A*",
                    lambda: puzzle.solve(puzzle.search(searches[1], curHeur, flag_debug))),
        key.I:     ('i',  "search IDA*",
                    lambda: puzzle.solve(puzzle.search(searches[2], curHeur, flag_debug))),
        key.SPACE: ('␣',  "step through solution", lambda: puzzle.step()),
        key.ENTER: ('↲',  "reset puzzle", lambda: puzzle.reset()),
        key.E:     ('e',  "change heuristic", lambda: toggleHeuristic()),
        key.H:     ('h',  "show/hide hint", lambda: toggleHint()),
        key.R:     ('r',  "random puzzle", lambda: puzzle.randomize(0, curHeur)),
        key.T:     (None, "(random) puzzle", lambda: puzzle.randomize(20, curHeur)),
        key.LEFT:  (None, "move left", lambda: puzzle.move(3)),
        key.UP:    (None, "move up", lambda: puzzle.move(1)),
        key.DOWN:  (None, "move down", lambda: puzzle.move(2)),
        key.RIGHT: (None, "move right", lambda: puzzle.move(0)),
        key.Y:     (None, "change directions", lambda: puzzle.twistmoves()),
        key.P:     (None, "print current solution", lambda: puzzle.debugsolution()),
        key.X:     (None, "toggle Debug", lambda: toggleDebug()),
        key.Q:     (None, "", lambda: puzzle.update(testpuzzle)),
        #key.Q:     (None, "", lambda: puzzle.update([0, 15, 14,  13,
        #                                             12, 11, 10, 9,
        #                                             8,  7,  6,  5,
        #                                             4, 3,  2,  1])),
        # deprecated
        key.C:     (None, "deprecated", lambda: puzzle.debugheuristic())}

    pyglet.app.run()


if __name__ == '__main__':
    main()
