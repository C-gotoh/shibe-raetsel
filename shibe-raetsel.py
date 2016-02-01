#!/usr/bin/env python3

# ######################## Imports
import sys
import math
import pyglet  # INSTALL
import pyglet.gl
from pyglet.window import key
from queue import Queue, PriorityQueue  # ,LifoQueue
import random
from timeit import default_timer as timer

import cProfile


# ######################## Globals

# will be initialized elsewhere
searches = []
curSearch = None
heuristics = []
curHeur = None
keys = {}

# Flags
flag_debug = False
flag_profile = False
flag_hint = False

# Performance Data
heuristic_calls = 0

# ui
font_large = 32
font_small = 11
font_tile = 20

window = pyglet.window.Window(resizable=True, caption='15-Puzzle')
maxdimension = min(window.width, window.height)

bgimg = None
# bgimg = pyglet.resource.image('data/img/img.png')

pyglet.gl.glClearColor(0.1, 0.1, 0.1, 1)


# ######################## Puzzle logic

# Objects of class Puzzle represent a singe game
class Puzzle(object):

    # Initialize with dimension of game
    def __init__(self, dimX, dimY):
        self.dim = (dimX, dimY)

        self.initstate = list(range(self.dim[0] * self.dim[1]))[1:]
        self.initstate.append(0)

        self.twisted = True

        self.reset()

        self.solve('')

        return None

    # Toggle twisted directions
    def twistmoves(self):
        self.twisted = not self.twisted

    # Print the solution
    def debugsolution(self):
        text = "Solution"
        for element in self.solution:
            if self.twisted:
                text = text + ' ' + ['←', '↓', '↑', '→'][int(element)]
            else:
                text = text + ' ' + ['→', '↑', '↓', '←'][int(element)]
        print(text)

    # Run heuristic
    def heuristic(self, _heuristic=curHeur):
        return _heuristic.run(self.state(), self.dim)

    # Update hint
    def calchint(self):
        if not self.solved:
            if self.solution is not '':
                hints = ['→', '↑', '↓', '←']
                if puzzle.twisted:
                    hints.reverse()
                self.hint = hints[int(self.solution[0])]
            else:
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

    # Provide a solution to the puzzle
    def solve(self, solution):
        if isinstance(solution, tuple):
            self.update(solution[1], _sol=solution[0])
        elif isinstance(solution, str):
            self.solution = solution
        else:
            raise(ValueError("The solution in solve() must be str or tuple"))

        self.calchint()

        return None

    # If there is a solution go one step further
    def step(self):
        if isinstance(self.solution, tuple):
            self.solution = self.solution[0]
            self.step()
        elif isinstance(self.solution, str):
            if len(self.solution) < 1:
                return None  # we are done here

            move = self.solution[0]
            rest = self.solution[1:]
            self.update(getNeighborStates(
                                self.board,
                                self.dim)[int(move)], _sol=rest)

        if self.solution == '':
            if not self.solved:
                print("the solution was wrong")

    # Return a copy of solved game state
    def initcopy(self):
        return self.initstate[:]

    # Return a copy of actual game state
    def boardcopy(self):
        return self.board[:]

    # Return index of given number in game
    def index(self, element):
        return self.board.index(element) % self.dim[0],\
               self.board.index(element) // self.dim[0]

    # Return element at given coords
    def tile(self, x, y):
        return self.board[y*self.dim[0] + x]

    # Set the game state and check for solvability
    def update(self, newfield, _paritycheck=True, _sol=''):
        self.board = newfield[:]
        if _paritycheck:
            self.checkparity()
        self.checksolved()

        self.solve(_sol)

    # Randomize puzzle (with a heuristic bound)
    def random(self, _bound=0, _heuristic=None):
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

    # Is the puzzle solved?
    def checksolved(self):
        self.solved = self.board == self.initcopy()

    # Is the puzzle solvable?
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

    # Swap the empty tile with neighbor
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

        return None

    # Run a search on puzzle
    def search(self, searchObject, heuristicObject,
               _debug=False, _profile=False):
        start = self.boardcopy()
        goal = self.initcopy()

        return searchObject.run(start, goal, self.dim,
                                heuristicObject, _debug, _profile)

    # Return a state with no solution
    def state(self):
        return '', self.boardcopy()


# ######################## Heuristic class

class Heuristic(object):

    # Initialize with name and function
    def __init__(self, name, function):
        self.name = name
        self.function = function
        return None

    # Calc heuristic cost
    def run(self, state, dim):
        return self.function(state, dim)


# ######################## Search class

class Search(object):

    # Initialize with name and data structure
    def __init__(self, name, _frontier=None):
        self.name = name
        self.frontier = _frontier

        return None

    # Run this search
    def run(self, start, goal, dim, _heuristic=None, _debug=False,
            _profile=False):
        if _heuristic is None:
            _heuristic = Heuristic("Zero", lambda p, d: 0)

        print("\nSearching with " + self.name +
              "\n    Heuristic function: " + _heuristic.name +
              "\n    Debug is " + str(_debug))

        solution = ('', [])

        if _profile:
            solution = self.runProfile(start, goal, dim, _heuristic, _debug)
        else:
            heurf = _heuristic.function
            frontier = self.frontier

            tstart = timer()

            if frontier is None:  # this is an ID search
                solution = idaSearch(start, goal, heurf, _debug=_debug)
                solution = (solution[0], solution[-1])
            else:                  # this is a normal search
                solution = genericSearch(start, goal, heurf,
                                         frontier, _debug=_debug)

            tend = timer()
            elapsed_time = tend - tstart

            print("\n" + self.name + " is complete." +
                  "\n    It took " + str(elapsed_time) + "s." +
                  "\n    Solution has " + str(len(solution[0])) + " steps.")

        return solution

    # Run search with cProfile
    def runProfile(self, start, goal, dim, heuristic, debug):
        heurf = heuristic.function
        frontier = self.frontier
        solution = ('', [])

        ref = [None]  # cProfile: need to pass a mutable object
        if frontier is None:      # this is an ID search
            cProfile.runctx('ref[0] = idaSearch(start, goal, heurf, debug)',
                            globals(), locals())
            solution = (ref[0][0], ref[0][-1])
        else:              # this is a normal search
            cProfile.runctx('ref[0] = genericSearch(start, goal, heurf,' +
                            'frontier, debug)', globals(), locals())
            solution = ref[0]

        print("\n" + self.name + " is complete." +
              "\n    Solution has " + str(len(solution[0])) + " steps.")

        return solution


# ######################## Heuristic functions

# highly used function!
#
# for a given path, calc the heuristic costs
# heuristic function: Toorac = tiles out of row and column
def hCostToorac(path, dim, _oldheur=0):
    state = path[-1]
    cost = 0
    for row in range(dim[1]):
        for col in range(dim[0]):
            num = state[row * dim[0] + col]
            if num is 0:
                continue
            should_row, should_col = divmod((num-1), dim[0])
            if row != should_row:
                cost += 1
            if col != should_col:
                cost += 1
    return cost


# highly used function!
#
# for a given path, calc the heuristic costs
# heuristic funktion: Mpt = Misplaced Tiles
def hCostMpt(path, dim, _oldheur=0):
    state = path[-1]
    cost = 0
    for i, num in enumerate(state):
        exp = i + 1
        if exp != num and exp != 16:
            cost += 1
    return cost


# highly used function!
#
# for a given path, calc the heuristic costs
# heuristic funktion: Manhattan Distance
def hCostManhattan(path, dim, _oldheur=0):
    state = path[-1]
    if _oldheur == 0 or len(path[0]) == 0:
        cost = 0
        for row in range(dim[1]):
            for col in range(dim[0]):
                num = state[row * dim[0] + col]
                if num is 0:
                    continue
                should_row, should_col = divmod(num-1, dim[0])

                if should_row > row:
                    cost += should_row - row
                else:
                    cost += row - should_row

                if should_col > col:
                    cost += should_col - col
                else:
                    cost += col - should_col
        return cost
    lastmove = int(path[0][-1])
    izero = state.index(0)

    swap_was_row, swap_was_col = divmod(izero, dim[0])
    swap_is_row = swap_was_row
    swap_is_col = swap_was_col

    if lastmove == 0:  # left
        iswap = izero + 1
        swap_is_col += 1

    elif lastmove == 1:  # up
        iswap = izero - dim[0]
        swap_is_row -= 1

    elif lastmove == 2:  # down
        iswap = izero + dim[0]
        swap_is_row += 1

    elif lastmove == 3:  # right
        iswap = izero - 1
        swap_is_col -= 1

    swap = state[iswap]

    swap_should_row, swap_should_col = divmod(swap-1, dim[0])

    swap_was_impact = abs(swap_should_row - swap_was_row) +\
        abs(swap_should_col - swap_was_col)
    swap_is_impact = abs(swap_should_row - swap_is_row) +\
        abs(swap_should_col - swap_is_col)

    cost = _oldheur - swap_was_impact + swap_is_impact
    return cost


# highly used function!
#
# for a given path, calc the heuristic costs
# heuristic funktion: LC = Linear Conflicts
def hCostLinearConflict(path, dim, _oldheur=0):
    state = path[-1]
    cost = 0

    for row in range(dim[1]):
        rowtimesdimzero = row * dim[0]
        for col in range(dim[0]):
            index = rowtimesdimzero + col
            num = state[index]
            if num == 0:
                continue
            should_row, should_col = divmod(num - 1, dim[0])
            if should_col == col:  # col num should
                for i in range(row):
                    pre = state[index - (i+1) * dim[0]]
                    if pre < num:
                        continue  # pre != 0 is checked implicitly
                    if (pre-1) % dim[0] != col:
                        continue  # col pre should
                    cost += 2

            if should_row == row:  # row num should
                for x in range(rowtimesdimzero, index):
                    pre = state[x]
                    if pre < num:
                        continue  # pre != 0 is checked implicitly
                    if (pre-1) // dim[0] != row:
                        continue  # row pre should
                    cost += 2

            if should_row > row:
                cost += should_row - row
            else:
                cost += row - should_row

            if should_col > col:
                cost += should_col - col
            else:
                cost += col - should_col
    return cost


# highly used function!
#
# for a given path, calc the heuristic costs
# Just for fun, calc LC times 3
def hCostLC3x(path, dim, _oldheur=0):
    return hCostLinearConflict(path, dim) * 3


# highly used function!
#
# for a given path, calc the heuristic costs
# Just for fun, calc LC times 2
def hCostLC2x(path, dim, _oldheur=0):
    return hCostLinearConflict(path, dim) * 2


# highly used function!
#
# for a given path, calc the heuristic costs
# Just for fun, calc LC times 1.5
def hCostLC1_5x(path, dim, _oldheur=0):
    return int(hCostLinearConflict(path, dim) * 1.5)


# highly used function!
#
# for a given path, calc the heuristic costs
# Just for fun, calc LC times 1.1
def hCostLC1_1x(path, dim, _oldheur=0):
    return int(hCostLinearConflict(path, dim) * 1.1)


# highly used function!
#
# for a element give coords (in state)
def getStatePosition(state, dim, element):
    index = state.index(element)
    return divmod(index, dim[0])


# ######################## Additional functions for search

# highly used function!
#
# for a given state give possible next states
def getNeighborStates(state, dim):
    izero = state.index(0)
    izero_fdiv, izero_mod = divmod(izero, dim[0])

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
    if iswap < dim[0]*dim[1] and izero_mod == iswap % dim[0]:
        up = state[:]
        up[izero] = up[iswap]
        up[iswap] = 0
    else:
        up = None

    # down:
    iswap = izero - dim[0]
    if iswap >= 0 and izero_mod == iswap % dim[0]:
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


# ######################## Search functions

# Do a search without ID
def genericSearch(start_pos, end_state, _heurf=lambda p, d: 0,
                  _data_struc=Queue, _debug=False):
    visited = set()
    frontier = _data_struc()

    global heuristic_calls
    heuristic_calls = 0
    max_frontier = 0

    item = (_heurf(('', 0, start_pos), puzzle.dim), 1, (' ', start_pos))
    frontier.put(item)

    while not frontier.empty():
        max_frontier = max(frontier.qsize(), max_frontier)

        hcost, plen, path = frontier.get()
        oldhcost = hcost - plen
        plen += 1

        head = path[-1]

        if str(head) in visited:
            continue

        visited.add(str(head))

        if head == end_state:
            return (path[0][1:], start_pos)

        l, u, d, r = getNeighborStates(head, puzzle.dim)

        if l is not None and path[0][-1] != '3' and str(l) not in visited:
            new_path = (path[0] + '0', l)
            frontier.put((
                _heurf(new_path, puzzle.dim, _oldheur=oldhcost) + plen,
                plen, new_path))
            heuristic_calls += 1

        if u is not None and path[0][-1] != '2' and str(u) not in visited:
            new_path = (path[0] + '1', u)
            frontier.put((
                _heurf(new_path, puzzle.dim, _oldheur=oldhcost) + plen,
                plen, new_path))
            heuristic_calls += 1

        if d is not None and path[0][-1] != '1' and str(d) not in visited:
            new_path = (path[0] + '2', d)
            frontier.put((
                _heurf(new_path, puzzle.dim, _oldheur=oldhcost) + plen,
                plen, new_path))
            heuristic_calls += 1

        if r is not None and path[0][-1] != '0' and str(r) not in visited:
            new_path = (path[0] + '3', r)
            frontier.put((
                _heurf(new_path, puzzle.dim, _oldheur=oldhcost) + plen,
                plen, new_path))
            heuristic_calls += 1

        if _debug and len(visited) % 10000 == 0:
            print("----------\n" +
                  "Heur. calls:   " + str(heuristic_calls) + "\n" +
                  "Visited nodes: " + str(len(visited)) + "\n" +
                  "Max. frontier: " + str(max_frontier) + "\n" +
                  "Cur Distance:  " + str(hcost) + " | " +
                  str(hcost-plen+1) + "h, " + str(plen - 1) + "p")
    return None


# Do a search with IDA
def idaSearch(start_pos, end_state, heurf, _data_struc=Queue, _debug=False):
    global global_added_nodes
    global_added_nodes = 0

    # for increasing bound by 2 you need to find the right start bound
    # that is 1 the MD of the blank tile to its final position is odd, 0 else
    y, x = getStatePosition(start_pos, puzzle.dim, 0)
    dist = abs(x - puzzle.dim[0]) + abs(y - puzzle.dim[1])
    bound = (dist % 2)
    if _debug:
        tstart = timer()
        prev_elapsed = 0

    while True:
        path = idaIteration(["x", start_pos], bound, end_state, heurf, _debug)

        if path is not None:
            return [path[0][1:], start_pos]

        if _debug:
            tnow = timer()
            elapsed_time = tnow - tstart
            diff = elapsed_time - prev_elapsed
            prev_elapsed = elapsed_time
            print("Iteration " + str(bound) + " done in " + str(elapsed_time) +
                  " (cumulated)" + " add.: " + str(diff))
        bound += 2


# Used by IDA to search until a given bound
def idaIteration(path, bound, end_state, heur, debug):
    global global_added_nodes

    visited = set()
    visited.add(str(path[-1]))
    visited_dict = {}
    visited_dict[str(path[-1])] = 0
    frontier = []
    frontier.append(path)

    added_nodes = 0

    start = timer()
    stop = timer()
    while frontier:
        path = frontier.pop()
        moves = path[0]
        node = path[-1]
        if node == end_state:
            if debug:
                print("Visited: " + str(len(visited)))
            return path

        # moves includes start-symbol x, therefore subtract 1
        movelen = len(moves) - 1

        if debug:
            current_added = added_nodes
            print("Visited: " + str(len(visited)))

        l, u, d, r = getNeighborStates(node, puzzle.dim)

        estlen = 0
        string = ""
        if l is not None and moves[-1] != '3':
            estlen = movelen + heur([l], puzzle.dim)
            if estlen <= bound:
                string = str(l)
                if string not in visited or visited_dict[string] > estlen:
                    global_added_nodes = added_nodes = added_nodes + 1
                    visited_dict[string] = estlen
                    visited.add(string)
                    frontier.append((moves + '0', l))

        if u is not None and moves[-1] != '2':
            estlen = movelen + heur([u], puzzle.dim)
            if estlen <= bound:
                string = str(u)
                if string not in visited or visited_dict[string] > estlen:
                    global_added_nodes = added_nodes = added_nodes + 1
                    visited_dict[string] = estlen
                    visited.add(string)
                    frontier.append((moves + '1', u))

        if d is not None and moves[-1] != '1':
            estlen = movelen + heur([d], puzzle.dim)
            if estlen <= bound:
                string = str(d)
                if string not in visited or visited_dict[string] > estlen:
                    global_added_nodes = added_nodes = added_nodes + 1
                    visited_dict[string] = estlen
                    visited.add(string)
                    frontier.append((moves + '2', d))

        if r is not None and moves[-1] != '0':
            estlen = movelen + heur([r], puzzle.dim)
            if estlen <= bound:
                string = str(r)
                if string not in visited or visited_dict[string] > estlen:
                    global_added_nodes = added_nodes = added_nodes + 1
                    visited_dict[string] = estlen
                    visited.add(string)
                    frontier.append((moves + '3', r))

        if debug and current_added//10000 != added_nodes//10000:
            stop = timer()
            deltaSecs = (stop - start)
            start = timer()
            print("\nCurrent State: ")
            print(node)
            print("\nPath: ")
            print(moves)
            print("\nLength Path: ", movelen)
            print("Bound: ", bound)
            print("Heuristic: ", heur(path, puzzle.dim))
            print("Length+Heuristic: ", estlen)
            print("Added nodes: ", added_nodes)
            print("Global added nodes: ", global_added_nodes)
            print("Closed nodes: ", len(visited))
            print("Stack Length: ", len(frontier))
            print('')
            print("Used Time: {}s".format(deltaSecs))
    return None


# ######################## GUI

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
                size = int(size * 2)
                if flag_hint and puzzle.hint is not None:
                    tile = str(puzzle.hint)
                else:
                    tile = '⋅'

            number = pyglet.text.Label(
                tile, font_size=size,
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
    labels.append(("Profile: " + str(flag_profile), right, top - 3*font_small))
    labels.append(("Solution: " + str(len(puzzle.solution)) + " steps",
                   right, top - 6*font_small))

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


def toggleProfile():
    global flag_profile
    flag_profile = not flag_profile


def toggleHint():
    global flag_hint
    flag_hint = not flag_hint


# ######################## Main function

def main():
    global puzzle, searches, curSearch, heuristics, curHeur, keys

    if len(sys.argv) == 1:
        puzzle = Puzzle(4, 4)
    elif len(sys.argv) == 2:
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
    else:
        print("Unable to parse given data")

    heuristics = [Heuristic("Misplaced Tiles", hCostMpt),
                  Heuristic("Tiles out of row & column", hCostToorac),
                  Heuristic("Manhattan Distance", hCostManhattan),
                  Heuristic("Linear Conflicts", hCostLinearConflict),
                  Heuristic("LC * 1.1", hCostLC1_1x),
                  Heuristic("LC * 1.5", hCostLC1_5x),
                  Heuristic("LC * 2", hCostLC2x),
                  Heuristic("LC * 3", hCostLC3x)]
    curHeur = heuristics[0]

    searches = [Search("BFS", Queue),
                Search("A*", PriorityQueue),
                Search("IDA*", None)]
    curSearch = searches[0]

    keys = {
        key.B:     ('b', "search BFS", lambda:
                    puzzle.solve(puzzle.search(searches[0], curHeur,
                                 _debug=flag_debug, _profile=flag_profile))),
        key.A:     ('a', "search A*", lambda:
                    puzzle.solve(puzzle.search(searches[1], curHeur,
                                 _debug=flag_debug, _profile=flag_profile))),
        key.I:     ('i', "search IDA*", lambda:
                    puzzle.solve(puzzle.search(searches[2], curHeur,
                                 _debug=flag_debug, _profile=flag_profile))),
        key.SPACE: ('␣', "step through solution", lambda: puzzle.step()),
        key.ENTER: ('↲', "reset puzzle", lambda: puzzle.reset()),
        key.E:     ('e', "change heur", lambda: toggleHeuristic()),
        key.H:     ('h', "toggle hint", lambda: toggleHint()),
        key.R:     ('r', "random", lambda: puzzle.random(0, curHeur)),
        key.T:     ('t', "random (limit)", lambda: puzzle.random(20, curHeur)),
        key.Y:     ('y', "switch key directions", lambda: puzzle.twistmoves()),
        key.X:     ('x', "toggle debug", lambda: toggleDebug()),
        key.C:     ('c', "toggle profile", lambda: toggleProfile()),
        key.P:     ('p', "print solution", lambda: puzzle.debugsolution()),
        key.LEFT:  (None, "move left", lambda: puzzle.move(3)),
        key.UP:    (None, "move up", lambda: puzzle.move(1)),
        key.DOWN:  (None, "move down", lambda: puzzle.move(2)),
        key.RIGHT: (None, "move right", lambda: puzzle.move(0))}

    pyglet.app.run()

if __name__ == '__main__':
    main()
