#!/usr/bin/env python3

# Imports
import pyglet  # INSTALL
import pyglet.gl
from pyglet.window import key
from queue import Queue, PriorityQueue  # ,LifoQueue
import random
from timeit import default_timer as timer

import cProfile


# ---- globals

# will be initialized elsewhere
searches = []
curSearch = None
heuristics = []
curHeur = None
keys = {}

# be a little bit verbose
debug = False

# Performance Data
added_nodes = 0
closed_nodes = 0
omitted_nodes = 0
heuristic_calls = 0

# ui
font_large = 32
font_small = 13
font_number = 20

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
            if self.solution is '':
                return None
            #    solution = puzzle.runIDA(heur=hCostMhtn2x)

            hints = ["left", "up", "down", "right"]
            if puzzle.twisted:
                hints.reverse()

            self.hint = hints[int(self.solution[0])]
        else:
            self.hint = None

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
            # self.solution = pathConv(solution, self.dim)[0]

        self.calchint()

        return None

    # If there is a solution,
    # go one strep throu it
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
        else:
            # this case is for old solution types
            self.update(self.solution.pop(0))

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
            self.checkboard()
        else:
            self.checksolved()

        self.solve(_sol)

    # ...
    def randomize(self, _bound=0, _heuristic=None):
        iter_max = 10000
        while iter_max > 0:
            board = self.boardcopy()
            random.shuffle(board)

            self.update(board)

            if self.solvable:
                if _bound != 0 and\
                   _heuristic.run(self.state(), self.dim) > _bound:
                    iter_max -= 1
                    continue
                break

        self.update(board)

    def checkboard(self):
        self.checksolved()
        self.checkparity()

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
        if direction not in [0, 1, 2, 3]:
            raise(ValueError("move() must be called with 0, 1, 2 or 3"))

        new = list(getNeighborStates(self.board, self.dim))

        if self.twisted:
            new.reverse()

        new = new[direction]

        if new is None:
            if debug:
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
        genpath = ""
        idapath = path[:]
        start = idapath[0]

        while len(idapath) > 1:
            possible = getNeighborStates(idapath.pop(0), dim)
            current = idapath[0]
            for i in range(4):
                if current == possible[i]:
                    genpath = genpath + str(i)

        return genpath, start


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
def hCostLinearConflict(path, dim):
    state = path[-1]
    cost = 0
    xtimesy = dim[0] * dim[1]
    for y in range(dim[1]):
        maxValue = 0
        for x in range(dim[0]):
            expectednumber = y * dim[0] + x + 1
            if expectednumber == xtimesy:  # expectednumber = 0
                continue
            # for rows
            value = state[y*dim[0]+x]
            if value in puzzle.rowlist[y] and value != 0:
                if value >= maxValue:
                    maxValue = value
                else:
                    cost += 2

            actualposition = getStatePosition(state, dim, expectednumber)
            manhattanDist = abs(x - actualposition[0])\
                + abs(y - actualposition[1])
            cost += manhattanDist

        # for cols
        maxValue = 0
        for i in range(dim[1]):
            value = state[y*dim[0]+x]
            if value in puzzle.columnlist[i]:
                if value == 0:
                    continue
                if value >= maxValue:
                    maxValue = value
                else:
                    cost += 2
    return cost


# highly used function!
#
# for a given path, calc the heuristic costs
#
def hCostManhattan(path, dim):
    state = path[-1]
    cost = 0
    xtimesy = dim[0] * dim[1]
    for y in range(dim[1]):
        for x in range(dim[0]):
            expectednumber = y * dim[0] + x + 1
            if expectednumber == xtimesy:
                # expectednumber = 0
                continue
            actualposition = getStatePosition(state, dim, expectednumber)
            manhattanDist = abs(x - actualposition[0])\
                + abs(y - actualposition[1])
            cost += manhattanDist
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
    for y in range(dim[1]):
        row_start = y*dim[0]
        for x in range(dim[0]):
            expectedNumber = x + row_start + 1
            if expectedNumber == dim[0]*dim[1]:
                continue
            if expectedNumber not in state[row_start:row_start+dim[1]]:
                cost += 1
            # uses global puzzle
            # TODO dont
            if expectedNumber not in puzzle.columnlist[x]:
                cost += 1
    return cost


# highly used function!
#
# for a given path, calc the heuristic costs
# heuristic funktion: Mpt = Misplaced Tiles
def hCostMpt(path, dim):
    state = path[-1]
    cost = 0
    for x in range(dim[0]):
        for y in range(dim[1]):
            expectedNumber = x + y * dim[0] + 1
            if expectedNumber == dim[0]*dim[1]:
                continue
            actualnumber = state[y * dim[0] + x]
            if expectedNumber != actualnumber:
                cost += 1
    return cost  # + len(path)


# highly used function!
#
# for a element give coords (in state)
def getStatePosition(state, dim, element):
    index = state.index(element)
    return index % dim[0], index // dim[0]


# highly used function!
#
# for a given state give possible next states
def getNeighborStates(state, dim):
    # precalc
    izero = state.index(0)
    izero_fdiv = izero // dim[0]
    izero_mod = izero % dim[1]

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
    if izero_mod == iswap % dim[1] and iswap < len(state):
        up = state[:]
        up[izero] = up[iswap]
        up[iswap] = 0
    else:
        up = None

    # down:
    iswap = izero - dim[0]
    if izero_mod == iswap % dim[1] and iswap >= 0:
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

    item = (_heurf(('', start_pos), puzzle.dim), 1, ('', start_pos))

    frontier.put(item)

    # while True:
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
                return (path[0], start_pos)

            if _debug and len(visited) % 10000 == 0:
                print("----------\n" +
                      "Heur. calls:   " + str(heuristic_calls) + "\n" +
                      "Visited nodes: " + str(len(visited)) + "\n" +
                      "Max. frontier: " + str(max_frontier) + "\n" +
                      "Cur Distance:  " + str(hcosts) + " | " +
                      str(hcosts-plength+1) + "h, " + str(plength - 1) + "p")

            left, up, down, right = getNeighborStates(head, puzzle.dim)

            if left is not None and str(left) not in visited:
                new_path = (path[0] + "0", left)
                frontier.put((_heurf(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            if up is not None and str(up) not in visited:
                new_path = (path[0] + "1", up)
                frontier.put((_heurf(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            if down is not None and str(down) not in visited:
                new_path = (path[0] + "2", down)
                frontier.put((_heurf(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            if right is not None and str(right) not in visited:
                new_path = (path[0] + "3", right)
                frontier.put((_heurf(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            # frontier elements: (hcost+plength, plength, (string, state))

    return None


# Do a search with IDA
def idaSearch(startPos, endPos, heurf,
              _dataStructure=Queue, _debug=False):
    bound = heurf([startPos], puzzle.dim)

    if _debug:
        tstart = timer()
        prev_elapsed = 0

    while True:
        path = idaIteration([startPos], 1, bound, endPos, heurf)

        if path is not None:
            return path

        if _debug:
            tnow = timer()
            elapsed_time = tnow - tstart
            diff = elapsed_time - prev_elapsed
            prev_elapsed = elapsed_time
            print("Iteration " + str(bound) + " done in " + str(elapsed_time) +
                  " (cumulated)" + " add.: " + str(diff))
        bound += 1


# Used by IDA to search until a given bound
def idaIteration(path, lenpath, bound, endPos, heur):
    visited = set()
    frontier = []
    frontier.append(path)
    while frontier:
        path = frontier.pop()
        node = path[-1]
        if str(node) not in visited:
            visited.add(str(node))
            if node == endPos:
                return path
            for neighbor in getNeighborStates(node, puzzle.dim):
                if neighbor is None or str(neighbor) in visited:
                    continue
                estlen = len(path) + heur([neighbor], puzzle.dim)
                if estlen > bound:
                    continue
                new_path = path[:]
                new_path.append(neighbor)
                frontier.append(new_path)

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
            if tile == '0':
                tile = '⋅'

            number = pyglet.text.Label(
                tile,
                font_size=font_number,
                x=offsetx+(x+1)*(maxdimension/(puzzle.dim[0]+1)),
                y=offsety+(y+1)*(maxdimension/(puzzle.dim[1]+1)),
                anchor_x='center',
                anchor_y='center')
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

    right = window.width - 130
    labels.append(("Hint: " + str(puzzle.hint), right, top))
    labels.append(("Debug: " + str(debug), right, top - font_large))

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
    keys[symbol][2]()


def toggleHeuristic():
    global curHeur, heuristics
    new_index = (heuristics.index(curHeur)+1) % len(heuristics)
    curHeur = heuristics[new_index]


def toggleDebug():
    global debug
    debug = not debug


def main():
    global puzzle, searches, curSearch, heuristics, curHeur, keys

    puzzle = Puzzle(4, 4)

    heuristics = [Heuristic("Manhattan distance", hCostManhattan),
                  Heuristic("Misplaced tiles", hCostMpt),
                  Heuristic("Tiles out of row & column", hCostToorac),
                  Heuristic("Linear conflicts", hCostLinearConflict),
                  Heuristic("Manhattan * 3", hCostMhtn3x),
                  Heuristic("Manhattan * 2", hCostMhtn2x),
                  Heuristic("Manhattan * 1.5", hCostMhtn1_5x)]
    curHeur = heuristics[0]

    searches = [Search("BFS", Queue),
                Search("A*", PriorityQueue),
                Search("IDA*", None)]
    curSearch = searches[0]

    keys = {
        key.B:     ('b',  "search BFS",
                    lambda: puzzle.solve(puzzle.search(searches[0], curHeur, debug))),
        key.A:     ('a',  "search A*",
                    lambda: puzzle.solve(puzzle.search(searches[1], curHeur, debug))),
        key.I:     ('i',  "search IDA*",
                    lambda: puzzle.solve(puzzle.search(searches[1], curHeur, debug))),
        key.SPACE: ('␣',  "step through solution", lambda: puzzle.step()),
        key.ENTER: ('↲',  "reset puzzle", lambda: puzzle.reset()),
        key.R:     ('r',  "random puzzle", lambda: puzzle.randomize()),
        key.E:     ('e',  "change heuristic", lambda: toggleHeuristic()),
        key.T:     (None, "(random) puzzle", lambda: puzzle.randomize(20, curHeur)),
        key.LEFT:  (None, "move left", lambda: puzzle.move(0)),
        key.UP:    (None, "move up", lambda: puzzle.move(1)),
        key.DOWN:  (None, "move down", lambda: puzzle.move(2)),
        key.RIGHT: (None, "move right", lambda: puzzle.move(3)),
        key.Y:     (None, "change directions", lambda: puzzle.twistmoves()),
        key.P:     (None, "print current solution", lambda: puzzle.debugsolution()),
        key.X:     (None, "toggle Debug", lambda: toggleDebug()),
        key.Q:     (None, "", lambda: puzzle.update([10, 2,  5,  4,
                                                     0,  11, 13, 8,
                                                     3,  7,  6,  12,
                                                     14, 1,  9,  15])),
        # deprecated
        key.C:     (None, "deprecated", lambda: puzzle.debugheuristic()),
        key.H:     (None, "deprecated", lambda: puzzle.calchint())}

    pyglet.app.run()


if __name__ == '__main__':
    main()
