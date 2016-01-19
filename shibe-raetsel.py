#!/usr/bin/env python3

# Imports
import pyglet  # INSTALL
import pyglet.gl
from pyglet.window import key
from queue import Queue, PriorityQueue  # ,LifoQueue
import random
from timeit import default_timer as timer
import datetime

# from heapq import heappop, heappush
# import cProfile

# will be initialized elsewhere
heuristics = []
global curHeur
curHeur = None

# be a little bit verbose
debug = False

# pyglet init
window = pyglet.window.Window(resizable=True, caption='15-Puzzle')

maxdimension = min(window.width, window.height)

bgimg = None
# bgimg = pyglet.resource.image('data/img/img.png')

pyglet.gl.glClearColor(0.1, 0.1, 0.1, 1)

# Performance Data
added_nodes = 0
closed_nodes = 0
omitted_nodes = 0
heuristic_calls = 0

# ui
font_large = 32
font_small = 14
font_number = 20


# Objects of class Puzzle represent a singe game
class Puzzle(object):

    # Initialize with dimension of game
    def __init__(self, dimX, dimY):
        self.dim = (dimX, dimY)

        self.initstate = list(range(self.dim[0] * self.dim[1]))[1:]
        self.initstate.append(0)

        self.reset()

        self.solution = ('', self.boardcopy())

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

        self.twisted = False
        self.hint = ''

    def twistmoves(self):
        self.twisted = not self.twisted

    def debugsolution(self):
        for element in puzzle.solution:
            print(element)

    def calchint(self):
        global curHeur
        tmp = curHeur 
        curHeur = hCostMhtn2x
        if not self.solved:
            solution = puzzle.runAStar()

            if solution is not None and len(solution[0]) != 0:
                print(solution)
                self.hint = solution[0][0]
                hints = ["left", "up", "down", "right"]

                if puzzle.twisted:
                    hints.reverse()

                self.hint = hints[int(solution[0][0])]
        curHeur = tmp

    def sethint(self, new_hint):
        self.hint = new_hint

    # Set the game to solved state
    def reset(self):
        self.board = self.initcopy()

        self.solved = True
        self.solvable = True

        self.solution = ('', self.boardcopy())

    #
    def solve(self, solution):
        self.solution = solution

    # If there is a solution,
    # go one strep throu it
    def step(self):
        if self.solution is None or len(self.solution) == 1 or\
                self.solution[0] == '':
            self.solution = ('', self.boardcopy())
        else:
            if not isinstance(self.solution, tuple):
                self.update(self.solution.pop(0))
            else:
                self.solution = (self.solution[0][1:],
                                 getNeighborStates(
                                    self.solution[1],
                                    puzzle.dim)[int(self.solution[0][0])])
                self.board = self.solution[1]

        self.solved = self.board == self.initcopy()

        return None

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
    def update(self, newfield):
        self.board = newfield[:]

        self.solved = self.board == self.initcopy()
        if self.solved:
            self.solvable = True
        else:
            self.checkparity()

        self.solution = ('', self.boardcopy())

    # ...
    def randomize(self, _heur=lambda p, d: 0, _bound=0):
        iter_max = 10000
        while iter_max > 0:
            random.shuffle(self.board)
            self.solved = self.board == self.initstate
            self.checkparity()
            if self.solvable:
                if _bound != 0 and\
                   _heur(('', self.boardcopy()), self.dim) > _bound:
                    iter_max -= 1
                    continue
                break

        self.solution = ('', self.boardcopy())

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
    def moveleft(self):
        self.solution = ('', self.boardcopy())

        new = list(getNeighborStates(self.board, self.dim))
        if self.twisted:
            new.reverse()
        new = new[0]
        if new is not None:
            self.board = new
        self.solved = self.board == self.initcopy()

        return None

    # swap the empty tile with up neighbor
    def moveup(self):
        self.solution = ('', self.boardcopy())

        new = list(getNeighborStates(self.board, self.dim))
        if self.twisted:
            new.reverse()
        new = new[1]
        if new is not None:
            self.board = new
        self.solved = self.board == self.initcopy()

        return None

    # swap the empty tile with down neighbor
    def movedown(self):
        self.solution = ('', self.boardcopy())

        new = list(getNeighborStates(self.board, self.dim))
        if self.twisted:
            new.reverse()
        new = new[2]
        if new is not None:
            self.board = new
        self.solved = self.board == self.initcopy()

        return None

    # swap the empty tile with right neighbor
    def moderight(self):
        self.solution = ('', self.boardcopy())

        new = list(getNeighborStates(self.board, self.dim))
        if self.twisted:
            new.reverse()
        new = new[3]
        if new is not None:
            self.board = new
        self.solved = self.board == self.initcopy()

        return None

    # start a BFS on the current state of game
    def runBFS(self, _debug=False):
        start = self.boardcopy()
        goal = self.initcopy()

        print("searching (BFS)")

        tstart = timer()

        solution, a, b = genericSearch(start, goal, None,
                                       _dataStructure=Queue,
                                       _debug=_debug)

        tend = timer()
        elapsed_time = tend - tstart

        print("BFS is complete. It took", elapsed_time, "s. Solution has ",
              len(solution[0]), " steps. Heuristic: ", str(curHeur).split()[1])

        return solution #, elapsed_time

    # start a A* on the current state of game
    def runAStar(self, _debug=False):
        global curHeur
        start = self.boardcopy()
        goal = self.initcopy()

        print("searching (A* | )")

        tstart = timer()

        solution, a, b = genericSearch(start, goal, curHeur,
                                       _dataStructure=PriorityQueue,
                                       _debug=_debug)

        tend = timer()
        elapsed_time = tend - tstart

        print("A* is complete. It took", elapsed_time, "s. Solution has ", len(solution[0]),
              " steps. Heuristic: ", str(curHeur).split()[1])

        return solution #, elapsed_time

    # start a IDA on the current state of game
    def runIDA(self, _debug=False):
        global curHeur
        start = self.boardcopy()
        goal = self.initcopy()

        print("searching (IDA*)")

        tstart = timer()

        solution = idaSearch(start, goal, curHeur, _debug=_debug)

        tend = timer()
        elapsed_time = tend - tstart
        #ida solution: [start-end] -> steps=len(solution)-1
        print("IDA is complete. It took", elapsed_time, "s. Solution has ", len(solution)-1,
              " steps. Heuristic: ", str(curHeur).split()[1])

        return solution #, elapsed_time


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
            if expectednumber == xtimesy:
                # expectednumber = 0
                continue
            # for rows
            value = state[y*dim[0]+x]
            if value in puzzle.rowlist[y] and value != 0:
                if value >= maxValue:
                    maxValue = value
                else:
                    #print("conflict rows")
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
                    #print("conflict cols")
                    cost += 2
    return cost  # + len(path)    


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
    return cost  # + len(path)


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
    return hCostManhattan(path, dim) * 1.5


# highly used function!
#
# for a given path, calc the heuristic costs
# heuristic function: Toorac = tiles out of row and column
def hCostToorac(path, dim):
    state = path[-1]
    cost = 0
    for y in range(dim[1]):
        for x in range(dim[0]):
            expectedNumber = x + y * dim[0] + 1
            if expectedNumber == dim[0]*dim[1]:
                continue
            if expectedNumber not in state[y*dim[0]:y*dim[0]+dim[1]]:
                cost += 1
                # print("wrong row: " + str(expectedNumber))
                # print(state[y])
            if expectedNumber not in puzzle.columnlist[x]:
                cost += 1
                # print("wrong col: " + str(expectedNumber))
    return cost  # + len(path)


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
def genericSearch(start_pos, end_state, _heur=lambda p, d: 0,
                  _dataStructure=Queue, _debug=False):

    visited = set()
    frontier = _dataStructure()
    # heap = []

    max_frontier = 0

    global heuristic_calls
    heuristic_calls = 0

    item = (_heur(('', start_pos), puzzle.dim), 1, ('', start_pos))

    # heappush(heap, item)
    frontier.put(item)

    start = datetime.datetime.now()

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
                print(len(visited))
                return (path[0], start_pos), len(visited), max_frontier

            if _debug and len(visited) % 10000 == 0:
                stop = datetime.datetime.now()
                delta = (stop - start).microseconds / 1000
                deltaSecs = (stop - start).seconds
                start = datetime.datetime.now()
                print("----------\n" +
                      "Heur. calls:   " + str(heuristic_calls) + "\n" +
                      "Visited nodes: " + str(len(visited)) + "\n" +
                      "Max. frontier: " + str(max_frontier) + "\n" +
                      "Cur Distance:  " + str(hcosts) + " | " +
                      str(hcosts-plength+1) + "h, " + str(plength - 1) + "p")
                print("Used Time: {}s {}ms".format(deltaSecs, delta))

            left, up, down, right = getNeighborStates(head, puzzle.dim)

            if left is not None:
                new_path = (path[0] + "0", left)
                frontier.put((_heur(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            if up is not None:
                new_path = (path[0] + "1", up)
                frontier.put((_heur(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            if down is not None:
                new_path = (path[0] + "2", down)
                frontier.put((_heur(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            if right is not None:
                new_path = (path[0] + "3", right)
                frontier.put((_heur(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            # frontier elements: (hcost+plength, plength, (string, state))

    return 0, len(visited), max_frontier


# Do a search with IDA
def idaSearch(startPos, endPos, _heur=lambda p, d: 0,
              _dataStructure=Queue, _debug=False):
    visited = set()
    bound = _heur([startPos], puzzle.dim)
    tstart = timer()
    prev_elapsed = 0
    
    while True:
        path = idaIteration([startPos], 1, bound, endPos, _heur)
        if path is not None:
            return path
        tnow = timer()
        elapsed_time = tnow - tstart
        diff = elapsed_time - prev_elapsed
        prev_elapsed = elapsed_time
        print("Iteration " + str(bound) + " done in " + str(elapsed_time) + " (cumulated)" + " add.: " + str(diff))
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
                print("Visited: " + str(len(visited)))
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
    # print('The window was resized to %dx%d' % (width, height))
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
            if puzzle.tile(x, y) is not 0:
                number = pyglet.text.Label(
                    str(puzzle.tile(x, y)),
                    font_size=font_number,
                    x=offsetx+(x+1)*(maxdimension/(puzzle.dim[0]+1)),
                    y=offsety+(y+1)*(maxdimension/(puzzle.dim[1]+1)),
                    anchor_x='center',
                    anchor_y='center')
                number.draw()

    # ---- Draw heuristic function chooser
    pyglet.text.Label("Current heuristic function: ",
                      font_name='Times New Roman',
                      font_size=font_small,
                      x=16,
                      y=window.height-font_large,
                      anchor_x='left',
                      anchor_y='center').draw()

    for i in range(len(heuristics)):
        if curHeur == heuristics[i]:
            pyglet.text.Label("* " + str(heuristics[i]).split(' ')[1],
                              font_name='Times New Roman',
                              font_size=font_small,
                              x=16,
                              y=window.height-font_large -
                              (len(heuristics)+1-i)*round(1.5*font_small),
                              anchor_x='left',
                              anchor_y='center').draw()
        else:
            pyglet.text.Label("  " + str(heuristics[i]).split(' ')[1],
                              font_name='Times New Roman',
                              font_size=font_small,
                              x=16,
                              y=window.height-font_large -
                              (len(heuristics)+1-i)*round(1.5*font_small),
                              anchor_x='left',
                              anchor_y='center').draw()

    # ---- Draw current heuristic cost
    pyglet.text.Label("Cost: " + str(curHeur(('', puzzle.boardcopy()),
                                     puzzle.dim)),
                      font_name='Times New Roman',
                      font_size=font_small,
                      x=window.width - font_large*3,
                      y=window.height-font_large,
                      anchor_x='left',
                      anchor_y='center').draw()
    # ---- Draw Hint
    pyglet.text.Label("Hint: " + str(puzzle.hint),
                      font_name='Times New Roman',
                      font_size=font_small,
                      x=window.width - font_large*3,
                      y=window.height-font_large*3,
                      anchor_x='left',
                      anchor_y='center').draw()

    # ---- Draw Debug lvl
    pyglet.text.Label("debug: " + str(debug),
                      font_name='Times New Roman',
                      font_size=font_small,
                      x=window.width - font_large*3,
                      y=window.height-font_large*6,
                      anchor_x='left',
                      anchor_y='center').draw()

    # ---- Draw controls
    controls = ["Controls: ",
                "Arrowkeys to move tiles",
                "'b' - search BFS",
                "'a' - search A*",
                "'i' - search IDA*",
                "'r' - generate random puzzle",
                "'ENTER' - reset puzzle",
                "'SPACE' - step through solution",
                "'c' - print current heuristic cost",
                "'e' - change heuristic function",
                "'h' - get a hint for next move"]

    for i in range(len(controls)):
        pyglet.text.Label(controls[i],
                          font_name='Times New Roman',
                          font_size=font_small,
                          x=16,
                          y=(len(controls)+1-i)*round(1.5*font_small),
                          anchor_x='left',
                          anchor_y='center').draw()


@window.event
def on_key_press(symbol, modifiers):
    global puzzle, curHeur, heuristics, debug

    if symbol == key.B:
        puzzle.solve(puzzle.runBFS(debug, _heur=curHeur))

    elif symbol == key.A:
        puzzle.solve(puzzle.runAStar(debug))

    elif symbol == key.X:
        debug = not debug

    elif symbol == key.ENTER:   # step to solution
        puzzle.reset()

    elif symbol == key.Q:
        #testpuzzles
        #len=61
        puzzle.update([14,12,15,13,6,1,8,9,10,11,4,7,0,2,5,3])
        #len=53
        #puzzle.update([7,15,10,6,4,9,0,3,11,12,1,5,2,14,13,8])
        #len=?
        #puzzle.update([10,2,5,4,0,11,13,8,3,7,6,12,14,1,9,15])
        puzzle.checkparity()
        print(puzzle.solvable)
        #old big
        #puzzle.update([6, 7, 14, 8, 5, 1, 15, 12, 13, 0, 10, 9, 4, 2, 3, 11])

    elif symbol == key.SPACE:
        puzzle.step()

    elif symbol == key.C:
        print("Absolute cost: " +
              str(curHeur(('', puzzle.boardcopy()), puzzle.dim)))

    elif symbol == key.H:
        puzzle.calchint()

    elif symbol == key.R:
        puzzle.randomize(_heur=curHeur)

    elif symbol == key.T:
        puzzle.randomize(_heur=curHeur, _bound=20)

    elif symbol == key.E:
        new_index = (heuristics.index(curHeur)+1) % len(heuristics)
        curHeur = heuristics[new_index]

    elif symbol == key.I:
        puzzle.solve(puzzle.runIDA())

    elif symbol == key.LEFT:
        puzzle.moveleft()

    elif symbol == key.RIGHT:
        puzzle.moderight()

    elif symbol == key.UP:
        puzzle.moveup()

    elif symbol == key.DOWN:
        puzzle.movedown()

    elif symbol == key.Y:
        puzzle.twistmoves()

    elif symbol == key.P:
        puzzle.debugsolution()

heuristics = [hCostManhattan,
              hCostMpt,
              hCostToorac,
              hCostLinearConflict,
              hCostMhtn3x,
              hCostMhtn2x,
              hCostMhtn1_5x]

# searches = [runBFS,
#             runAStar,
#             runIDA]

# keys = {
#     key.B:     (lambda: puzzle.runBFS(debug, _heur=curHeur), ""),
#     key.A:     (lambda: puzzle.runAStar(debug, _heur=curHeur), ""),
#     key.X:     (lambda: , ""),
#     key.ENTER: (lambda: puzzle.reset(), ""),
#     key.Q:     (lambda: puzzle.update([6, 7, 14, 8, 5, 1, 15, 12, 13, 0, 10, 9, 4, 2, 3, 11]), ""),
#     key.SPACE: (lambda: puzzle.step(), ""),
#     key.C:     (lambda: print("Absolute cost: " + str(curHeur(('', puzzle.boardcopy()), puzzle.dim))), ""),
#     key.H:     (lambda: puzzle.calchint(), ""),
#     key.R:     (lambda: puzzle.randomize(_heur=curHeur), ""),
#     key.T:     (lambda: puzzle.randomize(_heur=curHeur, _bound=20), ""),
#     key.E:     (lambda: , ""),
#     key.I:     (lambda: puzzle.solve(puzzle.runIDA()), ""),
#     key.LEFT:  (lambda: puzzle.moveleft(), ""),
#     key.RIGHT: (lambda: puzzle.moveright(), ""),
#     key.UP:    (lambda: puzzle.moveup(), ""),
#     key.DOWN:  (lambda: puzzle.movedown(), ""),
#     key.Y:     (lambda: puzzle.twistmoves(), ""),
#     key.P:     (lambda: puzzle.debugsolution(), "")}

# Into puzzle:
# make hint depended of solution
# toggle heuristics (key E)
# toggle debug (key X)

curHeur = heuristics[0]
if __name__ == '__main__':
    global puzzle  
    puzzle = Puzzle(4, 4)

    #global curHeur
    

    pyglet.app.run()
