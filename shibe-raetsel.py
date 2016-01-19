#!/usr/bin/env python3

# Imports
import pyglet  # INSTALL
import pyglet.gl
from pyglet.window import key
from queue import Queue, PriorityQueue  # ,LifoQueue
import random
from timeit import default_timer as timer

import cProfile

# will be initialized elsewhere
heuristics = []
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

    def heuristic(self, heuristic):
        return heuristic.run(('', self.boardcopy()), self.dim)

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
            # TODO reformat ida solutions
            self.solution = solution
            return None

        self.calchint()

        return None

    # If there is a solution,
    # go one strep throu it
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
                                self.dim)[int(move)], _sol = rest)
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
    def randomize(self, _heur=lambda p, d: 0, _bound=0):
        # BROKEN
        # TODO
        iter_max = 10000
        while iter_max > 0:
            board = self.boardcopy()
            random.shuffle(board)

            self.update(board)

            if self.solvable:
                if _bound != 0 and\
                   _heur(('', self.boardcopy()), self.dim) > _bound:
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
            print("Error wrong direction " + str(direction))
            return None

        new = list(getNeighborStates(self.board, self.dim))

        if self.twisted:
            new.reverse()

        new = new[direction]

        if new is not None:
            self.update(new, _paritycheck = False)
        else:
            print("This move is not possible (" + str(direction) + ")")

        return None

    def search(self, searchObject, heuristicObject, _debug=False):
        start = self.boardcopy()
        goal = self.initcopy()

        s_name = searchObject.name

        h_name = heuristicObject.name
        h_function = heuristicObject.function

        print("Searching with " + s_name + "\n"
              "    Heuristic function: " + h_name + "\n"+
              "    Debug is " + str(_debug))

        if _debug:
            cProfile.run('solution, time = searchObject.run(start, goal, h_function, True)')
        else:
            solution, time = searchObject.run(start, goal, h_function, False)

        print(s_name + " is complete.\n" +
              "    It took", time, "s.\n" +
              "    Solution has " + str(len(solution[0])), " steps.\n" +
              "    Heuristic: ", h_name)

        return solution

    def state(self):
        return ('', self.boardcopy()), self.dim



    # # start a BFS on the current state of game
    # def runBFS(self, _debug=False):
    #     start = self.boardcopy()
    #     goal = self.initcopy()

    #     print("searching (BFS)")

    #     tstart = timer()

    #     solution, a, b = genericSearch(start, goal, None,
    #                                    _dataStructure=Queue,
    #                                    _debug=_debug)

    #     tend = timer()
    #     elapsed_time = tend - tstart

    #     print("BFS is complete. It took", elapsed_time, "s. Solution has ",
    #           len(solution[0]), " steps. Heuristic: ", str(curHeur).split()[1])

    #     return solution

    # # start a A* on the current state of game
    # def runAStar(self, heur, _debug=False):
    #     start = self.boardcopy()
    #     goal = self.initcopy()

    #     print("searching (A* | )")

    #     tstart = timer()

    #     solution, a, b = genericSearch(start, goal, heur,
    #                                    _dataStructure=PriorityQueue,
    #                                    _debug=_debug)

    #     tend = timer()
    #     elapsed_time = tend - tstart

    #     print("A* is complete. It took", elapsed_time, "s. Solution has ",
    #           len(solution[0]), " steps. Heuristic: ", str(curHeur).split()[1])

    #     return solution

    # # start a IDA on the current state of game
    # def runIDA(self, heur, _debug=False):
    #     start = self.boardcopy()
    #     goal = self.initcopy()

    #     print("searching (IDA*)")

    #     tstart = timer()

    #     solution = idaSearch(start, goal, heur, _debug=_debug)

    #     tend = timer()
    #     elapsed_time = tend - tstart

    #     print("IDA is complete. It took", elapsed_time, "s. Solution has ", len(solution),
    #           " steps. Heuristic: ", str(curHeur).split()[1])

    #     return solution


class Search(object):
    def __init__(self, name, _frontier=None):
        self.name = name
        self.frontier = _frontier # ida is None,
        return None

    def run(self, start, goal, f_heur, _debug=False):
        dataStruc = self.frontier

        if self.frontier is None: # this is an ID search
            tstart = timer()
            solution = idaSearch(start, goal, f_heur,
                                 _debug=_debug)
            tend = timer()
        else:                # this is a normal search
            tstart = timer()
            solution = genericSearch(start, goal, f_heur, dataStruc,
                                     _debug=_debug)
            tend = timer()

        elapsed_time = tend - tstart

        return solution, elapsed_time


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
    if isinstance(_heur, Heuristic):
        print(_heur.name)
    heuristic = _heur

    visited = set()
    frontier = _dataStructure()
    # heap = []

    max_frontier = 0

    global heuristic_calls
    heuristic_calls = 0

    item = (heuristic(('', start_pos), puzzle.dim), 1, ('', start_pos))

    # heappush(heap, item)
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

    return None


# Do a search with IDA
def idaSearch(startPos, endPos, _heur=lambda p, d: 0,
              _dataStructure=Queue, _debug=False):
    heuristic = _heur

    bound = heuristic([startPos], puzzle.dim)

    tstart = timer()

    while True:
        path = idaIteration([startPos], 1, bound, endPos, _heur)

        if path is not None:
            return path

        tnow = timer()
        elapsed_time = tnow - tstart

        print("Iteration " + str(bound) + " done in " + str(elapsed_time) + " (cumulated)")

        bound += 1


# Used by IDA to search until a given bound
def idaIteration(path, lenpath, bound, endPos, heur):
    visited = set()
    frontier = []
    frontier.append(path)
    while frontier:
        path = frontier.pop()
        node = path[-1]
        estlen = len(path) + heur([node], puzzle.dim)
        if str(node) not in visited:
            visited.add(str(node))
            if estlen > bound:
                continue
            if node == endPos:
                return path
            for neighbor in getNeighborStates(node, puzzle.dim):
                if neighbor is None:
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
            tile = puzzle.tile(x, y)
            if tile is 0:
                tile = "⋅"
            else:
                tile = str(tile)

            number = pyglet.text.Label(
                tile,
                font_size=font_number,
                x=offsetx+(x+1)*(maxdimension/(puzzle.dim[0]+1)),
                y=offsety+(y+1)*(maxdimension/(puzzle.dim[1]+1)),
                anchor_x='center',
                anchor_y='center')
            number.draw()
 
    top = window.height - font_large
    labels = [("Current heuristic function: ", 16, top)]
    for h in heuristics:
        prefix = "  "
        if curHeur is h:
            prefix = "* "

        # (len(heuristics) - len(labels))
        y = top - len(labels) * round(1.5 * font_small)
        text = prefix + h.name + ": " + str(puzzle.heuristic(h))

        labels.append((text, 16, y))

    right = window.width - 110
    labels.append(("Hint: " + str(puzzle.hint), right, top))
    labels.append(("Debug: " + str(debug), right, top - font_large))

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
        labels.append((controls[i], 16,
                       (len(controls)+1-i)*round(1.5*font_small)))
    
    font = 'Times New Roman'
    for text, posx, posy in labels:
        pyglet.text.Label(text, font_name=font, font_size=font_small, x=posx,
                          y=posy, anchor_x='left', anchor_y='center').draw()


@window.event
def on_key_press(symbol, modifiers):
    global puzzle, curHeur, heuristics, debug

    if symbol == key.B:
        puzzle.solve(puzzle.search(searches[0], heuristics[0], _debug=debug))

    elif symbol == key.A:
        puzzle.solve(puzzle.search(searches[1], heuristics[0], _debug=debug))

    elif symbol == key.I:
        puzzle.solve(puzzle.search(searches[2], heuristics[0], _debug=debug))

    elif symbol == key.X:
        debug = not debug

    elif symbol == key.ENTER:   # step to solved state
        puzzle.reset()

    elif symbol == key.Q:
        puzzle.update([6, 7, 14, 8, 5, 1, 15, 12, 13, 0, 10, 9, 4, 2, 3, 11])

    elif symbol == key.SPACE:
        puzzle.step()

    elif symbol == key.C:
        puzzle.debugheuristic()

    elif symbol == key.H:
        puzzle.calchint()

    elif symbol == key.R:
        puzzle.randomize(curHeur.function)

    elif symbol == key.T:
        puzzle.randomize(curHeur.function, _bound=20)

    elif symbol == key.E:
        new_index = (heuristics.index(curHeur)+1) % len(heuristics)
        curHeur = heuristics[new_index]

    elif symbol == key.LEFT:
        puzzle.move(0)

    elif symbol == key.RIGHT:
        puzzle.move(3)

    elif symbol == key.UP:
        puzzle.move(1)

    elif symbol == key.DOWN:
        puzzle.move(2)

    elif symbol == key.Y:
        puzzle.twistmoves()

    elif symbol == key.P:
        puzzle.debugsolution()

    else:
        print("Unassigned Key: " + str(symbol))

    # controls = ["Controls: ",
    #             "Arrowkeys to move tiles",
    #             "'b' - search BFS",
    #             "'a' - search A*",
    #             "'i' - search IDA*",
    #             "'r' - generate random puzzle",
    #             "'ENTER' - reset puzzle",
    #             "'SPACE' - step through solution",
    #             "'c' - print current heuristic cost",
    #             "'e' - change heuristic function",
    #             "'h' - get a hint for next move"]

keys = {
    key.B:     (lambda: puzzle.runBFS(heuzr=curHeur, _debug=debug), ""),
    key.A:     (lambda: puzzle.runAStar(heur=curHeur, _debug=debug), ""),
    key.X:     (lambda: print(), ""),
    key.ENTER: (lambda: puzzle.reset(), ""),
    key.Q:     (lambda: puzzle.update([6, 7, 14, 8, 5, 1, 15, 12, 13, 0, 10, 9, 4, 2, 3, 11]), ""),
    key.SPACE: (lambda: puzzle.step(), ""),
    key.C:     (lambda: print("Absolute cost: " + str(curHeur(('', puzzle.boardcopy()), puzzle.dim))), ""),
    key.H:     (lambda: puzzle.calchint(), ""),
    key.R:     (lambda: puzzle.randomize(_heur=curHeur), ""),
    key.T:     (lambda: puzzle.randomize(_heur=curHeur, _bound=20), ""),
    key.E:     (lambda: print(), ""),
    key.I:     (lambda: puzzle.solve(puzzle.runIDA()), ""),
    key.LEFT:  (lambda: puzzle.moveleft(), ""),
    key.RIGHT: (lambda: puzzle.moveright(), ""),
    key.UP:    (lambda: puzzle.moveup(), ""),
    key.DOWN:  (lambda: puzzle.movedown(), ""),
    key.Y:     (lambda: puzzle.twistmoves(), ""),
    key.P:     (lambda: puzzle.debugsolution(), "")}

# Into puzzle:
# make hint depended of solution
# toggle heuristics (key E)
# toggle debug (key X)


if __name__ == '__main__':

    puzzle = Puzzle(4, 4)

    heuristics = [Heuristic("Manhattan Distance", hCostManhattan),
                  Heuristic("Misplaced Tiles", hCostMpt),
                  Heuristic("Tiles out of row and column", hCostToorac),
                  Heuristic("Lineo Conflict", hCostLinearConflict),
                  Heuristic("Manhattan * 3", hCostMhtn3x),
                  Heuristic("Manhattan * 2", hCostMhtn2x),
                  Heuristic("Manhattan * 1.5", hCostMhtn1_5x)]

    searches = [Search("BFS", Queue),
                Search("A*", PriorityQueue),
                Search("IDA*", None)]

    curHeur = heuristics[0]

    curSearch = searches[0]

    pyglet.app.run()
