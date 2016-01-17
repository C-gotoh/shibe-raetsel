#!/usr/bin/env python3

# Imports
import pyglet  # INSTALL
import pyglet.gl
from pyglet.window import key
from queue import Queue, PriorityQueue  # ,LifoQueue
import random
from timeit import default_timer as timer

# from heapq import heappop, heappush
import cProfile

# Functions


class Puzzle(object):
    def __init__(self, dimX, dimY):
        self.dim = (dimX, dimY)
        self.setSolved()

    def randomize(self):
        while True:
            random.shuffle(self.field)
            self.solved = self.field == self.getSolvedState()
            self.checkSolvable()
            if self.solvable:
                break

    def setSolved(self):
        self.field = self.getSolvedState()

        self.solved = True
        self.solvable = True

    def getSolvedState(self):
        solved = list(range(self.dim[0] * self.dim[1]))[1:]
        solved.append(0)
        return solved

    def isSolved(self):
        return self.solved

    def checkSolvable(self):
        inversions = 0
        for num in self.field:
            if num == 0:
                continue
            for other in self.field:
                if other == 0:
                    continue

                numpos = self.field.index(num)
                otherpos = self.field.index(other)
                if (numpos < otherpos) and (other < num):
                    inversions += 1

        if (self.dim[0] % 2) != 0:
            self.solvable = (inversions % 2) == 0
        else:
            if ((self.dim[1] - self.getPosition(0)[1]) % 2) != 0:
                self.solvable = (inversions % 2) == 0
            else:
                self.solvable = (inversions % 2) != 0

    def getPosition(self, element):
        return self.field.index(element) % self.dim[0],\
               self.field.index(element) // self.dim[0]

    def getElement(self, x, y):
        return self.field[y*self.dim[0] + x]

    def getState(self):
        return self.field[:]

    def setField(self, newfield):
        self.field = newfield[:]

        self.solved = self.field == self.getSolvedState()
        if self.solved:
            self.solvable = True
        else:
            self.checkSolvable()

    def moveLeft(self, reversed):
        new = list(getNeighborStates(self.field, self.dim))
        if reversed:
            new.reverse()
        new = new[0]
        if new is not None:
            self.field = new
        self.solved = self.field == self.getSolvedState()

    def moveRight(self, reversed):
        new = list(getNeighborStates(self.field, self.dim))
        if reversed:
            new.reverse()
        new = new[3]
        if new is not None:
            self.field = new
        self.solved = self.field == self.getSolvedState()

    def moveUp(self, reversed):
        new = list(getNeighborStates(self.field, self.dim))
        if reversed:
            new.reverse()
        new = new[1]
        if new is not None:
            self.field = new
        self.solved = self.field == self.getSolvedState()

    def moveDown(self, reversed):
        new = list(getNeighborStates(self.field, self.dim))
        if reversed:
            new.reverse()
        new = new[2]
        if new is not None:
            self.field = new
        self.solved = self.field == self.getSolvedState()


class Search(object):
    def __init__(self, arg1, arg2):
        print(arg1)
        print(arg2)


def heuristicA(path, dim):
    return 0


# highly used function
def getStatePosition(state, dim, element):
    index = state.index(element)
    return index % dim[0], index // dim[0]


# highly used function!
def hCostManhattan(path, dim):
    state = path[1]
    cost = 0
    for y in range(dim[1]):
        for x in range(dim[0]):
            expectednumber = y * dim[0] + x + 1
            if expectednumber == dim[0] * dim[1]:
                # expectednumber = 0
                continue
            actualposition = getStatePosition(state, dim, expectednumber)
            manhattanDist = abs(x - actualposition[0])\
                + abs(y - actualposition[1])
            cost += manhattanDist
            # print("expected: " + str(expectednumber))
            # print("pos: " + str(x) + str(y) + " pos of exp: " +
            #       str(actualposition[0]) + str(actualposition[1]))
            # print("cumulated cost: " + str(cost))
            # Linear Conflict for columns (x): add 2 for each conflict
            if y == 0 and manhattanDist > 0:
                cost += 0
        # Linear Conflict for rows (y): add 2 for each conflict
        cost += 0
    return cost  # + len(path)


# highly used function!
# heuristic function: Toorac = tiles out of row and column
def hCostToorac(path, dim):
    state = path[1]
    cost = 0
    global cols
    for y in range(dim[1]):
        for x in range(dim[0]):
            expectedNumber = x + y * dim[0] + 1
            if expectedNumber == dim[0]*dim[1]:
                continue
            if expectedNumber not in state[y*dim[0]:y*dim[0]+dim[1]]:
                cost += 1
                # print("wrong row: " + str(expectedNumber))
                # print(state[y])
            if expectedNumber not in cols[x]:
                cost += 1
                # print("wrong col: " + str(expectedNumber))
    return cost  # + len(path)


# highly used function!
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
# heuristic funktion: X-Y
def hCostYX(path, dim):
    state = path[-1]
    cost = 0
    for x in range(dim[0]):
        for y in range(dim[1]):
            val = y * dim[0] + x
            expectedNumber = val + 1
            if expectedNumber == dim[0]*dim[1]:
                continue
            actualnumber = state[val]
            if expectedNumber != actualnumber:
                cost += 1
    return cost + len(path)


# highly used function!
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
    # right:
    iswap = izero + 1
    if izero_fdiv == iswap // dim[0]:
        right = state[:]
        right[izero] = right[iswap]
        right[iswap] = 0
    # up:
    iswap = izero + dim[0]
    if izero_mod == iswap % dim[1] and iswap < len(state):
        up = state[:]
        up[izero] = up[iswap]
        up[iswap] = 0

    # down:
    iswap = izero - dim[0]
    if izero_mod == iswap % dim[1] and iswap > 0:
        down = state[:]
        down[izero] = down[iswap]
        down[iswap] = 0

    return (left, up, down, right)


def genericSearch(start_pos, end_state, _dataStructure=Queue,
                  _heuristic=False, _debug=False):
    visited = set()
    frontier = _dataStructure()
    # heap = []

    max_frontier = 0

    global heuristic_calls
    heuristic_calls = 0

    item = (curHeur(('', start_pos), puzzle.dim), 1, ('', start_pos))

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
                return (path[0], start_pos), len(visited), max_frontier

            if _debug and len(visited) % 10000 == 0:
                print("----------\n" +
                      "Heur. calls:   " + str(heuristic_calls) + "\n" +
                      "Visited nodes: " + str(len(visited)) + "\n" +
                      "Max. frontier: " + str(max_frontier) + "\n" +
                      "Cur Distance:  " + str(hcosts) + " | " +
                      str(hcosts-plength+1) + "h, " + str(plength - 1) + "p")
                # if len(visited) == 200000: return

            left, up, down, right = getNeighborStates(head, puzzle.dim)

            if left is not None:
                new_path = (path[0] + "0", left)
                frontier.put((curHeur(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            if up is not None:
                new_path = (path[0] + "1", up)
                frontier.put((curHeur(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            if down is not None:
                new_path = (path[0] + "2", down)
                frontier.put((curHeur(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            if right is not None:
                new_path = (path[0] + "3", right)
                frontier.put((curHeur(new_path, puzzle.dim) + plength,
                             plength,
                             new_path))

            # frontier elements: (hcost+plength, plength, (string, state))

    return 0, len(visited), max_frontier


def idaSearch(startPos, endPos, _dataStructure=Queue,
              _heuristic=False, _debug=False):
    global added_nodes
    global heuristic_calls
    heuristic_calls = 0
    added_nodes = 0

    cost = curHeur(("", startPos),puzzle.dim)
    bound = 2
    tstart = timer()
    
    while True:
        path = idaIteration(("",startPos), 1, bound, endPos)
        if path is not None:
            return path
        tnow = timer()
        elapsed_time = tnow - tstart
        print("Iteration " + str(bound) + " done in " + str(elapsed_time) + " (cumulated)")
        bound += 1


def idaIteration(path, lenpath, bound, endPos):
    visited = set()
    frontier = []
    frontier.append(path)
    while frontier:
        #print(frontier)
        path = frontier.pop()
        node = path[-1]
        estlen = len(path[0]) + curHeur(path, puzzle.dim)
        if str(node) not in visited:
            visited.add(str(node))
            if estlen > bound:
                # print("skipping")
                # print(len(frontier))
                continue
            if node == endPos:
                return path

            left, up, down, right = getNeighborStates(node, puzzle.dim)

            if left is not None:
                new_path = (path[0] + "0", left)
                frontier.append(new_path)

            if up is not None:
                new_path = (path[0] + "1", up)
                frontier.append(new_path)

            if down is not None:
                new_path = (path[0] + "2", down)
                frontier.append(new_path)

            if right is not None:
                new_path = (path[0] + "3", right)
                frontier.append(new_path)
    return None


def getHint(puzzle):
    # TODO: print or display real hint instead of puzzle
    global solution
    if puzzle.isSolved():
        print("hint: do nothing")
    else:
        print("calculating hint ...")
        solution, a, b = genericSearch(puzzle.getState(),
                                 puzzle.getSolvedState(),
                                 _dataStructure=PriorityQueue,
                                 _heuristic=True)[0]
        if len(solution) != 0:
            printPuzzle(solution[1])

global window, maxdimension, bgimg, puzzle,\
       solution, curHeur, heuristics,\
       arrow_keys_reversed, cols

arrow_keys_reversed = False

heuristics = [hCostManhattan,
              hCostYX,
              hCostMpt,
              hCostToorac]
curHeur = heuristics[0]

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
    if puzzle.isSolved():
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
            if puzzle.getElement(x, y) is not 0:
                number = pyglet.text.Label(
                    str(puzzle.getElement(x, y)),
                    font_size=font_number,
                    x=offsetx+(x+1)*(maxdimension/(puzzle.dim[0]+1)),
                    y=offsety+(y+1)*(maxdimension/(puzzle.dim[1]+1)),
                    anchor_x='center',
                    anchor_y='center')
                number.draw()

    # ---- Draw heuristic function chooser
    heuristicDialog = []
    for heu in heuristics:
        heuristicDialog.append(str(heu))

    pyglet.text.Label("Current heuristic function: ",
                      font_name='Times New Roman',
                      font_size=font_small,
                      x=16,
                      y=window.height-font_large,
                      anchor_x='left',
                      anchor_y='center').draw()

    for i in range(len(heuristics)):
        if curHeur == heuristics[i]:
            pyglet.text.Label("* " + str(heuristics[i]),
                              font_name='Times New Roman',
                              font_size=font_small,
                              x=16,
                              y=window.height-font_large -
                              (len(heuristics)+1-i)*round(1.5*font_small),
                              anchor_x='left',
                              anchor_y='center').draw()
        else:
            pyglet.text.Label("  " + str(heuristics[i]),
                              font_name='Times New Roman',
                              font_size=font_small,
                              x=16,
                              y=window.height-font_large -
                              (len(heuristics)+1-i)*round(1.5*font_small),
                              anchor_x='left',
                              anchor_y='center').draw()

    # ---- Draw current heuristic cost
    pyglet.text.Label("Cost " + str(curHeur(('', puzzle.getState()),
                                    puzzle.dim)),
                      font_name='Times New Roman',
                      font_size=font_small,
                      x=window.width - font_large*3,
                      y=window.height-font_large,
                      anchor_x='left',
                      anchor_y='center').draw()

    # ---- Draw controls
    controls = ["Controls: ",
                "Arrowkeys to move tiles",
                "'b' - search BFS",
                "'a' - search A*",
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
    global solution, puzzle,\
           curHeur, heuristics,\
           arrow_keys_reversed

    if symbol == key.B:
        print("searching...")
        tstart = timer()

        solution, a, b = genericSearch(puzzle.getState(), puzzle.getSolvedState())
        solution.append(puzzle.getState())

        tend = timer()
        elapsed_time = tend - tstart

        print("search complete, number of steps: ", len(solution[0]),
              ". time to complete: ", elapsed_time, "s.")

    elif symbol == key.A:
        print("searching...")
        tstart = timer()

        solution, a, b = genericSearch(puzzle.getState(),
                                 puzzle.getSolvedState(),
                                 _dataStructure=PriorityQueue,
                                 _heuristic=True)

        tend = timer()
        elapsed_time = tend - tstart

        print("search complete, number of steps: ", len(solution[0]),
              ". time to complete: ", elapsed_time, "s.")

    elif symbol == key.X:
        cProfile.run("genericSearch(puzzle.getState(),\
                                    puzzle.getSolvedState(),\
                                    _dataStructure=PriorityQueue,\
                                    _heuristic=True,\
                                    _debug=True)")

    elif symbol == key.ENTER:   # step to solution
        puzzle.setSolved()
        solution = ('', puzzle.getState())

    elif symbol == key.SPACE:
        if solution is not None and solution[0] != '':
            solution = (solution[0][1:],
                        getNeighborStates(solution[1],
                                          puzzle.dim)[int(solution[0][0])])
            puzzle.setField(solution[1])
        else:
            solution = ('', puzzle.getState())
            print("done")

    elif symbol == key.C:
        print("Absolute cost: " +
              str(curHeur(('', puzzle.getState()), puzzle.dim)))

    elif symbol == key.H:
        getHint(puzzle)

    elif symbol == key.R:

        print("Built random puzzle")
        puzzle.randomize()
        print("Done")

        solution = ('', puzzle.getState())

    elif symbol == key.T:
        bound = 20
        print("Built random puzzle with bound (" + str(bound) + ")")
        puzzle.randomize()
        while curHeur(('', puzzle.getState()), puzzle.dim) > bound:
            puzzle.randomize()
        print("Done")

        solution = ('', puzzle.getState())

    elif symbol == key.E:
        new_index = (heuristics.index(curHeur)+1) % len(heuristics)
        curHeur = heuristics[new_index]

    elif symbol == key.I:
        print("searching...")
        tstart = timer()
        solution = idaSearch(puzzle.getState(), puzzle.getSolvedState())
        tend = timer()
        elapsed_time = tend - tstart
        print("search complete, number of steps: ", len(solution[0]),
              ". time to complete: ", elapsed_time, "s.")

    elif symbol == key.LEFT:
        puzzle.moveLeft(arrow_keys_reversed)
        solution = ('', puzzle.getState())

    elif symbol == key.RIGHT:
        puzzle.moveRight(arrow_keys_reversed)
        solution = ('', puzzle.getState())

    elif symbol == key.UP:
        puzzle.moveUp(arrow_keys_reversed)
        solution = ('', puzzle.getState())

    elif symbol == key.DOWN:
        puzzle.moveDown(arrow_keys_reversed)
        solution = ('', puzzle.getState())

    elif symbol == key.Y:
        arrow_keys_reversed = not arrow_keys_reversed

if __name__ == '__main__':
    global puzzle, solution, cols

    puzzle = Puzzle(4, 4)
    solution = ('', puzzle.getState())

    cols = []
    for x in range(puzzle.dim[0]):
        for y in range(puzzle.dim[1]):
            cols.append([])
            cols[x].append(puzzle.getState()[y * puzzle.dim[0] + x])

    pyglet.app.run()
