#!/usr/bin/env python3

# Imports
import pyglet # INSTALL
import pyglet.gl
from pyglet.window import mouse # mouse unused
from pyglet.window import key
from enum import Enum
from queue import LifoQueue, Queue, PriorityQueue # LifoQueue unused
import sys # unused
import numpy as np  # INSTALL # unused
from copy import deepcopy
from random import randint
import random
from timeit import default_timer as timer

# Class


def MyPriorityQueue(PriorityQueue):
    def get(self):
        return super().get()[-1]

global window, maxdimension, bgimg, puzzle, solution,currentHeuristic,heuristics

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

# Functions


class Puzzle(object):
    def __init__(self, dimX, dimY):
        self.dim = (dimX, dimY)
        self.setSolved()

    def randomize(self):
        random.shuffle(self.field)

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

    def isSolvable(self):
        return self.solvable

    def getPosition(self, element):
        return self.field.index(element) % self.dim[0],\
               self.field.index(element) // self.dim[0]

    def getElement(self, x, y):
        return self.field[y*self.dim[0] + x]

    def getState(self):
        return self.field[:]

    def update(self, newfield):
        self.field = newfield[:]

        if self.field == self.getSolvedState():
            self.solvable = True
            self.solved = True
        else:
            self.solved = False

            inversions = 0
            print(self.field)
            for num in self.field:
                if num == 0:
                    continue
                for other in self.field:
                    if other == 0:
                        continue
                    if (self.field.index(num) < self.field.index(other))\
                        and (other < num):
                        inversions += 1

            if (self.dim[0] % 2) != 0:
                self.solvable = (inversions % 2) == 0
            else:
                if ((self.dim[1] - self.getPosition(0)[1]) % 2) != 0:
                    self.solvable = (inversions % 2) == 0
                else:
                    self.solvable = (inversions % 2) != 0
        print(self.field)


class Search(object):
    def __init__(self, arg1, arg2):
        print(arg1)
        print(arg2)

def heuristicA(path,dim):
    return 0

def getStatePosition(state, dim, element):
    return state.index(element) % dim[0] , state.index(element) // dim[0]

def heuristicCost(path, dim):
    state = path[-1]
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
            # print("pos: " + str(x) + str(y) + " pos of exp: " + str(actualposition[0]) + str(actualposition[1]))
            # print("cumulated cost: " + str(cost))
            # Linear Conflict for columns (x): add 2 for each conflict
            if y == 0 and manhattanDist > 0:
                cost += 0
        # Linear Conflict for rows (y): add 2 for each conflict
        cost += 0

    global heuristic_calls
    heuristic_calls += 1

    return len(path)-1 + cost

def getNeighborStates(state, dim):
    neighbors = []
    izero = state.index(0)

    # left:
    iswap = izero - 1
    if izero // dim[0] != iswap // dim[0]:
        neighbors.append(None)
    else:
        left = state[:]
        left[izero] = left[iswap]
        left[iswap] = 0
        neighbors.append(left)

    # right:
    iswap = izero + 1
    if izero // dim[0] != iswap // dim[0]:
        neighbors.append(None)
    else:
        right = state[:]
        right[izero] = right[iswap]
        right[iswap] = 0
        neighbors.append(right)

    # up:
    iswap = izero - dim[0]
    if izero % dim[1] != iswap % dim[1] or iswap < 0:
        neighbors.append(None)
    else:
        up = state[:]
        up[izero] = up[iswap]
        up[iswap] = 0
        neighbors.append(up)

    # down:
    iswap = izero + dim[0]
    if izero % dim[1] != iswap % dim[1] or iswap > len(state) - 1:
        neighbors.append(None)
    else:
        down = state[:]
        down[izero] = down[iswap]
        down[iswap] = 0
        neighbors.append(down)

    return neighbors


def genericSearch(startPosList, endPosList, _dataStructure=Queue,
                  _heuristic=False, _debug=False):
    visited = []
    frontier = _dataStructure()
    max_frontier_len = 0
    ####
    global added_notes
    global heuristic_calls
    heuristic_calls = 0
    ####
    for startPos in startPosList:
        if _heuristic:
            frontier.put((0, [startPos]))
        else:
            frontier.put([startPos])

    while not frontier.empty():
        if frontier.qsize() > max_frontier_len:
            max_frontier_len = frontier.qsize()
        path = []
        if _heuristic:
            path = frontier.get()[-1]
        else:
            path = frontier.get()

        head = path[-1]

        if head not in visited:
            visited.append(head)
            ####
            if len(visited) % 1000 == 0:
                print("Heuristic calls: " + str(heuristic_calls))
                print("Visited nodes: " + str(len(visited)))
                print("Max frontier: " + str(max_frontier_len))
            ####
            if head in endPosList:
                return path, len(visited), max_frontier_len

            for neighbor in getNeighbors(head):
                new_path = [n for n in path]
                new_path.append(neighbor)
                if _debug:
                    debug(space, new_path)
                if _heuristic:
                    #####
                    heuristic_calls += 1
                    #####
                    frontier.put((heuristicCost(new_path),
                                  id(new_path), new_path))
                else:
                    frontier.put(new_path)
    return 0, len(visited), max_frontier_len


def getHint(puzzle):
    # TODO: print or display real hint instead of puzzle
    global solution
    if puzzle.isSolved():
        print("hint: do nothing")
    else:
        print("calculating hint ...")
        solution = genericSearch([puzzle.getState()],
                                 [puzzle.getSolvedState])[0]
        if len(solution) != 0:
            printPuzzle(solution[1])

currentHeuristic = heuristicCost

heuristics = [heuristicCost,heuristicA]

# puzzle = shufflePuzzle(puzzle,steps)
# printPuzzle(puzzle)
# printPuzzle(endpuzzle)

# solution = genericSearch([puzzle],[endpuzzle])[0]

# printPuzzle(puzzle)


@window.event
def on_resize(width, height):
    global maxdimension
    maxdimension = min(width, height)
    print('The window was resized to %dx%d' % (width, height))
    if bgimg is not None:
        bgimg.width = maxdimension
        bgimg.height = maxdimension


@window.event
def on_draw():
    global puzzle
    global currentHeuristic

    if puzzle.isSolved():
        pyglet.gl.glClearColor(0.1, 0.3, 0.1, 1)
    else:
        pyglet.gl.glClearColor(0.1, 0.1, 0.1, 1)
    offsetx, offsety = ((window.width-maxdimension)/2,
                        (window.height-maxdimension)/2)
    window.clear()

    if bgimg is not None:
        bgimg.blit(offsetx, offsety)

    for y in range(puzzle.dim[1]):
        for x in range(puzzle.dim[0]):
            if puzzle.getElement(x,y) is not 0:
                number = pyglet.text.Label(
                    str(puzzle.getElement(x, y)),
                    font_size=font_number,
                    x=offsetx+(x+1)*(maxdimension/(puzzle.dim[0]+1)),
                    y=offsety+(y+1)*(maxdimension/(puzzle.dim[1]+1)),
                    anchor_x='center',
                    anchor_y='center')
                number.draw()

    #if solved:
    #    labelstring = "SOLVED"
    #else:
    #    labelstring = ''

    #label = pyglet.text.Label(labelstring, font_name='Times New Roman',
    #                          font_size=font_large, x=window.width//2,
    #                          y=window.height-font_large*3, anchor_x='center',
    #                          anchor_y='center')
    #label.draw()

    heuristicDialog = []
    for heu in heuristics:
        heuristicDialog.append(str(heu))

    pyglet.text.Label("Current heuristic function: ", font_name='Times New Roman',
                          font_size=font_small, x=16,
                          y=window.height-font_large,
                          anchor_x='left', anchor_y='center').draw() 
    
    for i in range(len(heuristics)):
        if currentHeuristic == heuristics[i]:
            pyglet.text.Label("* " + str(heuristics[i]), font_name='Times New Roman',
                          font_size=font_small, x=16,
                          y=window.height-font_large-(len(heuristics)+1-i)*round(1.5*font_small),
                          anchor_x='left', anchor_y='center').draw()    
        else:
            pyglet.text.Label("  " + str(heuristics[i]), font_name='Times New Roman',
                          font_size=font_small, x=16,
                          y=window.height-font_large-(len(heuristics)+1-i)*round(1.5*font_small),
                          anchor_x='left', anchor_y='center').draw()
    pyglet.text.Label("  " + str(currentHeuristic([puzzle.getState()],puzzle.dim)), font_name='Times New Roman',
                          font_size=font_small, x=window.width - font_large,
                          y=window.height-font_large-(len(heuristics)+1-i)*round(1.5*font_small),
                          anchor_x='left', anchor_y='center').draw()

    controls = ["Controls: ",
                "Arrowkeys to move tiles",
                "'b' - search BFS",
                "'a' - search A*",
                "'r' - generate random puzzle",
                "'ENTER' - reset puzzle",
                "'SPACE' - step through solution",
                "'c' - display current heuristic cost",
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
    global solution
    global puzzle
    global currentHeuristic
    global heuristics

    neighbors = getNeighborStates(puzzle.getState(),puzzle.dim)

    state = puzzle.getState()

    if symbol == key.B:
        solution = []
        puzzle.randomize()
        tstart = timer()
        solution = genericSearch([puzzle.getState()],
                                 [puzzle.getSolvedState()])[0]
        tend = timer()
        elapsed_time = tend - tstart
        print("search complete, number of steps: ", len(solution)-1,
              ". time to complete: ", elapsed_time, "s.")

    elif symbol == key.A:
        # puzzle = shufflePuzzle(puzzle,steps)
        print("searching...")
        tstart = timer()
        solution = genericSearch([puzzle.getState()], [puzzle.getSolvedState()],
                                 _dataStructure=PriorityQueue,
                                 _heuristic=True)[0]
        tend = timer()
        elapsed_time = tend - tstart

        if solution != 0:
            print("search complete, number of steps: ",len(solution)-1,
                  ". time to complete: ", elapsed_time, "s.")

    elif symbol == key.ENTER:   # step to solution
        solution = []
        puzzle.setSolved()

    elif symbol == key.SPACE:
        if len(solution) != 0:
            puzzle.update(solution.pop(0))
        else:
            print("done")

    elif symbol == key.C:
        print("Absolute cost: " + str(currentHeuristic([puzzle.getState()]),puzzle.dim))

    elif symbol == key.H:
        getHint(puzzle)

    elif symbol == key.R:
        puzzle.randomize()
        print("Built random puzzle")

    elif symbol == key.E:
        currentHeuristic = heuristics[(heuristics.index(currentHeuristic)+1)%len(heuristics)]

    elif symbol == key.LEFT:
        if neighbors[1] is not None:
            state = neighbors[1]  # right
            puzzle.update(state)

    elif symbol == key.RIGHT:
        if neighbors[0] is not None:
            state = neighbors[0]  # left
            puzzle.update(state)

    elif symbol == key.UP:
        if neighbors[2] is not None:
            state = neighbors[2]  # down
            puzzle.update(state)

    elif symbol == key.DOWN:
        if neighbors[3] is not None:
            state = neighbors[3]  # up
            puzzle.update(state)

if __name__ == '__main__':
    global puzzle, solution
    
    puzzle = Puzzle(4, 4)
    solution = []

    pyglet.app.run()
