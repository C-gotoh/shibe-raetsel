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


class Dim(Enum):
    x = 4
    y = 4


# Globals
global maxdimension
global puzzle
global solutions
global dimensions
global steps
global solved
global limit


# Performance Data
added_nodes = 0
closed_nodes = 0
omitted_nodes = 0
heuristic_calls = 0


# global values
solved = True
limit = 15  # maximum heuristicCost of new random Puzzle ( <= len(solution))
steps = 20  # number of steps to shuffle puzzle ( >= len(solution))


# ui
font_large = 32
font_small = 14
font_number = 20

dimensions = (Dim.x.value, Dim.y.value)

drawsize = 1
blankvalue = 0


# Functions


def printPuzzle(puzzle):
    for y in range(Dim.y.value):
        line = ""
        for x in range(Dim.x.value):
            line = line + " " + str(puzzle[y][x])
        print(line)
    print("-----")


def shufflePuzzle(puzzle, iterations):
    lastdirection = 0
    directions = ["left", "right", "up", "down"]
    direction = 0
    for i in range(iterations):
        while direction == lastdirection:
                direction = directions[randint(0, len(directions)-1)]
        lastdirection = direction
        puzzle = moveblank(puzzle, direction)
    return puzzle


def getPosition(puzzle, number):
    numberindex = []
    for y in range(Dim.y.value):
        for x in range(Dim.x.value):
            if puzzle[y][x] == number:
                # print(str(x) + str(y) + str(puzzle[y][x]) + str(number))
                return (x, y)
    return numberindex

# heuristic funktion: Manhattan Distanz
def heuristicCostMannahtan(path):
    puzzle = path[-1]
    cost = 0
    for y in range(dimensions[1]):
        for x in range(dimensions[0]):
            expectednumber = y*Dim.x.value+x+1
            if expectednumber == Dim.x.value*Dim.y.value:
                # expectednumber = 0
                continue
            actualposition = getPosition(puzzle, expectednumber)
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

# heuristic function: Toorac = tiles out of row and column
def heuristicCostToorac(path):
    puzzle = path[-1]
    cost = 0
    cols = []
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            cols.append([])
            cols[x].append(puzzle[y][x])
    for y in range(dimensions[1]):
        for x in range(dimensions[0]):
            expectedNumber = x + y * dimensions[0] +1
            if expectedNumber == 16:
                continue
            if not expectedNumber in puzzle[y]:
                cost += 1
                print("wrong row: " + str(expectedNumber))
                print(puzzle[y])
            if not expectedNumber in cols[x]:
                cost += 1
                print("wrong col: " + str(expectedNumber))
    return cost

# heuristic funktion: Mpt = Misplaced Tiles
def heuristicCostMpt(path):
    puzzle = path[-1]
    cost = 0
    cols = []
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            expectedNumber = x + y * dimensions[0] + 1
            if expectedNumber == 16:
                continue
            actualnumber = puzzle[y][x]
            if expectedNumber != actualnumber:
                cost += 1
    return cost


#---------------------------------------------------------------------------------------------------------------------------------


# heuristic funktion: X-Y
def heuristicCostYX(path):
    puzzle = path[-1]
    cost = 0
    cols = []
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            expectedNumber = x + y * dimensions[0] + 1
            if expectedNumber == 16:
                continue
            actualnumber = puzzle[y][x]
            if expectedNumber != actualnumber:
                cost += 1
    return cost




#---------------------------------------------------------------------------------------------------------------------------------

    global heuristic_calls
    heuristic_calls += 1
    return len(path)-1 + cost


def isSolveAble(puzzle):
    plist = []
    # printPuzzle(puzzle)
    for y in range(Dim.y.value):
        for x in range(Dim.x.value):
            plist.append(puzzle[y][x])
    inversions = 0
    plist.remove(0)
    for num in plist:
        for other in plist:
            if (plist.index(num) < plist.index(other)) and (other < num):
                inversions += 1
    blankrow = getPosition(puzzle, 0)[1]
    print("blankrow: " + str(blankrow))
    print("inversions: " + str(inversions))
    if (Dim.x.value % 2) != 0:
        return (inversions % 2) == 0
    else:
        if ((Dim.y.value - blankrow) % 2) != 0:
            return (inversions % 2) == 0
        else:
            return (inversions % 2) != 0


def getRandomPuzzle():
    while True:
        numbers = list(range(Dim.x.value*Dim.y.value))
        rndPuzzle = {}
        for y in range(Dim.y.value):
            rndPuzzle[y] = {}
            for x in range(Dim.x.value):
                num = random.choice(numbers)
                rndPuzzle[y][x] = num
                numbers.remove(num)
        if isSolveAble(rndPuzzle) and (heuristicCost([rndPuzzle]) < limit):
            print("heuristicCost: " + str(heuristicCost([rndPuzzle])))
            return rndPuzzle


def updatePuzzle(puzzle, clickedNumber):
    puzzle = deepcopy(puzzle)
    zeroindex = getPosition(puzzle, blankvalue)
    clickedindex = getPosition(puzzle, clickedNumber)

    if clickedindex[0] == zeroindex[0]:
        diff = abs(clickedindex[1] - zeroindex[1])
        if diff <= 1:
                puzzle[clickedindex[1]][clickedindex[0]] = blankvalue
                puzzle[zeroindex[1]][zeroindex[0]] = clickedNumber
    if clickedindex[1] == zeroindex[1]:
        diff = abs(clickedindex[0] - zeroindex[0])
        if diff <= 1:
                puzzle[clickedindex[1]][clickedindex[0]] = blankvalue
                puzzle[zeroindex[1]][zeroindex[0]] = clickedNumber
    return puzzle


def moveblank(puzzle, direction):
    adjacentnumber = 0
    changed = False
    zeroindex = getPosition(puzzle, blankvalue)
    if direction == "left":
        if zeroindex[0] > 0:
            adjacentnumber = puzzle[zeroindex[1]][zeroindex[0]-1]
            changed = True
    elif direction == "right":
        if zeroindex[0] < dimensions[0]-1:
            adjacentnumber = puzzle[zeroindex[1]][zeroindex[0]+1]
            changed = True
    elif direction == "up":
        if zeroindex[1] < dimensions[1]-1:
            adjacentnumber = puzzle[zeroindex[1]+1][zeroindex[0]]
            changed = True
    elif direction == "down":
        if zeroindex[1] > 0:
            adjacentnumber = puzzle[zeroindex[1]-1][zeroindex[0]]
            changed = True
    if changed:
        return updatePuzzle(puzzle, adjacentnumber)
    else:
        return puzzle


def switchTile(puzzle, position):
    # TODO implement
    return puzzle


def getNeighbors(puzzle):
    # TODO: optimize check, currently very slow
    # TODO: use switchTile instead of moveblank
    neighborTable = []
    left, right, up, down = (moveblank(puzzle, "left"),
                             moveblank(puzzle, "right"),
                             moveblank(puzzle, "up"),
                             moveblank(puzzle, "down"))
    if left != puzzle:
        neighborTable.append(left)
    if right != puzzle:
        neighborTable.append(right)
    if up != puzzle:
        neighborTable.append(up)
    if down != puzzle:
        neighborTable.append(down)
    return neighborTable


def genericSearch(startPosList, endPosList, _dataStructure=Queue,
                  _heuristic=False, _debug=False):
    visited = []
    frontier = _dataStructure()
    max_frontier_len = 0
    ####
    global added_notes
    added_nodes = 0 # Never used
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


def initPuzzle():
    # Init puzzle
    puzzle = {}
    for i in range(Dim.y.value):
        puzzle[i] = {}
        for j in range(Dim.x.value):
            puzzle[i][j] = (j+1+Dim.x.value*i)

    puzzle[Dim.y.value-1][Dim.x.value-1] = blankvalue
    return puzzle


def getHint(puzzle):
    # TODO: print or display real hint instead of puzzle
    global solution
    if puzzle == endpuzzle:
        print("hint: do nothing")
    else:
        print("calculating hint ...")
        solution = genericSearch([puzzle], [endpuzzle])[0]
        if len(solution) != 0:
            printPuzzle(solution[1])


# pyglet init
window = pyglet.window.Window(resizable=True, caption='15-Puzzle')

maxdimension = min(window.width, window.height)

bgimg = None
# bgimg = pyglet.resource.image('data/img/img.png')

pyglet.gl.glClearColor(0.1, 0.1, 0.1, 1)

puzzle = initPuzzle()

endpuzzle = deepcopy(puzzle)

solution = []

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
    window.clear()

    global puzzle
    global endpuzzle
    global solved

    if puzzle == endpuzzle:
        solved = True

    offsetx, offsety = ((window.width-maxdimension)/2,
                        (window.height-maxdimension)/2)

    if bgimg is not None:
        bgimg.blit(offsetx, offsety)

    for i in range(Dim.y.value):
        for j in range(Dim.x.value):
            if puzzle[i][j] != 0:
                number = pyglet.text.Label(
                    str(puzzle[i][j]), font_size=font_number,
                    x=offsetx+(j+1)*(maxdimension/(Dim.x.value+1)),
                    y=offsety+(i+1)*(maxdimension/(Dim.y.value+1)),
                    anchor_x='center', anchor_y='center')
                number.draw()

    if solved:
        labelstring = "SOLVED"
    else:
        labelstring = ''

    label = pyglet.text.Label(labelstring, font_name='Times New Roman',
                              font_size=font_large, x=window.width//2,
                              y=window.height-36, anchor_x='center',
                              anchor_y='center')
    label.draw()

    controls = ["Arrowkeys to move tiles",
                "'b' - shuffle and BFS",
                "'a' - randomize and A*",
                "'ENTER' - go to solution",
                "'SPACE' - step through solution",
                "'c' - display current heuristic cost",
                "'h' - get a hint for next move"]
    for i in range(len(controls)):
        pyglet.text.Label(controls[i], font_name='Times New Roman',
                          font_size=font_small, x=16,
                          y=(len(controls)+1-i)*round(1.5*font_small),
                          anchor_x='left', anchor_y='center').draw()
    solved = False


@window.event
def on_key_press(symbol, modifiers):
    global solution
    global puzzle
    global steps

    if symbol == key.B:     # randomize puzzle
        if len(solution) != 0:
            solution = []
        puzzle = deepcopy(endpuzzle)
        puzzle = shufflePuzzle(puzzle, steps)
        print("Built puzzle with", steps, "random moves.")
        print("searching...")
        tstart = timer()
        solution = genericSearch([puzzle], [endpuzzle])[0]
        tend = timer()
        elapsed_time = tend - tstart
        print("search complete, number of steps: ", len(solution)-1,
              ". time to complete: ", elapsed_time, "s.")
        print("Is solveable: " + str(isSolveAble(puzzle)))
    elif symbol == key.A:
        if len(solution) != 0:
            puzzle = solution.pop()
            solution = []
        # puzzle = shufflePuzzle(puzzle,steps)
        puzzle = getRandomPuzzle()
        print("Built complete random puzzle")
        print("searching...")
        tstart = timer()
        solution = genericSearch([puzzle], [endpuzzle],
                                 _dataStructure=PriorityQueue,
                                 _heuristic=True)[0]
        tend = timer()
        elapsed_time = tend - tstart

        if solution != 0:
            print("search complete, number of steps: ",len(solution)-1,
                ". time to complete: ", elapsed_time, "s.")

    elif symbol == key.ENTER:   # step to solution
        if len(solution) != 0:
            solution = []
        puzzle = deepcopy(endpuzzle)
    elif symbol == key.SPACE:
        if len(solution) != 0:
            puzzle = solution.pop(0)
        else:
            print("done")
    elif symbol == key.C:
        print("Absolute cost: " + str(heuristicCostMpt([puzzle])))
    elif symbol == key.H:
        getHint(puzzle)
    elif symbol == key.LEFT:
        puzzle = moveblank(puzzle, "right")
    elif symbol == key.RIGHT:
        puzzle = moveblank(puzzle, "left")
    elif symbol == key.UP:
        puzzle = moveblank(puzzle, "down")
    elif symbol == key.DOWN:
        puzzle = moveblank(puzzle, "up")


@window.event
def on_mouse_press(x, y, button, modifiers):
    global puzzle
    # if button == mouse.LEFT:


if __name__ == '__main__':
    pyglet.app.run()
