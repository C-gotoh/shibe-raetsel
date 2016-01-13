#!/usr/bin/python

# Imports
import pyglet ## INSTALL
from pyglet.gl import *
from pyglet.window import mouse
from pyglet.window import key
from enum import Enum
from queue import LifoQueue, Queue, PriorityQueue
import sys
import numpy as np ## INSTALL
from copy import deepcopy
from random import randint
from timeit import default_timer as timer


# Globals
class Dim(Enum):
	x = 4
	y = 4

# global to be able to write in event functions
global maxdimension
global puzzle
global solution
global dimensions
global steps

steps = 40  # number of steps to shuffle puzzle (!= len(solution))

dimensions = (Dim.x.value,Dim.y.value)

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
    directions = ["left","right","up","down"]
    direction = 0
    for i in range(iterations):
        while direction == lastdirection:
                direction = directions[randint(0,len(directions)-1)]
        lastdirection = direction
        puzzle = moveblank(puzzle, direction)
    return puzzle

def getPosition(puzzle, number):
    numberindex = []
    for y in range(Dim.y.value):
        for x in range(Dim.x.value):
            if puzzle[y][x] == number:
                #print(str(x) + str(y) + str(puzzle[y][x]) + str(number))
                return (x,y)
    return numberindex

def heuristicCost(puzzle):
    cost = 0
    for y in range(dimensions[1]):
        for x in range(dimensions[0]):
            expectednumber = y*Dim.x.value+x+1
            if expectednumber == Dim.x.value*Dim.y.value:
                #expectednumber = 0
                continue
            actualposition = getPosition(puzzle,expectednumber)
            manhattanDist = abs(x - actualposition[0]) + abs(y - actualposition[1])
            cost += manhattanDist
            print("expected: " + str(expectednumber))
            print("pos: " + str(x) + str(y) + " pos of exp: " + str(actualposition[0]) + str(actualposition[1]))
            print("cumulated cost: " + str(cost))
            #Linear Conflict for columns (x): add 2 for each conflict
            if y == 0 and manhattanDist > 0:

                cost += 0
        #Linear Conflict for rows (y): add 2 for each conflict
        cost += 0
    return cost

def updatePuzzle(puzzle, clickedNumber):
    #TODO: check if working as intended (copying subdictionarys)
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
        if zeroindex[0] <= dimensions[0]-1 and zeroindex[0] > 1:
            adjacentnumber = puzzle[zeroindex[1]][zeroindex[0]-1]
            changed = True
    elif direction == "right":
        if zeroindex[0] < dimensions[0]-1 and zeroindex[0] >= 1:
            adjacentnumber = puzzle[zeroindex[1]][zeroindex[0]+1]
            changed = True
    elif direction == "up":
        if zeroindex[1] <= dimensions[1]-1 and zeroindex[1] > 1:
            adjacentnumber = puzzle[zeroindex[1]-1][zeroindex[0]]
            changed = True
    elif direction == "down":
        if zeroindex[1] < dimensions[1]-1 and zeroindex[1] >=1:
            adjacentnumber = puzzle[zeroindex[1]+1][zeroindex[0]]
            changed = True
    if changed:
        return updatePuzzle(puzzle, adjacentnumber)
    else:
        return puzzle

def getNeighbors(puzzle):
    #TODO: optimize check, currently very slow
    neighborTable = []
    left, right, up, down = (moveblank(puzzle,"left"), moveblank(puzzle,"right"), 
        moveblank(puzzle,"up"), moveblank(puzzle,"down"))
    if left != puzzle:
        neighborTable.append(left)
    if right != puzzle:
        neighborTable.append(right)
    if up != puzzle:
        neighborTable.append(up)
    if down != puzzle:
        neighborTable.append(down)
    return neighborTable

def genericSearch(startPosList, endPosList, _dataStructure=Queue, _heuristic=False, _debug=False):
    visited = []
    frontier = _dataStructure()
    max_frontier_len = 0

    for startPos in startPosList:
        if _heuristic:
            frontier.put((0,[startPos]))
        else:
            frontier.put([startPos])

    while not frontier.empty():
        if frontier.qsize() > max_frontier_len: max_frontier_len = frontier.qsize()
        path = []
        if _heuristic:
            path = frontier.get()[-1]
        else:
            path = frontier.get()

        head = path[-1]

        if head not in visited:
            visited.append(head)
            if head in endPosList:
                return path,len(visited),max_frontier_len
            for neighbor in getNeighbors(head):
                new_path = [n for n in path]
                new_path.append(neighbor)
                if _debug:
                    debug(space,new_path)
                if _heuristic:
                    frontier.put((heuristicCost(new_path,endPosList[0]),new_path))
                else:
                    frontier.put(new_path)
    return 0,len(visited),max_frontier_len

# Init stuff

window = pyglet.window.Window(resizable=True, caption='15-Puzzle')

maxdimension = min(window.width,window.height)

bgimg = None
#bgimg = pyglet.resource.image('data/img/img.png')

pyglet.gl.glClearColor(0.1,0.1,0.1,1)

# Init puzzle

puzzle = {}
for i in range(Dim.y.value):
    puzzle[i] = {}
    for j in range(Dim.x.value):
        puzzle[i][j] = (j+1+Dim.x.value*i)

puzzle[Dim.y.value-1][Dim.x.value-1] = blankvalue

endpuzzle = deepcopy(puzzle)

solution = []

#puzzle = shufflePuzzle(puzzle,steps)
#printPuzzle(puzzle)
#printPuzzle(endpuzzle)

#solution = genericSearch([puzzle],[endpuzzle])[0]

#printPuzzle(puzzle)


@window.event
def on_resize(width, height):
    global maxdimension
    maxdimension = min(width,height)
    print('The window was resized to %dx%d' % (width, height))
    if bgimg != None:
        bgimg.width = maxdimension
        bgimg.height = maxdimension

@window.event
def on_draw():
    window.clear()

    offsetx, offsety = ((window.width-maxdimension)/2, (window.height-maxdimension)/2)

    if bgimg != None:
        bgimg.blit(offsetx,offsety)

    for i in range(Dim.y.value):
        for j in range(Dim.x.value):
            number = pyglet.text.Label(str(puzzle[i][j]), font_size=20, x=offsetx+(j+1)*(maxdimension/(Dim.x.value+1)), y=offsety+(i+1)*(maxdimension/(Dim.y.value+1)),
                anchor_x='center', anchor_y='center')
            number.draw()

    label = pyglet.text.Label('15-Puzzle', font_name='Times New Roman', font_size=36, 
	x=window.width//2, y=window.height//2, anchor_x='center', anchor_y='center')
    label.draw()

@window.event
def on_key_press(symbol, modifiers):
    global solution
    global puzzle
    global steps

    if symbol == key.R:     # randomize puzzle
        if len(solution) != 0:
            puzzle = solution.pop()
            solution = []
        puzzle = shufflePuzzle(puzzle,steps)
        print("Built puzzle with",steps,"random moves.")
        print("searching...")
        tstart = timer()
        solution = genericSearch([puzzle],[endpuzzle])[0]
        tend = timer()
        elapsed_time = tend - tstart
        print("search complete, number of steps: ",len(solution)-1,
            ". time to complete: ", elapsed_time, "s.")
    elif symbol == key.ENTER:   # step to solution
        if len(solution) != 0:
            puzzle = solution.pop()
            solution = []
    elif symbol == key.SPACE:
        if len(solution) != 0:
            puzzle = solution.pop(0)
        else:
            print("done")
    elif symbol == key.C:
        print("Absolute cost: " + str(heuristicCost(puzzle)))

@window.event
def on_mouse_press(x, y, button, modifiers):
    global puzzle
    if button == mouse.LEFT:
        print("mouse")

pyglet.app.run()