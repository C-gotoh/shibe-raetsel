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


# Globals
class Dim(Enum):
	x = 4
	y = 4

# global to be able to write in event functions
global maxdimension
global puzzle
global dimensions

dimensions = (4,4)

drawsize = 1
blankvalue = 0

# Functions

def printPuzzle(puzzle):
    for i in range(Dim.y.value):
        line = ""
        for j in range(Dim.x.value):
            line = line + " " + str(puzzle[i][j])
        print(line)
    print("-----")

def heuristicCost(puzzle):
    # TODO
    return 0


def getPosition(puzzle, number):
    for k,v in puzzle.items():
        for s,w in puzzle[k].items():
            if w == number:
                #(x,y)
                return (s,k)
    return []

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

print(getPosition(puzzle,1))

puzzle = moveblank(puzzle,"left")
printPuzzle(puzzle)
printPuzzle(endpuzzle)

puzzle = genericSearch([puzzle],[endpuzzle])[0][-1]

printPuzzle(puzzle)


@window.event
def on_resize(width, height):
    global maxdimension
    maxdimension = min(width,height)
    print('The window was resized to %dx%d' % (width, height))

@window.event
def on_draw():
    window.clear()

    offsetx, offsety = ((window.width-maxdimension)/2, (window.height-maxdimension)/2)
    for i in range(Dim.y.value):
        for j in range(Dim.x.value):
            number = pyglet.text.Label(str(puzzle[i][j]), font_size=20, x=offsetx+(j+1)*(maxdimension/(Dim.x.value+1)), y=offsety+(i+1)*(maxdimension/(Dim.y.value+1)),
                anchor_x='center', anchor_y='center')
            number.draw()
            #love.graphics.print(tostring(puzzle[i][j]),offset.x+j*(maxdimension/(dimensions.x+1)),offset.y+i*(maxdimension/(dimensions.y+1)),0,drawsize,drawsize)

    if bgimg != None:
    	bgimg.blit(0, 0)
    label = pyglet.text.Label('15-Puzzle', font_name='Times New Roman', font_size=36, 
	x=window.width//2, y=window.height//2, anchor_x='center', anchor_y='center')
    label.draw()

@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.LEFT:
        print('The left arrow key was pressed.')
    elif symbol == key.ENTER:
        print('The enter key was pressed.')

@window.event
def on_mouse_press(x, y, button, modifiers):
    if button == mouse.LEFT:
        print('The left mouse button was pressed.')
        print(maxdimension)


pyglet.app.run()