dimensions = {x=4,y=4}
blankvalue = 0

function updatePuzzle(puzzle, clickedNumber)
	local puzzle = deepcopy(puzzle)
	local zeroindex = getPosition(puzzle, blankvalue)
	local clickedindex = getPosition(puzzle, clickedNumber)

	if clickedindex.x == zeroindex.x then
		local diff = math.abs(clickedindex.y - zeroindex.y)
		if diff <= 1 then
				puzzle[clickedindex.y][clickedindex.x] = blankvalue
				puzzle[zeroindex.y][zeroindex.x] = clickedNumber
		end
	end
	if clickedindex.y == zeroindex.y then
		local diff = math.abs(clickedindex.x - zeroindex.x)
		if diff <= 1 then
				puzzle[clickedindex.y][clickedindex.x] = blankvalue
				puzzle[zeroindex.y][zeroindex.x] = clickedNumber
		end
	end
	return puzzle
end

function isEndCondition(puzzle)
	isEnd = true
	for i=1,dimensions.y do
		for j=1,dimensions.x do
			if i==dimensions.y and j==dimensions.x then
				break
			elseif puzzle[i][j] ~= dimensions.x*(i-1)+j then
				isEnd = false
				return isEnd
			end
		end
	end
	return isEnd
end

function getPosition(puzzle, number)
	numberindex = {}
	for k,v in ipairs(puzzle) do
		for s,w in ipairs(puzzle[k]) do
			if w == number then
				numberindex.x = s
				numberindex.y = k
				return numberindex
			end
		end
	end
	return numberindex
end

function isMember(element, list)
	for _,v in pairs(list) do
		if v == element then
			return true
		end
	end
	return false
end

function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function printPuzzle(puzzle)
	for i=1,dimensions.y do
		for j=1,dimensions.x do
			print(puzzle[i][j])
		end
		print("\n")
	end
	print("-----")
end

function shufflePuzzle(puzzle, iterations)
	local puzzle = puzzle
	for i=1, iterations do
		puzzle = updatePuzzle(puzzle, math.random(1,dimensions.x*dimensions.y))
	end
	return puzzle
end

function moveblank(puzzle, direction)
	local adjecentnumber = 0
	local changed = false
	local zeroindex = getPosition(puzzle, blankvalue)
	if direction == "left" then
		if zeroindex.x <= dimensions.x and zeroindex.x > 1 then
			adjecentnumber = puzzle[zeroindex.y][zeroindex.x-1]
			changed = true
		end
	elseif direction == "right" then
		if zeroindex.x < dimensions.x and zeroindex.x >= 1 then
			adjecentnumber = puzzle[zeroindex.y][zeroindex.x+1]
			changed = true
		end
	elseif direction == "up" then
		if zeroindex.y <= dimensions.y and zeroindex.y > 1 then
			adjecentnumber = puzzle[zeroindex.y-1][zeroindex.x]
			changed = true
		end
	elseif direction == "down" then
		if zeroindex.y < dimensions.y and zeroindex.y >=1 then
			adjecentnumber = puzzle[zeroindex.y+1][zeroindex.x]
			changed = true
		end
	end
	if changed then
		return updatePuzzle(puzzle, adjecentnumber)
	else
		return puzzle
	end
end

function getNeighbors(puzzle)
	local neighborTable = {}
	if moveblank(puzzle,"left") ~= puzzle then
		table.insert(neighborTable,moveblank(puzzle,"left"))
	end
	if moveblank(puzzle,"right") ~= puzzle then
		table.insert(neighborTable,moveblank(puzzle,"right"))
	end
	if moveblank(puzzle,"up") ~= puzzle then
		table.insert(neighborTable,moveblank(puzzle,"up"))
	end
	if moveblank(puzzle,"down") ~= puzzle then
		table.insert(neighborTable,moveblank(puzzle,"down"))
	end
	return neighborTable
end

function solvePuzzle(puzzle)
	local visited = {}
	local frontier = {}
	frontier[1] = {puzzle}
	while frontier[#frontier] ~= nil do
		local path = frontier[#frontier]
		local head = path[#path]
		print("star")
		--print(visited)
		for k,v in pairs(frontier) do
			print(k)
			printPuzzle(v)
		end
		table.remove(frontier,#frontier)
		if not isMember(head,visited) then
			table.insert(visited,head)
			--printPuzzle(head)
			if isEndCondition(head) then
				print("wtf")
				return path
			end
			for _,neighbor in pairs(getNeighbors(head)) do
				--printPuzzle(neighbor)
				local newpath = deepcopy(path)
				table.insert(newpath,head)
				table.insert(frontier,newpath)
				--print(newpath)
				--print(#frontier)
			end
		end
	end
	print(#frontier)
	return 0
end

function love.load(arg)
	defaultimg = love.graphics.newImage("data/img/img.png")
	maxdimension = math.min(love.graphics.getWidth(), love.graphics.getHeight())
	--init puzzle here
	puzzle = {}
	for i=1, dimensions.y do
		puzzle[i] = {}
		for j=1, dimensions.x do
			table.insert(puzzle[i],j+dimensions.x*(i-1))
		end
	end
	puzzle[dimensions.y][dimensions.x] = blankvalue
	--init puzzle done
	puzzle = updatePuzzle(puzzle,12)
	--puzzle = shufflePuzzle(puzzle, 10)
	--solution = solvePuzzle(puzzle)
	--print(solution)
	--for _,v in pairs(solution) do
	--	printPuzzle(v)
	--end
	--test = updatePuzzle(puzzle,12)
	--puzzle = test
	--print(isEndCondition(test))

end

function love.update(dt)
	windowheight = love.graphics.getHeight()
	windowwidth = love.graphics.getWidth()
	maxdimension = math.min(windowwidth, windowheight)
end

function love.draw(dt)
	love.graphics.setBackgroundColor{240,240,50}
	love.graphics.draw(defaultimg, (windowwidth-maxdimension)/2, (windowheight-maxdimension)/2, 0, maxdimension/defaultimg:getWidth(), maxdimension/defaultimg:getHeight())
	love.graphics.setColor(0,0,0)
	offset = {x=(love.graphics.getWidth()-maxdimension)/2, y=(love.graphics.getHeight()-maxdimension)/2}
	for i=1, dimensions.y do
		for j=1, dimensions.x do
			--print(puzzle[i][j])
			love.graphics.print(tostring(puzzle[i][j]),offset.x+j*(maxdimension/(dimensions.x+1)),offset.y+i*(maxdimension/(dimensions.y+1)),0,2,2)
		end
	end
	love.graphics.setColor(255,255,255)
end