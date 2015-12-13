dimensions = {x=4,y=4}
blankvalue = 0

function updatePuzzle(clickedNumber)
	zeroindex = {}
	clickedindex = {}
	for k,v in pairs(puzzle) do
		for s,w in pairs(puzzle[k]) do
			if w == blankvalue then
				zeroindex.x = s
				zeroindex.y = k
			elseif w == clickedNumber then
				clickedindex.x = s
				clickedindex.y = k
			end
		end
	end
	--print(clickedindex.x)
	--print(clickedindex.y)
	--print(zeroindex.x)
	--print(zeroindex.y)
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
				puzzle[clickedindex.x][clickedindex.y] = blankvalue
				puzzle[zeroindex.x][zeroindex.y] = clickedNumber
		end
	end
end

function isEndCondition()
	isEnd = true
	for i=1,dimensions.y do
		for j=1,dimensions.x do
			if i==dimensions.y and j==dimensions.x then
				break
			elseif not puzzle[i][j] == dimensions.x*(i-1)+j then
				isEnd = false
				return isEnd
			end
		end
	end
	return isEnd
end

function printPuzzle()
	for i=1,dimensions.y do
		for j=1,dimensions.x do
			print(puzzle[i][j])
		end
		print("\n")
	end
	print("-----")
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
	printPuzzle()
	print(isEndCondition())
	updatePuzzle(12)
	print(isEndCondition())
end

function love.update(dt)
	windowheight = love.graphics.getHeight()
	windowwidth = love.graphics.getWidth()
	maxdimension = math.min(windowwidth, windowheight)
end

function love.draw(dt)
	--print((windowwidth-maxdimension)/2)
	--print((windowheight-maxdimension)/2)
	--love.graphics.draw(defaultimg, (windowwidth-maxdimension)/2, (windowheight-maxdimension)/2, 0, maxdimension/defaultimg:getWidth(), maxdimension/defaultimg:getHeight())
	offset = {x=(love.graphics.getWidth()-maxdimension)/2, y=(love.graphics.getHeight()-maxdimension)/2}
	for i=1, dimensions.y do
		for j=1, dimensions.x do
			--print(puzzle[i][j])
			love.graphics.print(tostring(puzzle[i][j]),offset.x+j*(maxdimension/(dimensions.x+1)),offset.y+i*(maxdimension/(dimensions.y+1)),0,2,2)
		end
	end
end