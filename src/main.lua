function updatePuzzle(clickedNumber)
	zeroindex = {}
	clickedindex = {}
	for k,v in pairs(puzzle) do
		for s,w in pairs(puzzle[k]) do
			if w == 0 then
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
				puzzle[clickedindex.x][clickedindex.y] = 0
				puzzle[zeroindex.x][zeroindex.y] = clickedNumber
		end
	end
	if clickedindex.y == zeroindex.y then
		local diff = math.abs(clickedindex.x - zeroindex.x)
		if diff <= 1 then
				puzzle[clickedindex.x][clickedindex.y] = 0
				puzzle[zeroindex.x][zeroindex.y] = clickedNumber
		end
	end
end

function printPuzzle()
		for i=1,4 do
		for k=1,4 do
			print(puzzle[i][k])
		end
		print("\n")
	end
	print("-----")
end

function love.load(arg)
	defaultimg = love.graphics.newImage("data/img/img.png")
	maxdimension = math.min(love.graphics.getWidth(), love.graphics.getHeight())
	puzzle = {}
	for i=1, 4 do
		puzzle[i] = {1+4*(i-1),2+4*(i-1),3+4*(i-1),4+4*(i-1)}
	end
	puzzle[4][4] = 0

	printPuzzle()
	updatePuzzle(4)
end

function love.update(dt)
	windowheight = love.graphics.getHeight()
	windowwidth = love.graphics.getWidth()
	maxdimension = math.min(windowwidth, windowheight)
end

function love.draw(dt)
	--print((windowwidth-maxdimension)/2)
	--print((windowheight-maxdimension)/2)
	love.graphics.draw(defaultimg, (windowwidth-maxdimension)/2, (windowheight-maxdimension)/2, 0, maxdimension/defaultimg:getWidth(), maxdimension/defaultimg:getHeight())
end