--[[
The MIT License (MIT)

Copyright (c) <2013> <Josh Rowe>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

]]--

--Borrowed table persistence from http://lua-users.org/wiki/TablePersistence, MIT license.
--comments removed, condensed code to oneliners where possible.
local write, writeIndent, writers, refCount;
persistence =
{
	store = function (path, ...)
		local file, e = io.open(path, "w")
		if not file then return error(e)	end
		local n = select("#", ...)
		local objRefCount = {} -- Stores reference that will be exported
		for i = 1, n do refCount(objRefCount, (select(i,...))) end
		local objRefNames = {}
		local objRefIdx = 0;
		file:write("-- Persistent Data\n");
		file:write("local multiRefObjects = {\n");
		for obj, count in pairs(objRefCount) do
			if count > 1 then
				objRefIdx = objRefIdx + 1;
				objRefNames[obj] = objRefIdx;
				file:write("{};"); -- table objRefIdx
			end;
		end;
		file:write("\n} -- multiRefObjects\n");
		for obj, idx in pairs(objRefNames) do
			for k, v in pairs(obj) do
				file:write("multiRefObjects["..idx.."][");
				write(file, k, 0, objRefNames);
				file:write("] = ");
				write(file, v, 0, objRefNames);
				file:write(";\n");
			end;
		end;
		for i = 1, n do
			file:write("local ".."obj"..i.." = ");
			write(file, (select(i,...)), 0, objRefNames);
			file:write("\n");
		end
		if n > 0 then
			file:write("return obj1");
			for i = 2, n do
				file:write(" ,obj"..i);
			end;
			file:write("\n");
		else
			file:write("return\n");
		end;
		file:close();
	end;
	load = function (path)
		local f, e = loadfile(path);
		if f then
			return f();
		else
			return nil, e;
		end;
	end;
}
write = function (file, item, level, objRefNames)
	writers[type(item)](file, item, level, objRefNames);
end;
writeIndent = function (file, level)
	for i = 1, level do
		file:write("\t");
	end;
end;
refCount = function (objRefCount, item)
	if type(item) == "table" then
		if objRefCount[item] then
			objRefCount[item] = objRefCount[item] + 1;
		else
			objRefCount[item] = 1;
			for k, v in pairs(item) do
				refCount(objRefCount, k);
				refCount(objRefCount, v);
			end;
		end;
	end;
end;
writers = {
	["nil"] = function (file, item) file:write("nil") end;
	["number"] = function (file, item)
			file:write(tostring(item));
		end;
	["string"] = function (file, item)
			file:write(string.format("%q", item));
		end;
	["boolean"] = function (file, item)
			if item then
				file:write("true");
			else
				file:write("false");
			end
		end;
	["table"] = function (file, item, level, objRefNames)
			local refIdx = objRefNames[item];
			if refIdx then
				file:write("multiRefObjects["..refIdx.."]");
			else
				file:write("{\n");
				for k, v in pairs(item) do
					writeIndent(file, level+1);
					file:write("[");
					write(file, k, level+1, objRefNames);
					file:write("] = ");
					write(file, v, level+1, objRefNames);
					file:write(";\n");
				end
				writeIndent(file, level);
				file:write("}");
			end;
		end;
	["function"] = function (file, item)
			local dInfo = debug.getinfo(item, "uS");
			if dInfo.nups > 0 then
				file:write("nil --[[functions with upvalue not supported]]");
			elseif dInfo.what ~= "Lua" then
				file:write("nil --[[non-lua function not supported]]");
			else
				local r, s = pcall(string.dump,item);
				if r then
					file:write(string.format("loadstring(%q)", s));
				else
					file:write("nil --[[function could not be dumped]]");
				end
			end
		end;
	["thread"] = function (file, item)
			file:write("nil --[[thread]]\n");
		end;
	["userdata"] = function (file, item)
			file:write("nil --[[userdata]]\n");
		end;
}

local luann = {}
local Layer = {}
local Cell = {}
local exp = math.exp

--We start by creating the cells.
--The cell has a structure containing weights that modify the input from the previous layer.
--Each cell also has a signal, or output.
function Cell:new(numInputs)
	local cell = {delta = 0, weights = {}, signal = 0}
		for i = 1, numInputs do
			cell.weights[i] = math.random() * .1
		end
		setmetatable(cell, self)
		self.__index = self
	return cell
end

function Cell:activate(inputs, bias, threshold)
		local signalSum = bias
		local weights = self.weights
		for i = 1, #weights do
			signalSum = signalSum + (weights[i] * inputs[i])
		end
	self.signal = 1 / (1 + exp((signalSum * -1) / threshold))
end

--Next we create a Layer of cells. The layer is a table of cells.
function Layer:new(numCells, numInputs)
	numCells = numCells or 1
	numInputs = numInputs or 1
	local cells = {}
		for i = 1, numCells do cells[i] = Cell:new(numInputs) end
		local layer = {cells = cells, bias = math.random()}
		setmetatable(layer, self)
		self.__index = self
	return layer
end

--layers = {table of layer sizes from input to output}
function luann:new(layers, learningRate, threshold)
	local network = {learningRate = learningRate, threshold = threshold}
	--initialize the input layer
	network[1] = Layer:new(layers[1], layers[1])
	--initialize the hidden layers and output layer
	for i = 2, #layers do
		network[i] = Layer:new(layers[i], layers[i-1])
	end
	setmetatable(network, self)
	self.__index = self
	return network
end

function luann:activate(inputs)
	local threshold = self.threshold
	for i = 1, #inputs do
		self[1].cells[i].signal = inputs[i]
	end
	for i = 2, #self do
		local passInputs = {}
		local cells = self[i].cells
		local prevCells = self[i-1].cells
		for m = 1, #prevCells do
			passInputs[m] = prevCells[m].signal
		end
		local passBias = self[i].bias
		for j = 1, #cells do
			--activate each cell
			cells[j]:activate(passInputs, passBias, threshold)
		end
	end
end

function luann:decode(hiddenSignal)

	--iterate over the hidden layer and set their signals to hiddenInputs
	for i = 1, #self[2].cells do
		self[2].cells[i].signal = hiddenSignal[i]
	end

	local threshold = self.threshold

	for i = 3, #self do
		local passInputs = {}
		local cells = self[i].cells
		local prevCells = self[i-1].cells
		for m = 1, #prevCells do
			passInputs[m] = prevCells[m].signal
		end
		local passBias = self[i].bias
		for j = 1, #cells do
			--activate each cell
			cells[j]:activate(passInputs, passBias, threshold)
		end
	end

end


function luann:bp(inputs, outputs)
	self:activate(inputs) --update the internal inputs and outputs
	local numSelf = #self
	local learningRate = self.learningRate
	for i = numSelf, 2, -1 do --iterate backwards (nothing to calculate for input layer)
		local numCells = #self[i].cells
		local cells = self[i].cells
		for j = 1, numCells do
			local signal = cells[j].signal
			if i ~= numSelf then --special calculations for output layer
				local weightDelta = 0
				local layer = self[i+1].cells
				for k = 1, #self[i+1].cells do
					weightDelta = weightDelta + layer[k].weights[j] * layer[k].delta
				end
				cells[j].delta = signal * (1 - signal) * weightDelta
			else
				cells[j].delta = (outputs[j] - signal) * signal * (1 - signal)
			end
		end
	end

	for i = 2, numSelf do
		self[i].bias = self[i].cells[#self[i].cells].delta * learningRate
		for j = 1, #self[i].cells do
			for k = 1, #self[i].cells[j].weights do
				local weights = self[i].cells[j].weights
				weights[k] = weights[k] + self[i].cells[j].delta * learningRate * self[i-1].cells[k].signal
			end
		end
	end
end

function luann:saveNetwork(network, savefile)
	print(savefile)
	persistence.store(savefile, network)
end

function luann:loadNetwork(savefile)
	local ann = persistence.load(savefile)
		ann.bp = luann.bp
		ann.activate = luann.activate
		for i = 1, #ann do
			for j = 1, #ann[i].cells do
				ann[i].cells[j].activate = Cell.activate
			end
		end
	return(ann)
end

function luann:loadTrainingDataFromFile(fileName)
local trainingData = {}
local fileLines = {}
    local f = io.open(fileName, "rb")
		 for line in f:lines() do
			table.insert (fileLines, line);
		 end
	f:close()

	for i = 1, #fileLines do
		if i%2 == 0 then
				local tempInputs = {}
				for input in fileLines[i]:gmatch("%S+") do table.insert(tempInputs, tonumber(input)) end
				local tempOutputs = {}
				for output in fileLines[i+1]:gmatch("%S+") do table.insert(tempOutputs, tonumber(input)) end
			table.insert(trainingData, {tempInputs, tempOutputs})
		end
	end
return(trainingData)
end

return(luann)
