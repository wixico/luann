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

local write, writeIndent, writers, refCount;

persistence =
{
	store = function (path, ...)
		local file, e = io.open(path, "w");
		if not file then
			return error(e);
		end
		local n = select("#", ...);
		-- Count references
		local objRefCount = {}; -- Stores reference that will be exported
		for i = 1, n do
			refCount(objRefCount, (select(i,...)));
		end;
		-- Export Objects with more than one ref and assign name
		-- First, create empty tables for each
		local objRefNames = {};
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
		-- Then fill them (this requires all empty multiRefObjects to exist)
		for obj, idx in pairs(objRefNames) do
			for k, v in pairs(obj) do
				file:write("multiRefObjects["..idx.."][");
				write(file, k, 0, objRefNames);
				file:write("] = ");
				write(file, v, 0, objRefNames);
				file:write(";\n");
			end;
		end;
		-- Create the remaining objects
		for i = 1, n do
			file:write("local ".."obj"..i.." = ");
			write(file, (select(i,...)), 0, objRefNames);
			file:write("\n");
		end
		-- Return them
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
	["nil"] = function (file, item)
			file:write("nil");
		end;
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
				-- Table with multiple references
				file:write("multiRefObjects["..refIdx.."]");
			else
				-- Single use table
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
			-- Does only work for "normal" functions, not those
			-- with upvalues or c functions
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

math.randomseed(os.time())
local Network = {}
local Layer = {}
local Cell = {}
local exp = math.exp

function sigmoid(input, threshold)
	if not(threshold) then threshold = 1 end
	out = 1 / (1 + math.exp((input*-1) / threshold))
	return(out)
end

function sum(a,b)
	local sum = 0
	local len = #a
		for i = 1, len do sum = sum + (a[i]*b[i]) end
	return sum
end

--We start by creating the cells, or neurons. The cell has a structure containing weights that modify the input from the previous layer.
--Each cell also has a signal, or output.
function Cell:new(numInputs, structure)
	structure = structure or {delta = 0, weights = {}, signal = 0}
		for i = 1, numInputs do
			structure.weights[i] = math.random() * .1
			if math.random(1,2) == 2 then structure.weights[i] = structure.weights[i] * -1 end
		end
		setmetatable(structure, self)
		self.__index = self
	return structure
end

--The neuron has an activation function; in this case, the squashed sigmoid sum of all inputs.
function Cell:activate(inputs)
		self.signal = 1 / (1 + exp((sum(self.weights, inputs)*-1) / 1))
end

--Next we create a Layer of cells. The layer is a table of neurons.
function Layer:new(numCells, numInputs, structure)
	numCells = numCells or 1
	numInputs = numInputs or 1
	cells = {}
		for i = 1, numCells do cells[i] = Cell:new(numInputs) end
		structure = structure or {bias = 1, cells = cells}
		setmetatable(structure, self)
		self.__index = self
	return structure
end

--Now we create a Network. A network is a table of layers.
function Network:new(params)
	if params == nil then print("No parameters detected") return end
	structure = {learningRate = .05}
	for i = 1, #params do
		if i == 1 then
			structure[i] = Layer:new(params[1], params[i])
		else
			structure[i] = Layer:new(params[i], params[i-1])
		end
	end
	setmetatable(structure, self)
	self.__index = self
	return structure
end

--accepts a table of inputs, causes all cells in the network to update their signal
function Network:activate(inputs)
	local passInputs = {}
	for i = 1, #self do
		for keys,values in pairs(passInputs) do passInputs[keys] = nil end
		if i >= 2 then for k = 1, #self[i-1].cells do passInputs[k] = self[i-1].cells[k].signal end end
		for j = 1, #self[i].cells do
			if i == 1 then self[i].cells[j]:activate(inputs) end
			if i >= 2 then self[i].cells[j]:activate(passInputs) end
		end
	end
end

function Network:backProp(inputs, outputs)
	self:activate(inputs) --update the signal for each cell
	--iterate over each layer
		for i = #self, 2, -1 do
			--iterate over each cell in each layer
			for cellIndex, cell in ipairs(self[i].cells) do
				--update the deltas of the first layer
				local signal = cell.signal
				if i == #self then
					cell.delta = ((outputs[cellIndex] - signal) * signal * (1 - signal))
				else
				--initialize weightDelta
				local delta = 0
						for nextCellIndex, nextCell in ipairs(self[i + 1].cells) do
							delta = delta + (nextCell.weights[cellIndex] * nextCell.delta)
						end
					cell.delta = signal * (1 - signal) * delta
				end
			end
		end

		for i = 2, #self do
			for j = 1, #self[i].cells do
				for k = 1, #self[i].cells[j].weights do
						self[i].cells[j].weights[k] = self[i].cells[j].weights[k] + self[i].cells[j].delta * self.learningRate * self[i-1].cells[k].signal
				end
			end
		end
end

function saveNetwork(network, savefile)
	persistence.store(savefile, network)
end

function loadNetwork(savefile)
	local network = persistence.load(savefile)
		network.backProp = Network.backProp
		network.activate = Network.activate

		for i = 1, #network do
			for j = 1, #network[i].cells do
				network[i].cells[j].activate = Cell.activate
			end
		end
	return(network)
end

function loadTrainingDataFromFile(fileName)
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


function Network:train(trainingData, testData, numCycles, numTests, targetError, cyclesBetweenReports)
	--the trainingData format is as follows: trainingData[1][1] = table of inputs , trainingData[1][2] = table of outputs
	--testData is in the same form. testData is used to measure the accuracy of the outputs of the network.
	--if the MSE of the collected outputs is higher than targetError, training continues.
	--if the MSE of the outputs is lower, training stops and the network is saved
	--if there is more training data than the number of cycles, only do the first numCycles number of i/o pairs
	for i = 1,  numCycles do
		local meanSquaredError = 0
			self:backProp(trainingData[i][1], trainingData[i][2])
			--every cyclesBetweenReports cycles, report the current status.
			if i%cyclesBetweenReports == 0 then
				--print a report of current network status
				--use the first numTests number of i/o pairs to get estimate
				local MSE = {}
				for j = 1, numTests do
					self:activate(trainingData[i][1])
					--get array of outputs
					local outputs = {}
					for k = 1, #self[#self].cells do
						table.insert(outputs, self[#self].cells.signal)
					end
					--get the squared error for each pair in outputs and trainingData[i][2]
					for l = 1, #outputs do
						local squaredError = (trainingData[i][2][l] - outputs[l])^2
						table.insert(MSE, squaredError)
					end

					--get the mean of the squared error of all outputs
					for m = 1, #MSE do
						meanSquaredError = meanSquaredError + MSE[m]
					end
					meanSquaredError = meanSquaredError / #MSE
					print("Mean Squared Error = " .. meanSquaredError)
				end
			end
		if meanSquaredError <= targetError then break end
	end
	--training is concluded when numCycles or MSE is lower than targetError
end



--create a network
myTestNetwork = Network:new({2, 3, 1})
--myTestNetwork:activate({1,1})

--run the network through some training
for i = 1, 10000 do
	myTestNetwork:backProp({0,0},{0})
	myTestNetwork:backProp({1,0},{1})
	myTestNetwork:backProp({0,1},{1})
	myTestNetwork:backProp({1,1},{0})
end

myTestNetwork:activate({0,0})
print("0 0 : " .. myTestNetwork[3].cells[1].signal)
myTestNetwork:activate({0,1})
print("1 0 : " .. myTestNetwork[3].cells[1].signal)
myTestNetwork:activate({1,0})
print("0 1 : " .. myTestNetwork[3].cells[1].signal)
myTestNetwork:activate({1,1})
print("1 1 : " .. myTestNetwork[3].cells[1].signal)

--Save the network to a file
saveNetwork(myTestNetwork, "myTestNetwork.network")

--Load the network from a file
newNetwork = loadNetwork("myTestNetwork.network")

--train the loaded network
newNetwork:backProp({1,1}, {0})

--test the output of the loaded network
print(newNetwork[3].cells[1].signal)


