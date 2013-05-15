local luann = require("luann")
math.randomseed(os.time())
--math.randomseed(47)
learningRate = 5 -- set between 1, 100
attempts = 10000 -- number of times to do backpropagation
threshold = 1 -- steepness of the sigmoid curve

--create a neural network with 9 inputs
--the neural network is initially trained to be an autoencoder
--the format is 0,0,0,0,0,0,0,0,0
patternNetwork = luann:new({9,4, 9}, learningRate, threshold)

--run backpropagation (bp)
for i = 1,attempts do
	patternNetwork:bp({1,0,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0,0})
	patternNetwork:bp({0,1,0,0,0,0,0,0,0},{0,1,0,0,0,0,0,0,0})
	patternNetwork:bp({0,0,1,0,0,0,0,0,0},{0,0,1,0,0,0,0,0,0})
	patternNetwork:bp({0,0,0,1,0,0,0,0,0},{0,0,0,1,0,0,0,0,0})
	patternNetwork:bp({0,0,0,0,1,0,0,0,0},{0,0,0,0,1,0,0,0,0})
	patternNetwork:bp({0,0,0,0,0,1,0,0,0},{0,0,0,0,0,1,0,0,0})
	patternNetwork:bp({0,0,0,0,0,0,1,0,0},{0,0,0,0,0,0,1,0,0})
	patternNetwork:bp({0,0,0,0,0,0,0,1,0},{0,0,0,0,0,0,0,1,0})
	patternNetwork:bp({0,0,0,0,0,0,0,0,1},{0,0,0,0,0,0,0,0,1})
end

--Save the network to a file
--luann:saveNetwork(patternNetwork, "autoencoder.network")

--We will activate the sequence manually to begin with. In a live system, the process would be automated.
--start with 1

local inputsTable = {}
patternNetwork:activate({1,0,0,0,0,0,0,0,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do
	table.insert(nextInputs, patternNetwork[2].cells[i].signal)
end
table.insert(inputsTable, nextInputs)
patternNetwork:activate({0,1,0,0,0,0,0,0,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do
	table.insert(nextInputs, patternNetwork[2].cells[i].signal)
end
table.insert(inputsTable, nextInputs)
patternNetwork:activate({0,0,1,0,0,0,0,0,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do
	table.insert(nextInputs, patternNetwork[2].cells[i].signal)
end
table.insert(inputsTable, nextInputs)
patternNetwork:activate({0,0,0,1,0,0,0,0,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do
	table.insert(nextInputs, patternNetwork[2].cells[i].signal)
end
table.insert(inputsTable, nextInputs)
patternNetwork:activate({0,0,0,0,1,0,0,0,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do
	table.insert(nextInputs, patternNetwork[2].cells[i].signal)
end
table.insert(inputsTable, nextInputs)
patternNetwork:activate({0,0,0,0,0,1,0,0,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do
	table.insert(nextInputs, patternNetwork[2].cells[i].signal)
end
table.insert(inputsTable, nextInputs)
patternNetwork:activate({0,0,0,0,0,0,1,0,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do
	table.insert(nextInputs, patternNetwork[2].cells[i].signal)
end
table.insert(inputsTable, nextInputs)
patternNetwork:activate({0,0,0,0,0,0,0,1,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do
	table.insert(nextInputs, patternNetwork[2].cells[i].signal)
end
table.insert(inputsTable, nextInputs)
patternNetwork:activate({0,0,0,0,0,0,0,0,1})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do
	table.insert(nextInputs, patternNetwork[2].cells[i].signal)
end
table.insert(inputsTable, nextInputs)

sequenceNetwork = luann:new({4, 4, 4}, learningRate, threshold)

--a new network is generated, trained to predict current input from previous input
for i = 1, attempts do
	for j = 1, #inputsTable-1 do
	sequenceNetwork:bp(inputsTable[j],inputsTable[j+1])
	end
end

--and now we go back up the chain
--first, activate the sequence.
--Get the encoded output of the hidden layer
local inputsTable = {}
patternNetwork:activate({1,0,0,0,0,0,0,0,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do table.insert(nextInputs, patternNetwork[2].cells[i].signal) end
--next, feed the output nextInputs to the sequenceNetwork
sequenceNetwork:activate(nextInputs)
--now get the predicted output
local prediction = {}
for i = 1, #sequenceNetwork[3].cells do table.insert(prediction, sequenceNetwork[3].cells[i].signal) end
--now decode the predicted output
patternNetwork:decode(prediction)
outputsString = ""
for i = 1, #patternNetwork[3].cells do outputsString = outputsString .. " " .. string.format("%.1f", patternNetwork[3].cells[i].signal) end
print("1,0,0,0,0,0,0,0,0 : " .. outputsString)
local inputsTable = {}

patternNetwork:activate({0,1,0,0,0,0,0,0,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do table.insert(nextInputs, patternNetwork[2].cells[i].signal) end
--next, feed the output nextInputs to the sequenceNetwork
sequenceNetwork:activate(nextInputs)
--now get the predicted output
local prediction = {}
for i = 1, #sequenceNetwork[3].cells do table.insert(prediction, sequenceNetwork[3].cells[i].signal) end
--now decode the predicted output
patternNetwork:decode(prediction)
outputsString = ""
for i = 1, #patternNetwork[3].cells do outputsString = outputsString .. " " .. string.format("%.1f", patternNetwork[3].cells[i].signal) end
print("0,1,0,0,0,0,0,0,0 : " .. outputsString)

patternNetwork:activate({0,0,1,0,0,0,0,0,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do table.insert(nextInputs, patternNetwork[2].cells[i].signal) end
--next, feed the output nextInputs to the sequenceNetwork
sequenceNetwork:activate(nextInputs)
--now get the predicted output
local prediction = {}
for i = 1, #sequenceNetwork[3].cells do table.insert(prediction, sequenceNetwork[3].cells[i].signal) end
--now decode the predicted output
patternNetwork:decode(prediction)
outputsString = ""
for i = 1, #patternNetwork[3].cells do outputsString = outputsString .. " " .. string.format("%.1f", patternNetwork[3].cells[i].signal) end
print("0,0,1,0,0,0,0,0,0 : " .. outputsString)

patternNetwork:activate({0,0,0,1,0,0,0,0,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do table.insert(nextInputs, patternNetwork[2].cells[i].signal) end
--next, feed the output nextInputs to the sequenceNetwork
sequenceNetwork:activate(nextInputs)
--now get the predicted output
local prediction = {}
for i = 1, #sequenceNetwork[3].cells do table.insert(prediction, sequenceNetwork[3].cells[i].signal) end
--now decode the predicted output
patternNetwork:decode(prediction)
outputsString = ""
for i = 1, #patternNetwork[3].cells do outputsString = outputsString .. " " .. string.format("%.1f", patternNetwork[3].cells[i].signal) end
print("0,0,0,1,0,0,0,0,0 : " .. outputsString)

patternNetwork:activate({0,0,0,0,1,0,0,0,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do table.insert(nextInputs, patternNetwork[2].cells[i].signal) end
--next, feed the output nextInputs to the sequenceNetwork
sequenceNetwork:activate(nextInputs)
--now get the predicted output
local prediction = {}
for i = 1, #sequenceNetwork[3].cells do table.insert(prediction, sequenceNetwork[3].cells[i].signal) end
--now decode the predicted output
patternNetwork:decode(prediction)
outputsString = ""
for i = 1, #patternNetwork[3].cells do outputsString = outputsString .. " " .. string.format("%.1f", patternNetwork[3].cells[i].signal) end
print("0,0,0,0,1,0,0,0,0 : " .. outputsString)

patternNetwork:activate({0,0,0,0,0,1,0,0,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do table.insert(nextInputs, patternNetwork[2].cells[i].signal) end
--next, feed the output nextInputs to the sequenceNetwork
sequenceNetwork:activate(nextInputs)
--now get the predicted output
local prediction = {}
for i = 1, #sequenceNetwork[3].cells do table.insert(prediction, sequenceNetwork[3].cells[i].signal) end
--now decode the predicted output
patternNetwork:decode(prediction)
outputsString = ""
for i = 1, #patternNetwork[3].cells do outputsString = outputsString .. " " .. string.format("%.1f", patternNetwork[3].cells[i].signal) end
print("0,0,0,0,0,1,0,0,0 : " .. outputsString)

patternNetwork:activate({0,0,0,0,0,0,1,0,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do table.insert(nextInputs, patternNetwork[2].cells[i].signal) end
--next, feed the output nextInputs to the sequenceNetwork
sequenceNetwork:activate(nextInputs)
--now get the predicted output
local prediction = {}
for i = 1, #sequenceNetwork[3].cells do table.insert(prediction, sequenceNetwork[3].cells[i].signal) end
--now decode the predicted output
patternNetwork:decode(prediction)
outputsString = ""
for i = 1, #patternNetwork[3].cells do outputsString = outputsString .. " " .. string.format("%.1f", patternNetwork[3].cells[i].signal) end
print("0,0,0,0,0,0,1,0,0 : " .. outputsString)

patternNetwork:activate({0,0,0,0,0,0,0,1,0})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do table.insert(nextInputs, patternNetwork[2].cells[i].signal) end
--next, feed the output nextInputs to the sequenceNetwork
sequenceNetwork:activate(nextInputs)
--now get the predicted output
local prediction = {}
for i = 1, #sequenceNetwork[3].cells do table.insert(prediction, sequenceNetwork[3].cells[i].signal) end
--now decode the predicted output
patternNetwork:decode(prediction)
outputsString = ""
for i = 1, #patternNetwork[3].cells do outputsString = outputsString .. " " .. string.format("%.1f", patternNetwork[3].cells[i].signal) end
print("0,0,0,0,0,0,0,1,0 : " .. outputsString)

patternNetwork:activate({0,0,0,0,0,0,0,0,1})
local nextInputs = {}
for i = 1, #patternNetwork[2].cells do table.insert(nextInputs, patternNetwork[2].cells[i].signal) end
--next, feed the output nextInputs to the sequenceNetwork
sequenceNetwork:activate(nextInputs)
--now get the predicted output
local prediction = {}
for i = 1, #sequenceNetwork[3].cells do table.insert(prediction, sequenceNetwork[3].cells[i].signal) end
--now decode the predicted output
patternNetwork:decode(prediction)
outputsString = ""
for i = 1, #patternNetwork[3].cells do outputsString = outputsString .. " " .. string.format("%.1f", patternNetwork[3].cells[i].signal) end
print("0,0,0,0,0,0,0,0,1 : " .. outputsString)
