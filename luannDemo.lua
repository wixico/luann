local luann = require("luann")
math.randomseed(89890)

learningRate = 50 -- set between 1, 100
attempts = 10000 -- number of times to do backpropagation
threshold = 1 -- steepness of the sigmoid curve

--create a network with 2 inputs, 3 hidden cells, and 1 output
myNetwork = luann:new({2,3, 1}, learningRate, threshold)

--run backpropagation (bp)
for i = 1,attempts do
	myNetwork:bp({0,0},{0})
	myNetwork:bp({1,0},{1})
	myNetwork:bp({0,1},{1})
	myNetwork:bp({1,1},{0})
end

--print the signal of the single output cell when :activated with different inputs
print("Results:")
myNetwork:activate({0,0})
print("0 0 | " .. myNetwork[3].cells[1].signal)
myNetwork:activate({0,1})
print("0 0 | " .. myNetwork[3].cells[1].signal)
myNetwork:activate({1,0})
print("0 0 | " .. myNetwork[3].cells[1].signal)
myNetwork:activate({1,1})
print("0 0 | " .. myNetwork[3].cells[1].signal)

--Save the network to a file
luann:saveNetwork(myNetwork, "myTestNetwork.network")

--Load the network from a file
newNetwork = luann:loadNetwork("myTestNetwork.network")

--run the loaded network
print("Results:")
newNetwork:activate({0,0})
print("Output of 0,0: " .. newNetwork[3].cells[1].signal)
newNetwork:activate({0,1})
print("Output of 0,1: " .. newNetwork[3].cells[1].signal)
newNetwork:activate({1,0})
print("Output of 1,0: " .. newNetwork[3].cells[1].signal)
newNetwork:activate({1,1})
print("Output of 1,1: " .. newNetwork[3].cells[1].signal)
