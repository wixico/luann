luann
=====

A fast neural network module written for Lua.
This is based loosely off another neural network script
http://www.forums.evilmana.com/psp-lua-codebase/lua-neural-networks/
Thanks soulkiller for the inspiration!

This basic low level neural network module can be used to train and run neural networks.
I plan on abstracting the activation function and creating a few convenience training functions to do training from files.

The demo script illustrates basic creation, training, saving, and loading.

The output of any cell can be seen and manipulated by accessing .signal for the cell.
This will allow the simple creation of complex neural network hierarchies.
The output of hidden layers can easily be made the input layer for other networks, and training is done incrementally at the network level.

For now, the demo is a simple implementation of XOR
