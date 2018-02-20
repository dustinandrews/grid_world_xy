**Grid World X,Y example**


A basic example of using Keras and machine learning with a grid world.
The model is a copy of same convolutional Q learning network used in many
if the Atari game solving papers. With an additional "head" that learns other
domian specific knowlege that is either useful in it's own right or helps the 
main model to learn faster.

In this simplified example the auxilary doesn't speed convergence of the Q score,
however other experiments have shown it to be helpful in terms of sample effeciency.

**Grid Word**
![grid world example](sample_grid.png)
This example grid world has a blue player, a green goal and a red failure location.

![Coordinates Example](sample_location.png)
In this sample you can see the probability the grid world assigns to the player being
in each location. It's output is a softmax(-dist) to each cell.

When trained the network will predict the most likely location of the player.

Note that this method with work with just the distances from the corners, but it 
will converge faster if you can supply more cells.