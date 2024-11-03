# How should we train the model?
To train the model we need an image proposal and a corresponding target. 
We need to figure out if we should make the proposals used in training on the fly or before. The advantage of making it before is that training will be faster. The disadvantage is that it requires 0.5 GB storage on the GPU

# How are we going to validate the model?
For each image we are going to get the corresponding image proposals and target proposals. Then we will send all the proposals through the model to get a prediction for certainty and box location.
These values need to be saved and I suggest we will do this in the tensorDict
Since many of the proposals may show the same pothole we need to apply non max supression. This algorithm find the most certain perdiction for a pothole with multiple candidates and remove the rest.
Therefore, use the saved values from above to carry out this algorithm

Another problem that is going to occur is to draw the potholes on the original image. When we are getting the proposale the origin for each proposal will change (I think it will be the upper left corner)
and therefore the position for the box will also change. We therefore need to keep track of this location so we can take the "inverse" when drawing on the original image.
Furthermore, we will also apply a transfor on the proposal to ensure that it fits though the model which is a change we also need to take the inverse of, when drawing.
Again I think these should be in the TensorDict

I also think that if we predict a proposal as background then we don't have to make the regression or keep track or the box coordinates and probability.

# How are we going to load the data
Since the training and validation/testing is two different tasks (we need the original image and target in validation/testing but not in training) I suggest that we use two different classes

- class proposals
Use on the training data to return proposals and targets (we need to assign pothole or background to these targets.)
Should be normalized and transformed so they can be applied to the training of the model
Should take class imbalance into account


- class potholes
use on validation and test data to return original image, original targets, proposals for the image, targets for proposals.
Should also be normalized and transformed
Here we need to keep track of the transformation and the change in coordinates when the proposal is selected so we can draw it

# What model should we use?
idk, find one... But the last layer of the model should be changed so it only outputs pothole or background

# How are we going to calculate the validation and test metrics?
We need to make a function that takes the original target and predicted target as an input. We should then display metrics for both the classification and regression task.


  
