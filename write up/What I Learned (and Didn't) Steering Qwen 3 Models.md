### Investigating Steering Vectors Across Model Scale

+ calls for pragmatic approaches to technical AI safety which usually includes linear methods like probes and steering vectors but major accomplishments on the mechanistic interpretability side with circuits and transcoders
+ Linear methods provide a low compute way of achieving desired safety outcomes but at the price of little understanding of how those methods work within the model
+ Probes have been deployed in anthropic systems such as the constitutional classifiers for identifying unwanted behavior on the inputs and outputs with remarkable success
+ Steering vectors have shown success in changing model personas as well as steering models away from evaluation awareness to better assess misalignment
+ Steering vectors can have reliability issues, confounds, and dataset sensitivity
+ I ran several steering experiments over the holidays across the Qwen3 family (4B to 235B) on several behavior concepts
+ My origianl intention was examining how model size impacts steerability. [one paper examined this.](https://arxiv.org/abs/2507.11771v1)
+ This post walks though I found-including where the initial hypothesis fell apart

### Steering Vectors

+ Steering vectors are a fairly simple concept at an operational level. 
+ Apply some bias vector to the residual stream at a specific layer
+ Push the activations in a direction that corresponds with a specific concept (i.e. be more aggreable)
+ Usually a concept is found and then that direction in the model's internal representation is either added or substracted during inference to modulate behavior

### Contrastive Activation Addition (CAA)

+ The dominant method for extracting steering vectors involves taking the difference of activations for sets of constrastive pairs
+ Pairs will often consistent of chat model inputs with a 'user' inputted prompt with two multiple choice answers followed by a 'chatbot' completed response selecting one of the multiple choice answers.
+ Both pairs are run through the model and the residual stream post some selected layer is taken
+ The difference of those two residual stream representations of the pairs should elicit a direction in representation space for the concept of interest, the steering vector
+ This process can be effective with as little as a single prompt pair but more recent approaches take the average over many pairs
+ At inference time the model can then be steered by add or subtracting the steering vector often moldulated by some scalar to elicit specific behavior
+ Steering vectors can have reliability concerns and not every concept produces clean steering vectors
+ [Tan et al. (2024)](https://arxiv.org/abs/2407.12404) demonstrated that steerability is high variable across inputs from tests sets of constrastive pairs
+ Spurious biases can inflate apparent effectiveness and reasonable prompt changes can often diminish to reverse steering vector effects

