### Investigating Steering Vectors Across Model Scale

+ more of a call for pragmatic approaches to technical AI safety which usually includes linear methods like probes and steering vectors but major accomplishments on the mechanistic interpretability side with circuits and transcoders
+ Linear methods provide a low compute way of achieving desired safety outcomes but at the price of little understanding of how those methods work within the model
+ Probes have been deployed in anthropic systems such as the constitutional classifiers for identifying unwanted behavior on the inputs and outputs with remarkable success
+ Steering vectors have shown success in changing model personas as well as steering models away from evaluation awareness to better assess misalignment
+ Steering vectors can have reliability issues, confounds, and dataset sensitivity
+ I ran several steering experiments over the holidays across the Qwen3 family (4B to 235B) on several behavior concepts
+ My origianl intention was examining how model size impacts steerability. one paper examined this --reference-- 
+ This post walks though I found-including where the initial hypothesis fell apart

### Steering Vectors

+ Steering vectors are a fairly simple concept at an operational level. 
+ Apply some bias vector to the residual stream at a specific layer
+ Push the activations in a direction that corresponds with a specific concept (i.e. be more aggreable)
+ Usually a concept is found and then that direction in the model's internal representation is either added or substracted during inference to modulate behavior

### Contrastive Activation Addition (CAA)

+ The dominant method for extracting steering vectors involves taking the difference of activations for sets of constrastive pairs