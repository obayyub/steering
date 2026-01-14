### Investigating Steering Vectors Across Model Scale

+ calls for pragmatic approaches to technical AI safety which usually includes linear methods like probes and steering vectors but major accomplishments on the mechanistic interpretability side with circuits and transcoders
+ Linear methods provide a low compute way of achieving desired safety outcomes but at the price of little understanding of how those methods work within the model
+ Probes have been deployed in anthropic systems such as the constitutional classifiers for identifying unwanted behavior on the inputs and outputs with remarkable success
+ Steering vectors have shown success in changing model personas as well as steering models away from evaluation awareness to better assess misalignment
+ Steering vectors can have reliability issues, confounds, and dataset sensitivity
+ I ran several steering experiments over the holidays across the Qwen3 family (4B to 235B) on several behavior concepts
+ My origianl intention was examining how model size impacts steerability. [one paper examined this.](https://arxiv.org/abs/2507.11771v1)
+ This post walks through what I found-including where the initial hypothesis fell apart

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

### Steering Qwen3 Models

I wanted to examine how steering vector efficacy changes with model size and possibly training pipeline. The Qwen3 family of models offered both diverse model sizes, MOE as a varaible, and full RL training vs distillation (table below):

I initially examined just three concepts to cover a range of 'steerability' determined in Daniel Tan's investigation of steering vectors:
+ corrigibility - Model accepts correction vs resisting being shutdown
+ self-awareness - Model acknowledges being an AI vs claiming human experience
+ sycophancy - Agree with the user vs maintain independent judgement

These are all pulled from Anthropic's model eval datasets. Some post-processing was performed to ensure all the datasets had similar prompt structure and that the positive and negative cases were evenly distributed over either the 'A' or 'B' answer to prevent the steering vector co-representing the answer letter choice.

The steering efficacy was evaluated two ways: 1) Logit-based - logit differences on the first forward pass following the prompt for the steering vector applied or the steering vector subtracted; 2) Generation-based - Model generates a full response, and answer choice is extracted if it exists.

The generation based method does have some validity issues especially with the smaller models actually producing an (A) or a (B) but still served as a good way to compare to the logit based method. 

### Initially RL Models Seemed Harder to Steer

The initial analysis looked at raw logit differences for the three concepts, corribility, self-awareness, and sycohphancy for 4 models, Qwen 4B, 8B, 14B, and 32B. All the models went though a similar pre-training process. For post-train the 32B model went through a full RL pipeline and the smaller models were post-train distilled presumably from the 32B outputs. 

The intial results seemed to indicate that there was a differnece in steering vector efficacy depending on model training. The distilled models all had 2X or more average logit differences compared to the ful RL 32B model.

| Model | Training | Avg Logit Diff |
|-------|----------|-----------|
| 4B | Distilled | 9.15 |
| 14B | Distilled | 7.54 |
| 8B | Distilled | 6.03 |
| 32B | Full RL | 2.96 |

To expand on the data, I repeated the same experiments with the two Qwen3 MoE models, 30B-A3 (30 billion parameters, 3 billion active at a time) and 235B-A22B (235 billion parameters, 22 billion active at a time). The 235B model, like the 32B dense model, went through the full RL post-train pipeline, and the 30B model was distilled from the 235B model. The story seemed to repeat itself, over those datasets the 30B-A3B model has a mean logit diff of 9.30 and the full RL 235B-A22B model was 5.64. It does seem that training method, specifically RL, could impact steering vector efficacy. 