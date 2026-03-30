
| Model                | Activation | Repeat | Learning Rate | Batch Size | Notes | Kv Cache                                                                                                                                                                                                                                                                                                                                                                                   | Mask                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------- | ---------- | ------ | ------------- | ---------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CotFormer full depth | Gelu       |        |               |            |       | in GPTBase.forward before middle layers loop: total_expected_length = self.n_repeat * T (exact length cache will ever need to be). pass that to block.attn.init_cache() which creates empty tensors (all_keys and all_values) during loop use custom autograd InPlaceSetSlide to forcefully inject new loops Key/Value states into pre allocated buffer at correct index apply_inplace_set | CausalSelfAttention.forward:1. self.bias[:,:,:T,:T] grabs a standard lower-triangular causal mask for a single loop (where token 5 can only look at tokens 1, 2, 3, 4, 5).<br>    <br>2. k.shape[2] // T calculates exactly how many loops are currently in the KV cache.<br>    <br>3. .repeat(...) takes that standard triangle mask and **tiles it horizontally** for every past loop. |
|                      |            |        |               |            |       |                                                                                                                                                                                                                                                                                                                                                                                            |                                                                                                                                                                                                                                                                                                                                                                                           |
|                      |            |        |               |            |       |                                                                                                                                                                                                                                                                                                                                                                                            |                                                                                                                                                                                                                                                                                                                                                                                           |
|                      |            |        |               |            |       |                                                                                                                                                                                                                                                                                                                                                                                            |                                                                                                                                                                                                                                                                                                                                                                                           |
Ignore the table

I wrote this to basically give an idea of what I plan to do. These are what I think is best. Please read all of it and try to understand where I'm coming from.
 
 
 ### Criticisms
 
 >We emphasize that using CoTFormers does not introduce an overhead in terms of memory. This is because the storage of intermediary tokens is needed given the need for the KV cache even when using Block Universal Transformers.


what about during inference? the BUT wouldnt need to store intermediary tokens during inference?

>the performance of a CoTFormer with 24 layers and 5 repeats, surpasses the performance of a standard 48 layer Transformer. We call the final resulting architecture LN-CoTFormer. We note that while LN-CoTFormer’s final performance is better than CoTFormer, we observed some benign spikes in the loss during training. Though the model quickly recovers without intervention from these spikes, this might suggest CoTFormer are more stable to train than LN-CoTFormers. Still, we focus on using LN-CoTFormers when building our adaptive model in the next section.

Layer Norm was literally invented to _stop_ training spikes. The fact that the authors added extra Layer Norms and it _caused_ spikes is a massive red flag that the architecture is fighting itself mathematically

This causes me to think that the model might be naturally trying to use the _magnitude_ of the vector to track what loop it is on (e.g., "if my vector is huge, I must be on Loop 4").

By injecting an extra Layer Norm at the end of every single repeat, you are violently squashing the vector back to a mean of 0 and variance of 1. The model is trying to build a thought, and the extra LN keeps resetting the scale of the blackboard. The "spike" is likely the optimizer panicking when the gradients from the final loss clash with this repeated forced squashing, forcing the weights to aggressively readjust
- **Log Activation Magnitudes (Right before the extra LN):**
    
    - _What to track:_ The L2 norm of the hidden states at the exact moment before they hit that new Layer Norm at the end of the loop.
        
    - _What it proves:_ If you see the vector magnitudes steadily growing across loops, and then the magnitude suddenly goes completely haywire right before a loss spike, you've proved the model is fighting the normalization constraint.
        
- **Log Gradient Norms (The Culprit Check):**
    
    - _What to track:_ Separate the gradient norms for the Attention weights, the FFN weights, and the _Extra Layer Norm_ weights.
        
    - _What it proves:_ If the gradient norm of the Extra Layer Norm suddenly spikes $10 \times$ higher than the FFN gradients a few steps before the loss explodes, you have mathematically indicted the extra Layer Norm as the source of the instability.
        
- **Residual Growth Ratio:**
    
    - _What to track:_ $\| \text{Block}(x) \| / \| x \|$ at each repeat.
        
    - _What it proves:_ Does the model rely on massive residual updates in the later loops to reach an answer? If this ratio spikes, the tied weights are acting like a chaotic amplifier.
    
## ideas

### Issues and Ideas

#### Confounding variables in table 2
In table 2 vanilla CF perp = 24.48
when they add reserved layers perp goes up, which probably was the reason they added layer norm. They justify this by saying 
>"note that while the accuracy does not improve, the computation cost decreases since the total number of layers is kept fixed at 24"

But how does that make sense? They didn't just add reserve layers they shrunk the allocated reasoning space from 24 layers to 21 layers. We can't know whether perplexity got worse because of smaller reasoning space or no LN. This is why we need to do ablations on this:

(the numbers are there just to illustrate a point as long as it "learns" we can make it smaller)

1. **Baseline (The Control):** `24x5` (No reserved layers, no extra LN).
    
2. **Exp 1 (Pure Reserved Layer Test):** `2 -> 24x5 -> 1` (No extra LN). _This answers: Does separating the reasoning space inherently help or hurt?_
    
3. **Exp 2 (The Paper's Confounded Test):** `2 -> 21x5 -> 1` (No extra LN). _This verifies their 24.51 perplexity claim._
    
4. **Exp 3 (Pure LN Test):** `24x5` + Layer Norm (No reserved layers). _This answers: Is the Layer Norm just generally good for recurrent loops, or is it specifically needed to fix the reserved layers?_
    

Then they also say that LN suffers from "benign spikes during training". If they tamed the model by strapping LN's everywhere to control explosions, were the explosions caused reserved layers? If so why did the LN suffer from "benign spikes during training"?


#### The convexity/interpolation issue


They give the router score as 
$$x^{(i+1)} := (1 - s_i) \cdot x^{(i)} + s_i \cdot B(x^{(i)})$$
This is basically a linear interpolation. If a token halts $s_i$ is close to zero. that would mean $x^{(i+1)} \approx x^{(i)}$. So if token A halts at loop 2 token A at loop 2 is also at loop 3 4 5 ... You could assume that attention just sorts this out. But given everything we've seen I don't think so. If the attention mechanism looks at all past tokens by the time you reach loop 5 attention mechanism is full of token A's clones. When you compute attention scores for a token other than A what happens? Does it just learn to split the scores between the copies? What kind of logic does it learn?

The authors explicitly claim that their interpolation, $x^{(i+1)} := (1-s_i)\cdot x_i + s_i \cdot B(x^{(i)})$, acts as a safety net. They state it "provides a way to the model to ensure that increasing capacity will not hurt the performance" because a low router score ($s_i$) ensures the representations of those tokens "remain unchanged". Furthermore, they claim CoTFormer naturally solves the problem of attending to halted tokens because "a halted token is already represented when invoking the attention mechanism".

I think the main selling point of the paper is that they found a solution to the problem of carrying forward past representations in universal transformer variants. If we can prove that this is not really the case, that would be huge.

In BUT if a token halts you just copy the representation forward. But if $s_i$ is 0 this just does the same thing but only possibly worse?

***What should we do about this then?

1) We can track the tokens that halt early. During later loops we will check the attention matrix. Does the model learn to put %100 of its score on the first instance or does it distribute the attention in some nonsensical way? If it does the "nonsensical" thing the MoR router is actively confusing the model. This might be why the adaptive model requires longer training. 
2) We run a forward pass on a trained variant. We find tokens with low scores early. We track ==attention weights only== of the final repeat 5. If the weights are neatly focused on the first instance of the halted token then yeah fair. BUT if attention is smeared (high entropy) the interpolation mechanism is actively confusing the attention heads.
3) In section 5 they admit there is a noticeable performance gap: an adaptive CoTFormer achieves a perplexity of 23.83, while a non-adaptive model (fixed 5 repeats) achieves 23.19, even when given the same compute budget. They hypothesise this is due to "reduced amount of gradient information reaching higher number of repeats". If the attention entropy experiments show smearing we can strongly argue that the performance gap is not a gradient flow issue but is a product of the interpolationn mechanism polluting the kv cache.

>While the above performance is remarkable, we can observe a gap between an adaptive CoTFormer and a non-adaptive CoTFormer trained with exactly 5 repeats even when the adaptive variant is allowed the same amount of compute at inference. For example, after 60k steps, the former reaches perplexity 23.83 while the latter achieves 23.19. One possible reason is the reduced amount of gradient information reaching higher number of repeats in the adaptive training since a good portion of tokens will halt before reaching those repeats. As such, we conjecture that the adaptive CoTFormer would benefit more from longer training. We verify this in Figure 5 where we plot the ratio of different values of router weights for the final repeat when the model is trained for 40k steps and compare it with training for 60k steps. We can clearly see that the model starts to favor using the final repeat more when the model is trained for longer. We note that training time of an adaptive model is significantly lower than training directly at the maximum number of repeats. For example, when training the model with fixed 5 repeats, the training takes roughly around 1.5x longer.

- The routing mechanism uses a Top-K sorting function. This is non-differentiable. If a token gets a low score and is not selected for the Top-K, it bypasses the block entirely, meaning **zero gradients flow back to update its router score**.
-  To prevent tokens from getting permanently stuck in this "dead zone", the authors use **Random Capacity Sampling** during training (randomly setting the capacity to 100% sometimes). This violently shoves stuck tokens through the block to force gradients to flow. _This is a crucial detail to explain why their training is so unstable and takes 60k steps._

base 2 ->4->5
***Does it not make sense to compute some sort of similarity measure for the tokens just to see if they are changing?

4) I looked into this. It's a whole thing of its own. In such high dimensions simple measures like cosine similarity fall apart. There are some methods that seem promising but they do require quite a bit of work. IMO our best option if we want to do such a thing is logit lens. It doesn't require us to train complex probes, to SVD on a billion activations etc. My suggestion is furing the forward pass lets just take the tokens representation at repeat 1 2 3 4 5 etc. Then shove that into the lm head. Maybe pass it through the layer norm first (this is where it gets kinda dodgy imo). One could criticise this heavily but imo if we show that it effectively does represent "reasoning space" in a no LN variant maybe it might make sense? This part kinda confuses me we can talk about this more later. I think that if we applied this to a 2 -> 21x5->1 model (the thing i don't agree with from table 2) the outputs would be nonsense. You would assume middle layers have nothing to do with lm head if the "reasoning subspace" is a reasoning subspace. Idk though i guess we will see.

***What about figure 5?
  
  
  ![[Pasted image 20260327233555.png]]
  
  
  5) I think they show this to say: "When you just train it for more steps the density of 0 weight predictions decrease. It was gradient starvation all along trust me" But even at 60k steps the distribution seems to indicate 55% of tokens are still getting a router score of 0 for the final repeat. This means over half of the cache is made up of redundant tokens. They present this to say the adaptive model just needs more training. But I think that the reason the model is fighting so hard to keep tokens out of the final repeat is not gradient starvation. I think it is because they shoved everything in the KV cache. This would cause the attention matrix to become a mess imo. I think the router literally learns to avoid the final layer.


#### Does the depth embedding work as advertised?

1) To figure out if the depth embedding actually help convert representations we can freeze the index to 4,4,4,4,1 to see if the representations shift. We can reuse the logit lens here. Take the tokens in the middle layers and push them through the LM head.  See if it makes sense
2) Instead of just looking at what word it predicts we can look at the entropy of the softmax distribution. We need to control for some things though and have solid baselines for entropy in the normal case etc

*Why I think we should do these instead of MHLA?*
I think MHLA is a good idea. I think that it would benefit this model, and it would be a huge flex and it would grab a lot of attention. ***But*** it screams premature optimisation. And it's an engineering optimisation (which I don't like because in all of our modules we are encouraged to be scientific). CoTFormer an untested, very new and innovative model. It has basically zero documentation, thousands of lines of spaghetti research code etc. Before we fully understand why it "works" and why it doesnt work, I can't justify for myself slapping on MHLA . Additionally it would require rewriting the custom PyTorch InPlaceSetSlice autograd functions and the horizontallt tilted causal mask. It's a massive engineering challenge. Im not even gonna go in to how that affects tokenization (I probably couldn't). If we managed to do it it would be hugely rewarding but like I said its difficult to do. IMO the main thing they want us to do in this exercise is to "be scientific".  Ablations require writing minimal code, our main bottleneck would be waiting for training runs to end, and interpreting results. Even if the results don't come as expected, I find the questions interesting enough, and I have a feeling that they will too. I also think it aligns better with Antonia and John's seminar lectures and general thinking style, and what they expect from this assignment. Also it would allow us to use the things they literally teach.

When it comes to compute we will still use around 12 Blocks, helping us avoid OOM.