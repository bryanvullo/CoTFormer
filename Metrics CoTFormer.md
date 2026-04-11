# CoTFormer Diagnostic Metrics Suite

To unconfound the ablation studies and peek inside the "black box" of the reasoning loops, we implemented a custom, highly memory-efficient tracking suite. We track five primary categories of metrics.

### 1. Step-Wise Gradient Flow (The "Vanishing Gradient" Check)

- **What we are tracking:** The magnitude (size) of the learning signal as it flows backward from the final answer through each of the 5 reasoning repeats.
    
- **How often:** Captured only on the final micro-batch of an evaluation step (e.g., every 10 steps).
    
- **How we track it:** We plant "backward hooks" on the intermediate hidden states (`x`) during the forward pass. When PyTorch runs `loss.backward()`, these hooks intercept the gradient tensor, calculate its length (L2 norm), save it to `raw_model.backward_metrics`, and immediately delete the tensor to save VRAM.
    
- **Why we care:** Weight tying in CoTFormer can cause the learning signal to die out before reaching the early loops. If Repeat 5 gets a gradient of `2.0` but Repeat 1 gets a gradient of `0.001`, we have empirical proof of vanishing gradients.
    

### 2. Step-Wise Convergence (The "Thought Maturation" Check)

- **What we are tracking:** How much the representation of the final token changes between one loop and the next (Cosine Distance = 1.0 - Cosine Similarity).
    
- **How often:** Calculated on every evaluation step.
    
- **How we track it:** Inside the `forward` function, we take the last token's representation at the end of Loop 2, compare it to the end of Loop 1, and save the distance. We repeat this down the chain and save it to `raw_model.forward_metrics`.
    
- **Why we care:** This measures if the model has "made up its mind." If the distance drops to `0.000` by Repeat 4, it means Repeats 4 and 5 are completely redundant. This provides the mathematical justification for "Adaptive Computation Time" (allowing the model to exit early).
    

### 3. Pre-LayerNorm Variance (The "Exploding Math" Check)

- **What we are tracking:** How "wide" or extreme the numbers in the hidden state tensor get before they are squashed by a LayerNorm.
    
- **How often:** Calculated on every evaluation step.
    
- **How we track it:** Right at the end of each reasoning loop, we calculate `.var().mean()` on the final token's hidden state and save it alongside the convergence metrics.
    
- **Why we care:** Residual connections ($x = x + f(x)$) cause numbers to grow larger with depth. If variance explodes, the attention mechanism breaks (the softmax gets saturated and acts like a rigid one-hot vector). Tracking this proves exactly _why_ the authors needed to add LayerNorms between repeats to maintain good perplexity.
    

### 4. Macro Stream Geometry (The "Summary" Check)

- **What we are tracking:** The total start-to-finish change across the entire reasoning block.
    
- **How often:** Calculated on every evaluation step.
    
- **How we track it:** We clone the tensor right _before_ Repeat 1, and compare it to the tensor right _after_ Repeat 5. We calculate total cosine similarity, entering variance, and exiting variance.
    
- **Why we care:** This acts as our baseline "sanity check." It tells us if the entire CoT sequence actually did anything at all, acting as a macro-summary for the micro-metrics happening inside the loops.
    

### 5.  Thinking Metrics: Attention Dynamics

To understand exactly *how* the CoTFormer is "thinking" and using its reasoning loops, we reshape the massive attention matrix into 5 distinct chunks (one for each repeat). From this reshaped data, we track four core attention metrics.

### 1. Repeat Budget (`macro_budget`)
* **What we are tracking:** How much total attention the model gives to each specific reasoning loop.
* **How often we are tracking:** Calculated on every evaluation step.
* **How we are tracking:** We sum up all the attention weights that point to tokens inside a single repeat. For example, if the total attention is 100%, we calculate what percentage went to Repeat 1, Repeat 2, etc.
* **Why we are tracking:** It reveals the model's "memory horizon." It tells us iff the model only cares about its most recent thought (putting all its budget into Repeat 4) or if it actively retrieves data from the very beginning of its reasoning process (Repeat 1).

### 2. Repeat Entropy (`macro_rep_entropy`)
* **What we are tracking:** How "focused" or "spread out" the model's attention is across the different loops.
* **How often we are tracking:** Calculated on every evaluation step.
* **How we are tracking:** We calculate the mathematical entropy (`-sum(p * log(p))`) of the Repeat Budgets. 
* **Why we are tracking:** It measures decisiveness. Low entropy means an attention head has laser-focused on one specific past loop (specialization). High entropy means the head is "smearing" its attention across all loops to gather a general summary.

### 3. Within-Repeat Entropy (`macro_in_entropy`)
* **What we are tracking:** Once the model decides to look at a specific repeat, how focused is it on the individual words/tokens *inside* that repeat?
* **How often we are tracking:** Calculated on every evaluation step.
* **How we are tracking:** We isolate the attention given to a single repeat, normalize it so it equals 100%, and calculate the entropy across just those 256 tokens.
* **Why we are tracking:** It differentiates between *summarization* and *precise retrieval*. High entropy means it is skimming the whole sequence for a general vibe. Low entropy means it is acting like a pointer network, hunting for the exact value of one specific token.

### 4. Same-Position Budget (`macro_same_pos`)
* **What we are tracking:** How much a token looks at *itself* in previous reasoning loops (e.g., Token #42 looking at Token #42 from Repeat 2).
* **How often we are tracking:** Calculated on every evaluation step.
* **How we are tracking:** We extract the diagonal of the reshaped attention matrix, which perfectly isolates instances where the Query position index matches the Key position index.
* **Why we are tracking:** This proves the core thesis of the paper. If this number is high, the model is acting like a Recurrent Neural Network (RNN)—the token is independently updating its own state loop after loop. If it's low, the token is actively reading the thoughts of *other* tokens to figure out the answer.

---

### 🛠 System Design Notes (For the Report)

All of the above metrics were carefully designed to run on 24GB L4 GPUs **without causing Out-Of-Memory (OOM) crashes**.

- The heavy tensor math is reduced to simple Python floats (`.item()`) instantly.
    
- The dictionaries are explicitly cleared (`.clear()`) after every WandB log.
    
- The backward hooks are shielded behind a `log_metrics` flag so they do not pollute the gradient accumulation microsteps, ensuring zero impact on training throughput.