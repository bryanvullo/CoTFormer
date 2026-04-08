Dear Prof. Hare and Dr. Marcu,

I hope my email finds you well.

As you may be aware of, we are reproducing the COTFORMER: A CHAIN-OF-THOUGHT DRIVEN ARCHITECTURE WITH BUDGET-ADAPTIVE COMPUTATION  COST AT INFERENCE paper for our reproducibility project. It is a weight tied transformer with recurrence, that passes representations forward across depth with each repeat. I have a couple directions of research I wanted to share with you. I apologise if this is too long.

The recurrent inductive bias

I have recently stumbled upon the  LANGUAGE MODELS NEED INDUCTIVE BIASES TO  COUNT INDUCTIVELY paper that was published in ICLR. In this paper the authors show that unlike RNN's which have a recursive inductive bias, Transformer's don't have this bias and hence are unable to count and state track to the same extent due to their parallel nature and positional encoding schemes.


I realise that it is not exactly horizontal, but since CoTFormer explicitly alters the architecture to allow tokens in later loops to attend to the representations of previous tokens from earlier thought steps (in the same head), it is a deliberate attempt to inject the horizontal, time-recurrent mechanics of CoT into a vertical, depth-recurrent architecture. The authors don't seem to mention this explicitly in the paper or conduct experiments to see if it gained better counting and tracking abilities.

I am particularly interested in whether this recurrence can overcome the known failures of RoPE and transformers. Chang and Bisk (inductive biases paper) found RoPE to be fragile on periodic tasks like modular counting, so I plan to test if the CoTFormer's weight-tied loop can learn a generalized update rule that overrides these embedding-based limitations.

The experiments I want to conduct are as follows:


Baseline: 4 layer transformer in accordance with the inductive biases paper

Test: CoTFormer with 1 layer and 4 repeats

Main idea: The models are given a random starting number and must count upward.
Example
Input:45,a,a,a,
Output: 45,46,47,48
We can train the model on sequences up to a length of 50 and shift the starting position to make sure the model sees every number in the vocabulary. Then we test it on sequences up to length 200.


We do this to uncouple sequence length from number vocabulary. We want to prevent the model from memorizing that position 10=output 10. Standard transformers with RoPe struggle with this task at these IND and OOD sequence lengths. So it might be interesting to see how the CoTFormer handles it.


We can also conduct experiments with mod 10 counting to check periodicity. 

Another idea might be to check whether the model can track multiple distinct counts for different tokens. We would make sure the data is balanced as per the inductive biases paper to mitigate bias towards average values.

This is the research direction I am most unsure about, however I find it very interesting and wanted to hear your thoughts on it. I would highly appreciate it if you could tell me if I'm missing something here. Please let me know if I'm completely off base.

Trying to understand the model itself
Furthermore, we think that we could push the intermediate representations (between layers or repeats) through the LM head to see how the representations change (interpreting GPT: the logit lens, blogpost by nostalgebraist), intervene on the depth embeddings to see if the representations change, track residual stream scale to try and understand why they needed to add the extra layer norms etc. We will inform you about these as we conduct them.

While we thought about sparse autoencoders I think that analysing everything might take too long. We are also aware the logit lens might be a bit flawed. We might look into the Tuned Lens if it isn't as reliable as we'd hoped. (Eliciting Latent Predictions from Transformers with the Tuned Lens, Belrose et al)

Additionally I thought that in this table:

The reason for the drop in performance with just the reserved layers might be due to the fact that the "reasoning space" was lessened. This is why I thought it might make sense to test 2->24x5->1 and 27x5 to see how that by itself affects things by itself without the layer norm.

Furthermore, currently the model has no KV cache implemented and the intermediate representations are stored in what we refer to as the "CoT cache" (to avoid confusion) per head. This might hurt practicality, and it might be an omission by the authors, however I think KV cache is a purely inference time autoregressive generation optimization, so I kind of understand where they are coming from. We could write code to combine these both it should be relatively straight forward on the models with no adaptive depth (I'm not so sure about the adaptive depth tokens) but I don't see any immediate value in it for our current evaluation scope. Because our planned counting and interpretability experiments rely on known sequences, we can evaluate them using parallel next-token prediction (a single forward pass) rather than auto-regressive generation, completely bypassing the O(N**2) bottleneck. 
We have a few other ideas in this domain, but I have omitted them here to avoid an information dump and to allow space for your initial feedback.


    3) Architectural extensions
Additionally teammates are researching how to implement Multi Head Latent Attention and manifold constrained hyper connections by DeepSeek onto this model.



Finally, I want to emphasize that our team is aiming to treat this project with the rigor of a potential submission (for example, to a venue like the ML Reproducibility Challenge or a related workshop). Because we want to meet that standard, please do not hesitate to be brutally honest if any of our proposed directions, especially the counting experiments or my stance on the KV cache seem misguided, theoretically flawed, or out of scope. We are highly open to pivoting based on your feedback to ensure we focus our time where it matters most.
Thank you so much for your time and guidance.


Inductive Biases paper: https://openreview.net/forum?id=s3IBHTTDYl
CoTFormer paper: https://openreview.net/forum?id=7igPXQFupX
Logit Lens blogpost: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
Tuned Lens: https://arxiv.org/abs/2303.08112

Kind Regards

Aras Kavuncu