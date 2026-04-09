# Investigating Recurrent Inductive Biases in Language Models: A CoTFormer Evaluation

## 1. Theoretical Background: The Missing Inductive Bias in Transformers
The ability to count inductively (inferring that if a rule holds for $N$, it holds for $N+1$) is a fundamental cognitive and algorithmic skill. However, standard Transformer architectures fundamentally lack the inductive bias necessary to perform this task out-of-distribution (OOD).

Unlike traditional RNNs that inherently maintain and update a hidden state through sequential transitions, Transformers process all input tokens in parallel. Because computations are parallelized, sequential dependency must be simulated across the depth of the network (layers) rather than across time. As a result, standard Transformers struggle to generalize algorithmic reasoning to sequences longer than those encountered during training, failing when forced to track out-of-distribution "counter states".

## 2. The Role (and Limitations) of Positional Embeddings
Because self-attention operations are performed in parallel, standard Transformers cannot intrinsically distinguish tokens based on their sequence position. They rely on Positional Embeddings (PEs) to break the symmetry of homogeneous sequences. 

However, empirical evidence shows that PEs often act as a crutch. Instead of learning the underlying algorithmic rule (e.g., $+1$), Transformers use PEs to memorize rigid, absolute mappings from a specific position ID to a specific output. When the model encounters an OOD sequence length, it encounters unfamiliar position embeddings and catastrophically fails. 

**The CoTFormer Hypothesis:** The CoTFormer architecture replaces standard independent layers with recurrent weight-tying across depth. We want to test whether that this vertical recurrence simulates the state-transition dynamics of an RNN, granting the model the true recurrent inductive bias necessary for algorithmic counting and extrapolation, independent of absolute Positional Embeddings.

---

## 3. Experimental Design & Theoretical Justifications

To isolate the effect of *recurrent weight-tying* from the effect of *raw compute depth*, all experiments will utilize a **compute-matched baseline**. We will compare a standard 4-Layer non-recurrent Transformer against a 1-Layer CoTFormer looped 4 times. Both models will execute exactly 4 attention/MLP passes per token, ensuring that any performance delta is strictly due to the architectural inductive bias, not computational budget.

### Experiment 1: The Shifted Start Task (Pure Extrapolation)
* **Objective:** Test the model's ability to extrapolate counting to unseen sequence lengths (OOD Cardinality) without relying on rigid positional memorization.
* **Mechanism:** The model is given a random starting number and must count upward (e.g., `Input: 45 a a a` $\rightarrow$ `Output: 45 46 47 48`). 
* **Parameters:** `MAX_TRAIN_SEQLEN = 50`, `MAX_OOD_SEQLEN = 200`. The starting values in training will be randomized high enough to expose the model to the full vocabulary up to 200.
* **Theoretical Justification:** This task explicitly uncouples sequence length from number vocabulary. By randomizing the start, we prevent the model from memorizing that "Position 10 equals Output 10". Standard Transformers (even at 4 layers with RoPE) suffer massive performance degradation when pushed to 4x their training length. If the CoTFormer succeeds, it proves the recurrent layers successfully maintain a persistent state variable capable of unbounded `+1` incrementation.

### Experiment 2: Modular (Mod 10) Counting (Periodicity)
* **Objective:** Test the model's ability to act as a finite state machine that maintains a perfect rhythmic loop (periodicity) over long distances.
* **Mechanism:** The model must count to a specific base and reset (e.g., `1 2 3 ... 10 1 2 3`) over long, homogeneous sequences of identical tokens.
* **Caveat:** To counteract RoPE's known theoretical inability to break the symmetry of homogeneous sequences, a `<bos>` (Beginning of Sequence) token will be explicitly prepended to all inputs.
* **Theoretical Justification:** Modular counting eliminates the "OOD Vocabulary" problem entirely, as the model only needs to output the numbers 1 through 10. Instead, it tests whether the architecture can maintain a continuous, repeating loop without drifting. A successful OOD result demonstrates that the CoTFormer's depth-recurrence provides the necessary inductive bias to internally simulate deterministic state automata (DSA).

### Experiment 3: Selective Counting (State Tracking)
* **Objective:** Test the model's capacity to maintain and update multiple, independent counter states simultaneously over an extended sequence.
* **Mechanism:** The input consists of a heterogeneous mix of distinct tokens (e.g., $a_1, a_2, a_1, a_3$). The model must output a running tally for each specific token independent of the others. 
* **Data Generation Constraint:** Training data must be explicitly balanced (upweighted) so that the count of every distinct token follows a uniform distribution from $[0, 10]$ to prevent bias towards average values.
* **Theoretical Justification:** This task simulates tracking distinct "variables" within an algorithmic execution. It requires the model to correctly route information, ignoring irrelevant tokens (Token-based Attention) while selectively incrementing the correct running tally. Success here indicates that CoTFormer can manage complex, multi-variable internal states similar to those required in formal algorithm simulation.