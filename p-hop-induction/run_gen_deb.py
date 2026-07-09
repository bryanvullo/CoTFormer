from generate import InductionHopsFinalAnswerTask
import numpy as np
task = InductionHopsFinalAnswerTask(seq_len=32, char_tokens=4, min_hops=4, max_hops=4, rng=np.random.RandomState(0), sampling_strategy="constructive", avoid_adjacent_repeats=False)

x, y, meta = task.get_tokens(metadata=True)
x_pos = [] 
for i in range(len(x)):
  x_pos.append(i)


print("input: ", "".join(x))
print(f"input:  {x_pos}")
print("labels:", " ".join(y))
print("answer:", meta["answer"])
print("path:", meta["path"])

