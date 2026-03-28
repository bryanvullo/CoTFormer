import torch 
from models.tak_custom_cot import GPTBase
from config.base import get_config
# load base
config = get_config()
config.n_layer = 12
config.n_repeat = 5
config.model = "base"

model = GPTBase(config)
model.eval()

x.torch.randint(0,50304, (1, 28))

print("Running forrward pass with ablation hooks")
with torch.no_grad():
	logits, extra_Data = model(x, get_logits=True)

print("forward pass hopefully succesful")
if extra_data:
	print(f"captured {len(extra_data)} loops of Logit lens data")

