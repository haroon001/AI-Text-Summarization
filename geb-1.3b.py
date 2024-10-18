from transformers import AutoTokenizer, AutoModel
import torch

model = AutoModel.from_pretrained("GEB-AGI/geb-1.3b", trust_remote_code=True).bfloat16() #.cuda()
tokenizer = AutoTokenizer.from_pretrained("GEB-AGI/geb-1.3b", trust_remote_code=True)

query = "你好"
response, history = model.chat(tokenizer, query, history=[])
print(response)
