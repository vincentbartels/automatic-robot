from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6B')
inputs = tokenizer('Hello my Dog is cute', return_tensors='pt')

print('start model interference')

outputs = model(**inputs, labels=inputs['input_ids'])
loss = outputs.loss
logits = outputs.logits

print(f'outputs: {outputs}')
print(f'loss: {loss}')
print(f'logits: {logits}')
