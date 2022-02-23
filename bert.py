from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
result = unmasker("Hello I'm a [MASK] model.")
print(result)