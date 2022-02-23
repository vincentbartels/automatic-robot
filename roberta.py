from transformers import pipeline
unmasker = pipeline('fill-mask', model='roberta-base')
result = unmasker("Hello I'm from africa and <mask>.")
print(result)