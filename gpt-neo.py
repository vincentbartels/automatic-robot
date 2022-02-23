from transformers import pipeline, AutoTokenizer
from enum import Enum


class Model(Enum):
    NEO_125M = 'EleutherAI/gpt-neo-125M'
    NEO_1_3B = 'EleutherAI/gpt-neo-1.3B'
    NEO_2_7B = 'EleutherAI/gpt-neo-2.7B'  # to big
    GPT_2 = 'gpt2'


# following Haikus by Matsuo Basho & Yosa Buson

def one_shot():
    return """
Haiku:
In the moonlight,
The color and scent of the wisteria
Seems far away.

Haiku:

"""


def few_shot():
    return """
Haiku:
In the twilight rain
these brilliant-hued hibiscus -
A lovely sunset.

Haiku:
A summer river being crossed
how pleasing
with sandals in my hands!

Haiku:

"""


def zero_shot_with_topics():
    return "Write an english Haiku based on the following topics:\n" \
           "Topics:\n" \
           "- Summer\n" \
           "- Beach\n" \
           "Haiku:"


def zero_shot():
    return "Write an english Haiku:\n" \
           "Haiku:"


input_text = zero_shot()
# input_text = "Write a Haiku in english:\n"
# input_text = "Write an english Haiku:\n"

model_name = Model.NEO_1_3B.value

tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline('text-generation', model=model_name, tokenizer=tokenizer)
result = generator(input_text, do_sample=True, max_length=50)
generated_text = result[0]['generated_text']

print(generated_text)
