import array
import torch as pytorch

import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)
chat = []
chater = ''.join(str(chat) for x in chat)
print(chater)
while True:
	inputer = input("Me: ")
	chater = ''.join(str(chat) for x in chat)
	print(chater)
	sentence = chater + "Me: "+ inputer + "\n" + "You:"
	input_ids = tokenizer.encode(sentence, return_tensors='pt')
	output = model.generate(input_ids, max_new_tokens=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
	
	print(tokenizer.decode(output[0], skip_special_tokens=True))
	gpt_2_prompt = tokenizer.decode(output[0], skip_special_tokens=True)
	single_liner = result = " ".join(line.strip() for line in gpt_2_prompt.splitlines())
	chat.append(single_liner)
