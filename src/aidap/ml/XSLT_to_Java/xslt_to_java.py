from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import pandas as pd
import os

from fastapi import FastAPI
import uvicorn
import json

model = None

with open("../properties.json", "r") as f:
	data = json.load(f)

model_path = data["codetranslation"]['model_path_codellama2']
prompt_path = data["codetranslation"]['prompt_path']

def get_prompt(category, query):
    with open(prompt_path+category,"r") as f:
        prompt = f.read()
    prompt = prompt.replace("<TEST CODE>", query)
    return prompt

def load_model(models_path):
    model_dir = os.path.join(models_path,"codellam2_7b_hf_model")
    tokenizer_dir = os.path.join(models_path,"codellam2_7b_hf_tknzr")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16
    )

    return model, tokenizer

if model == None:
    model, tokenizer = load_model(models_path)
    model = model.cuda()

app = FastAPI()

def get_llama_for_completion(prompt, max_new_tokens=256, temperature=0.3, do_sample=True):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
    output = model.generate(
      input_ids,
      max_new_tokens=max_new_tokens,
      pad_token_id=tokenizer.eos_token_id,
      temperature=temperature,
      do_sample=do_sample
      )
    output = output[0].to("cpu")
    input_ids = input_ids.to('cpu')
    filling = tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True)
    # prompt.replace("<FILL_ME>", filling)
    return filling

@app.get("/xslttojava")
async def generate_java(category, query):
    prompt = get_prompt(category, query)
    result = get_llama_for_completion(prompt)
    result = result.split('/*')[0]
    for i in result:
        if (result[0].isalnum()):
            break
        else:
            result = result.replace(result[0], "")
    result = result.split(";")[0]+';'
    return result

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)