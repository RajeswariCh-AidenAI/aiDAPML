from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import os

import json
import requests
#from fastapi import FastAPI
#import uvicorn


torch.cuda.empty_cache()

model = None
prompt = None
generate_prompt_from_csv = True
#convert_series_of_xslt = False


# with open("../properties.json", "r") as f:
with open("properties.json", "r") as f:
	data = json.load(f)

base_path = data["xslt_to_java"]['base_path']
model_path = data["xslt_to_java"]['model_path']
tokenizer_path =  data["xslt_to_java"]['tokenizer_path']
prompt_path = data["xslt_to_java"]['prompt_path']
prompt_input_path = data["xslt_to_java"]['prompt_input_csv_path']
input_data_path = data["xslt_to_java"]['input_csv_path']
output_data_path = data["xslt_to_java"]['output_csv_path']
output_path = data["xslt_to_java"]['output_path']
template_path = data["xslt_to_java"]['template_path']
device = data["xslt_to_java"]['device']


#with open(output_path, 'w') as f:
#        f.write('category, xslt, java')

def get_prompt(src, tgt, category, query):
    with open(f'{src}_to_{tgt}_{prompt_path}_{category}.txt',"r") as f:
        prompt = f.read()
    prompt = prompt.replace("<TEST CODE>", query)
    return prompt

def load_model(model_path, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    )

    return model, tokenizer

def convert_series(src, tgt):
    df = pd.read_csv(f'{src}_to_{tgt}_' + input_csv_path)
    for i in range(len(df)):
       df.loc[i,"java"] = requests.get('http://54.242.243.62:8000/text2text', params = {'src': src, 'tgt':tgt, "query":df.loc[i, 'xslt'],"category":df.loc[i,'category'],  model:'codellama'})
    df.to_csv(output_data_path, index = False)

if model == None:
     model, tokenizer = load_model(model_path, tokenizer_path)
     model = model.to(device)

#app = FastAPI()

def get_llama_for_completion(prompt, max_new_tokens=256, temperature=0.3, do_sample=True):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
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

#@app.get("/getdynamicprompt")
def get_dynamic_prompt(src, tgt, prompt, sample_xslt, java_equivalent):
    split = prompt.split("####")
    text = """/*implement the hierarchy of nodes for <sample_xslt> */
    <java_equivalent>\n\n\n    """
    text = text.replace("<sample_xslt>",sample_xslt)
    text = text.replace("<java_equivalent>",java_equivalent)
    prompt = split[0] + text + "####" +split[1]
    return prompt

def generate_prompt(src,tgt):
    prompt_dict = {}
    df = pd.read_csv(base_path + f'{src}_to_{tgt}_'+prompt_input_path)
    for cat in df['category'].unique():
        with open(base_path + f'{src}_to_{tgt}_'+ template_path,"r") as f:
            prompt = f.read()
        inputs = df[df['category'] == cat]
        for i,r in inputs.iterrows():
            prompt = get_dynamic_prompt(src,tgt,prompt, r[src],r[tgt])
        prompt = prompt.replace("####", "")
        prompt_dict[cat] = prompt
        # with open(base_path + f'{src}_to_{tgt}_{prompt_path}_{cat}.txt',"w") as f:
        #     f.write(prompt)

    return prompt_dict


#@app.get("/xslttojava")
async def convert_xslt_to_java(src, tgt, query, category):
    if generate_prompt_from_csv:
        prompt_dict = generate_prompt(src, tgt)
        prompt = prompt_dict[category]
    else:
        prompt = get_prompt(src, tgt, category, query)
    prompt = prompt.replace("<TEST CODE>",query)
    print(prompt)
    result = get_llama_for_completion(prompt)
    result = result.split('/*')[0]
    for i in result:
        if (result[0].isalnum()):
            break
        else:
            result = result.replace(result[0], "")
    result = result.split(";")[0]
    print(f'{category}, {query}, {result}')
 #   with open(output_path, 'a') as f:
 #       f.write(f'{category}, {query}, {result}')
    return result


# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0", port=8000)

    # inputs = [["/bookstore/book[1]","parent.getBookstore().getbook().get(0)"],["/bookstore/book[1]","parent.getBookstore().getbook().get(0)"],["/bookstore/book[1]","parent.getBookstore().getbook().get(0)"]]
    # query = "/menu/item[1]"
    # for i in inputs:
    #     prompt = requests.get("http://54.242.243.62:8000/getdynamicprompt/", params = {})
    #     with open("dynamic_prompt.txt","w") as f:
    #         f.write(prompt)
    # prompt = prompt.replace("####\n\n", "")
    # prompt = prompt.replace("<TEST CODE>", query)
    # with open("dynamic_prompt.txt","w") as f:
    #         f.write(prompt)
