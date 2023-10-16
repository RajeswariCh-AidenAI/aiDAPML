'''
Created on 28-Apr-2023

@author: rajagopal
'''
#from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer#,CodeGenModel, CodeGenConfig
#from aidap.ml.classification.gp2 import inference
#from fastapi import FastAPI

import json

#app = FastAPI()
#print(os.listdir("./"))


#with open("./src/aidap/ml/properties.json", "r") as f:
with open("../properties.json", "r") as f:
    data = json.load(f)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
output_path = data["codetranslation"]["output"]
checkpoint = "Salesforce/codegen-350M-mono"
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#if __name__ == '__main__':


async def convert_esql_to_java(src,tgt, query, category):
    print(f"src = {src} tgt={tgt} query = {query} category={category}")
    #category = inference(query, model_name, version)
    
    # filename = "prompt.txt"
    # category = "SET I = I + 1;"
    # checkpoint = "EleutherAI/gpt-j-6B"
   
    #f =  open(f"../{filename}", "r")
    # f =  open(data["codetranslation"]["prompt_path"] + f"{category}_prompt.txt", "a")
    # f.write(f"esql: {query}")
    # f.close()
    #

    f =  open(data["codetranslation"]["prompt_path"] + f"{category}_prompt.txt", "r")
    prompt = f.read()
    f.close()
    
    prompt = prompt + (f": {query}")
    #print(f"prompt = *******************\n{prompt}\n*******************")

    
    f1 = open(data["codetranslation"]["input_prompt"], "r")
    #f1 = open(f"../{filename}", "r")
    esqlCode = f1.read()
    
    
    

    # print(device)
    
    #revision="float16", low_cpu_mem_usage=True
    #configuration = CodeGenConfig()
    # model = CodeGenModel(configuration).to(device)


    end_sequence = "####"
    
    
    promptSize = len(prompt.split("\n"))
    #print(promptSize)
    f = open(output_path, "w")
    code = ""
    c = category
    c = c.strip()
    if c!= "":
        text = prompt + f"\n{tgt}:"
        #print(f"************\n{text}\n******************")
        emb = tokenizer(text, return_tensors="pt").to(device)
        completion = model.generate(**emb,max_new_tokens =50, temperature=0.9, eos_token_id = int(tokenizer.convert_tokens_to_ids(end_sequence)))
        resp = tokenizer.decode(completion[0]).split("\n")
        respLength = len(resp)
        #print(promptSize)
        #print(respLength)        
        for i in range(promptSize,promptSize+1):    
            f.write(resp[i].replace(f"{tgt}:", "")+"\n\r")     
            #code = code + c + "<br>" + resp[i] + "<br><br>"
            code = code + resp[i]
            print(c + "\n" + resp[i]) 
        
    f.close()   
    return code[6:]


    #codeLines = esqlCode.split("\n")
            
    # for c in codeLines:
    #     c = c.strip()
    #     if c!= "":
    #         text = prompt + c +"\njava:"
    #         emb = tokenizer(text, return_tensors="pt").to(device)
    #         completion = model.generate(**emb,max_new_tokens =50, temperature=0.9, eos_token_id = int(tokenizer.convert_tokens_to_ids(end_sequence)))
    #         resp = tokenizer.decode(completion[0]).split("\n")
    #         respLength = len(resp)
    #         print(promptSize)
    #         print(respLength)        
    #         for i in range(promptSize,promptSize+1):    
    #             f.write(resp[i].replace("java:", "")+"\n\r")     
    #             code = code + c + "<br>" + resp[i] + "<br><br>"
    #             print(c + "\n" + resp[i]) 
    # f.close()
    
    # code = code.strip()
    # return code

#print(main('prompt.txt','SET i = i + 1'))
