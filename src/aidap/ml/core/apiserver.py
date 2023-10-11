'''
Created on 23-Jun-2023

@author: BLR001
'''
from fastapi import FastAPI
import uvicorn
from aidap.ml.classification.gp2 import trainModel, inference
from aidap.ml.sematicsearch.minilm import SemanticSearch 
from aidap.ml.document_search.QA_flan_t5_base import generate_answer
from aidap.ml.codetranslation.codegen import convert_code
from aidap.ml.codetranslation.XSLT_to_Java.codellama import convert_xslt_to_java
from aidap.ml.sematicsearch.minilm import loadData

app = FastAPI()

@app.get("/Classifier/Train/")
async def trainAPI(filepath, model_name, version):
    return await trainModel(filepath, model_name, version)

@app.get("/Classifier/Inference/")
async def inferenceAPI(query, model_name, version):
    return await inference(query, model_name, version)

@app.get("/increment/")
def add_one(number):
    return int(number) + 1

@app.get("/semanticsearch")
async def SemanticSearchAPI(query):
    return SemanticSearch(query)

@app.get("/Q&A_flan-t5-base/")
async def QAAPI(query:str):
    return generate_answer(query)

@app.get("/text2text/")#, response_class=HTMLResponse)
async def CodegenAPI(src, tgt, query, category, model):
    if model == 'codegen':
        return await convert_code(src, tgt, query, category)
    if (src == 'xslt') & (tgt == 'java') & (model == 'codellama'):
        return await convert_xslt_to_java(src, tgt, query, category)

@app.get("/reloadSearchData")
async def reloadSearchData():
    return loadData()

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == '__main__':
    uvicorn.run(app)
    #uvicorn.run(app, host="0.0.0.0", port=8000)