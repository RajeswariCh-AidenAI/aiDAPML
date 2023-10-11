# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:58:50 2023

@author: Rajeswari
"""

from sentence_transformers import SentenceTransformer, util
import mysql.connector as connection
import pandas as pd
import torch
from dataclasses import dataclass

dataLoaded=False
embedder = SentenceTransformer('all-MiniLM-L6-v2')


@dataclass()
class SearchResult:
    """Class for keeping track of an item in inventory."""
    id: int
    name: str    
    score: float = 0

def loadData():
    global corpus, corpus_embeddings,dataLoaded,result_dataFrame
    print("Loading Data from the DB")
    # mydb = connection.connect(host="localhost", database = 'aiDAPTechDTrainer', user="axelra", passwd="axelra", use_pure=True)
    # q = "SELECT * FROM aiDAPTechDTrainer.node_data;"
    mydb = connection.connect(host="localhost", database = 'techDTrainer', user="axelra", passwd="axelra", use_pure=True)
    q = "SELECT * FROM techDTrainer.node where platform='Canonical';;"
    result_dataFrame = pd.read_sql(q,mydb)
    print(result_dataFrame)
    corpus = result_dataFrame['name'].values
    #corpus = result_dataFrame['nodename'].values
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    mydb.close()
    print("Loaded Data from the DB")
    dataLoaded=True

def SemanticSearch(query):    
    global corpus, corpus_embeddings, dataLoaded, result_dataFrame
    print(dataLoaded)
    if dataLoaded == False:
        loadData()
    print(corpus)
    top_k = min(3, len(corpus))        
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)       
    result = []
    for score, idx in zip(top_results[0], top_results[1]):
        print(idx.item())        
        sr = SearchResult(result_dataFrame["id"].iloc[idx.item()].item(),result_dataFrame["name"].iloc[idx.item()],score.item())
        #sr = SearchResult(result_dataFrame["id"].iloc[idx.item()].item(),result_dataFrame["nodename"].iloc[idx.item()],score.item())
        print(sr)
        result.append(sr)
    print(result)
    return result