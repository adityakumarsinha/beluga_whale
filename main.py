import subprocess

subprocess.run(["python", "effb5_main.py"])



from pathlib import Path
import numpy as np
from loguru import logger
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors 
from sklearn.model_selection import StratifiedKFold,GroupKFold
from sklearn.metrics import average_precision_score
import typer

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import torch.nn.functional as F
import timm
import cv2
from types import SimpleNamespace
import albumentations as A
import torch
import torch.nn as nn
import timm
import pickle

import torch.nn as nn


ROOT_DIRECTORY = Path("/code_execution")
PREDICTION_FILE = ROOT_DIRECTORY / "submission" / "submission.csv"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"


        
    
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from sklearn.preprocessing import MinMaxScaler
    
def main():

    query_scenarios = pd.read_csv(DATA_DIRECTORY / "query_scenarios.csv", index_col="scenario_id")
    metadata = pd.read_csv(DATA_DIRECTORY / "metadata.csv", index_col="image_id")
    
    scenario_imgs = []
    for row in query_scenarios.itertuples():
        scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.queries_path).query_image_id.values)
        scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values)
    scenario_imgs = sorted(set(scenario_imgs))
    metadata = metadata.loc[scenario_imgs]
    
    with open('./effnet_b5_embeddings.pkl', 'rb') as handle2:
        effnet_b5_embeddings = pickle.load(handle2)
        

        
    test_embeddings = {}
    
    for k in effnet_b5_embeddings.keys():
        emb = np.concatenate([effnet_b4_embeddings[k]])
        #emb = np.concatenate([effnet_b4_embeddings[k],effnet_b5_embeddings[k], effnet_b6_embeddings[k], eca_nfnet_l2_embeddings[k]])
        #emb = np.concatenate([effnet_b4_embeddings[k],effnet_b5_embeddings[k], effnet_b6_embeddings[k], eca_nfnet_l2_embeddings[k]])
        test_embeddings[k] = emb

    print('emb',emb.shape)
    
    logger.info(f"Precomputed embeddings for {len(test_embeddings)} images")

    logger.info("Generating image rankings")
    # process all scenarios
    results = []
    for row in query_scenarios.itertuples():
        # load query df and database images; subset embeddings to this scenario's database
        qry_df = pd.read_csv(DATA_DIRECTORY / row.queries_path)
        db_img_ids = pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values
        
        db_embeddings = []
  
        for dimgid in db_img_ids:
            db_embeddings.append(test_embeddings[dimgid])

        db_embeddings = np.stack(db_embeddings)

        # predict matches for each query in this scenario
        for qry in qry_df.itertuples():
            query_image_id = qry.query_image_id
        
            # get embeddings; drop query from database, if it exists
            qry_embedding = test_embeddings[query_image_id]
            sims = cosine_similarity(qry_embedding.reshape(1,-1), db_embeddings)[0]
            
            qry_result = pd.DataFrame()
            qry_result['database_image_id'] = db_img_ids
            qry_result['score'] = (sims+1)/2
            qry_result['query_id'] = qry.query_id
            qry_result = qry_result[qry_result.database_image_id != query_image_id]
          
            qry_result = qry_result.sort_values('score',ascending=False).head(20)
            
            #print('qry_result',query_image_id,qry_result.shape)

            results.append(qry_result)

    
    submission = pd.concat(results)
    
    logger.info(f"Writing predictions file to {PREDICTION_FILE}")
    print('submission',submission.head(40))
    
    submission = submission[['query_id','database_image_id','score']]
    
    submission.to_csv(PREDICTION_FILE, index=False)
    


if __name__ == "__main__":
    main()