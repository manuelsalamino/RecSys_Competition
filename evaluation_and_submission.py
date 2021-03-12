from DataReader import DataReader
from DataSplitter import DataSplitter
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Base.HybridRecommender import Hybrid

import pandas as pd
import scipy.sparse as sps
import numpy as np
from tqdm import tqdm
import similaripy

def evaluate(urm, ICM):
    URM_train, URM_val, URM_test = splitter.split(urm, testing=0.1, validation=0.2)
    
    evaluator_validation = EvaluatorHoldout(URM_val, [10])
    evaluator_test = EvaluatorHoldout(URM_test, [10])

    recommender = Hybrid(URM_train, ICM)
    recommender.fit()    

    results_run_dict, results_run_string = evaluator_validation.evaluateRecommender(recommender)
    print(results_run_string)
    results_run_dict, results_run_string = evaluator_test.evaluateRecommender(recommender)
    print(results_run_string)

def submission(urm, ICM):
    URM_train = sps.csr_matrix((urm.rating, (urm.user_id, urm.item_id)), shape=(7947, 25975))

    targets = reader.load_target()

    recommender = Hybrid(URM_train, ICM)
    recommender.fit()

    f = open("submission.csv", "w")
    f.write("user_id,item_list\n")
    for t in tqdm(targets):
        recommended_items = recommender.recommend(t, cutoff=10, remove_seen_flag=True)
        well_formatted = " ".join([str(x) for x in recommended_items])
        f.write(f"{t}, {well_formatted}\n")

reader = DataReader()
splitter = DataSplitter()

urm = reader.load_urm()
ICM = reader.load_icm()

submission(urm, ICM)
#evaluate(urm, ICM)

