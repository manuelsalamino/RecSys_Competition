from Utils.DataSplitter import DataSplitter
from Utils.DataReader import DataReader
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Base.ALS import ALS

import pandas as pd
import numpy as np
from tqdm import tqdm
import similaripy
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize


# return an urm with only interactions of user with n_interactions between min and max (max excluded)

def n_interaction_interval(urm, min_n_interaction, max_n_interaction):
    urm_group = urm.copy()
    profile_length = np.ediff1d(urm_group.indptr)
    #row_mask = np.logical_not(profile_length<5)
    row_mask = np.logical_not(np.logical_and(profile_length>=min_n_interaction, profile_length<max_n_interaction))

    urm_group.data[np.repeat(row_mask, profile_length)] = 0
    # ask scipy.sparse to remove the zeroed entries
    urm_group.eliminate_zeros()
    
    return urm_group

reader = DataReader()
splitter = DataSplitter()

urm = reader.load_urm()
ICM = reader.load_icm()

# DECLARE parameters to tune
@use_named_args([Integer(low=100, high=1000, name='latent_factors'),
                 Real(low=0, high=1, name='regularization'),
                 Integer(low=1, high=200, name='alpha')])

def objective(latent_factors, regularization, alpha):    # parameters must be the same defined above
    average_map = 0.0
    n_tests = 3             # number of tests (on different data split)
    seed = [1234, 12, 34]               # seed to define the split
    
    for i in range(n_tests):
        URM_train, URM_test = splitter.split_train_test(urm, testing=0.15, seed=seed[i])
        URM_test = n_interaction_interval(URM_test, 0, 5)     # maintain only users with a number of interaction between 0 and 5 (excluded)
        
        evaluator_test = EvaluatorHoldout(URM_test, [10])
        
        rec = ALS(URM_train)             # can be used also with other recommenders
        rec.fit(latent_factors=latent_factors, regularization=regularization, iterations=100, alpha=alpha)         # pass the parameter we are tuning
        
        results_run_dict, results_run_string = evaluator_test.evaluateRecommender(rec)
        
        cumulative_MAP = results_run_dict[10]['MAP']
        
        average_map += cumulative_MAP
    
    print(f"\nlatent_factors: {latent_factors}, regularization: {regularization}\navg MAP: {average_map/n_tests}\n\n")
    return -average_map/n_tests                # return the avg_map among the different test (to avoid overfitting on a specific data split)

# DEFINE parameter to tune
space = [Integer(low=100, high=1000, name='latent_factors'),
         Real(low=0, high=1, name='regularization'),
         Integer(low=1, high=200, name='alpha')]

res_gp = gp_minimize(objective, space, n_calls=10, random_state=0, verbose=True)

