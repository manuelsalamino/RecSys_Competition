from tqdm import tqdm
import numpy as np

def precision(recommended_items, relevant_items):

    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


def recall(recommended_items, relevant_items):

    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


def MAP(recommended_items, relevant_items):

    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def evaluate(self, recommender, user_list):
    
    mean_average_precision_final = 0
    recommender.fit(recommender.URM_train)
    for target_user in tqdm(user_list):
        recommended_items = recommender.recommend(self.training_urm, target_user, 10)
        relevant_items = self.test_urm[target_user].indices
        mean_average_precision = utils.mean_average_precision(recommended_items, relevant_items)
        mean_average_precision_final += mean_average_precision
    mean_average_precision_final /= len(self.target_users)
    return mean_average_precision_final