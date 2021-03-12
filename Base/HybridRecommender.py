#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Base.BaseRecommender import BaseRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from Base.ALS import ALS

import numpy as np
from numpy import linalg as la
import scipy.sparse as sps
import similaripy


class Hybrid(BaseRecommender):
    RECOMMENDER_NAME = "Hybrid"

    def __init__(self, URM_train, ICM):
        super(Hybrid, self).__init__(URM_train)
        self.ICM = ICM

    def fit(self):

        ICM = similaripy.normalization.bm25plus(self.ICM)
        URM_aug = sps.vstack([self.URM_train, ICM.T])

        # initialize recommenders
        self.ItemCF1 = ItemKNNCFRecommender(self.URM_train)
        self.ItemCF2 = ItemKNNCFRecommender(self.URM_train)
        self.ItemCF4 = ItemKNNCFRecommender(self.URM_train)
        self.RP3BetaY = RP3betaRecommender(URM_aug)

        self.als2 = ALS(URM_aug)
        self.als3 = ALS(URM_aug)

        # fit recommenders
        self.ItemCF1.fit(2237, 1397, "rp3beta", "bm25plus", "bm25", rp3_alpha=0.301146, rp3_beta=0.602561)
        self.ItemCF2.fit(4820, 1998, "rp3beta", "bm25plus", "bm25", rp3_alpha=0.298947, rp3_beta=0.613693)
        self.ItemCF4.fit(2521, 330, "rp3beta", "bm25plus", "bm25", rp3_alpha=0.749552, rp3_beta=0.987574)
        self.RP3BetaY.fit(alpha=0.638822, beta=0.429194, topK=4161, implicit=False, normalize_similarity=False)

        self.h_4 = ItemKNNSimilarityHybridRecommender(self.URM_train, self.ItemCF4.W_sparse, self.RP3BetaY.W_sparse)
        self.h_4.fit(topK=3583, alpha=0.717524)

        self.als2.fit(latent_factors=610, regularization=0.09143425, iterations=10, alpha=44)
        self.als3.fit(latent_factors=418, regularization=0.08360787, iterations=10, alpha=34)

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        item_weights = np.empty([len(user_id_array), 25975])

        for i in range(len(user_id_array)):
            n_interactions = len(self.URM_train[user_id_array[i], :].indices)

            if n_interactions < 5:
                w = self.ItemCF1._compute_item_score(user_id_array[i], items_to_compute)
                item_weights[i, :] = w

            elif n_interactions > 4 and n_interactions < 10:
                w1 = self.ItemCF2._compute_item_score(user_id_array[i], items_to_compute)
                w2 = self.als2._compute_item_score(user_id_array[i], items_to_compute)
                w1 = w1 / la.norm(w1, 2)
                w2 = w2 / la.norm(w2, 2)
                w = w1 * 0.995739 + w2 * 0.997557
                item_weights[i, :] = w

            elif n_interactions > 9 and n_interactions < 50:
                w1 = self.ItemCF1._compute_item_score(user_id_array[i], items_to_compute)
                w2 = self.als3._compute_item_score(user_id_array[i], items_to_compute)
                w1 = w1 / la.norm(w1, 2)
                w2 = w2 / la.norm(w2, 2)
                w = w1 * 0.988447 + w2 * 0.924441
                item_weights[i, :] = w

            else:
                w = self.h_4._compute_item_score(user_id_array[i], items_to_compute)
                item_weights[i, :] = w

        return item_weights