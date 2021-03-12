#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender_utils import check_matrix
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from DataReader import DataReader

import scipy.sparse as sps
import similaripy


class ItemKNNCFRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "none"]

    def __init__(self, URM_train, verbose=True):
        super(ItemKNNCFRecommender, self).__init__(URM_train, verbose=verbose)

    def fit(self, topK=50, shrink=100, similarity='cosine', normalization="none", feature_weighting="none",
            rp3_alpha=0.5, rp3_beta=0.5):

        self.topK = topK
        self.shrink = shrink

        reader = DataReader()
        icm = reader.load_icm()

        if normalization == "bm25plus":
            self.URM_train = similaripy.normalization.bm25plus(self.URM_train, axis=1)

        if feature_weighting == "bm25":
            icm = similaripy.normalization.bm25(icm, axis=1)

        matrix = sps.hstack((self.URM_train.transpose().tocsr(), icm))

        if similarity == "cosine":
            self.W_sparse = similaripy.cosine(matrix, k=self.topK, shrink=self.shrink, binary=False, threshold=0)
        if similarity == "dice":
            self.W_sparse = similaripy.dice(matrix, k=self.topK, shrink=self.shrink, binary=False, threshold=0)
        if similarity == "rp3beta":
            self.W_sparse = similaripy.rp3beta(matrix, k=self.topK, shrink=self.shrink, binary=False, threshold=0,
                                               alpha=rp3_alpha, beta=rp3_beta)

        self.W_sparse = check_matrix(self.W_sparse, format='csr')