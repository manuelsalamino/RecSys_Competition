from implicit.als import AlternatingLeastSquares
import numpy as np

from Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender


class ALS(BaseMatrixFactorizationRecommender):
    name = 'ALS'

    def fit(self, latent_factors, regularization, iterations, alpha):
        model = AlternatingLeastSquares(factors=latent_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        use_gpu=True,
                                        num_threads=0)

        data_confidence = (self.URM_train.transpose().tocsr() * alpha).astype(np.float32)
        model.fit(data_confidence, show_progress=True)

        self.USER_factors = model.user_factors
        self.ITEM_factors = model.item_factors