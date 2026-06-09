import numpy as np

class WeightedEnsemble:
    def __init__(self, models, weights):
        """
        models: list of model instances that have predict(df)
        weights: list of weights summing to 1.0
        """
        assert len(models) == len(weights), "Must provide a weight for each model"
        assert np.isclose(sum(weights), 1.0), "Weights must sum to 1.0"
        
        self.models = models
        self.weights = weights
        
    def fit(self, train_df):
        for model in self.models:
            model.fit(train_df)
            
    def predict(self, test_df):
        ensemble_preds = np.zeros(test_df.height)
        
        for model, weight in zip(self.models, self.weights):
            preds = np.array(model.predict(test_df))
            ensemble_preds += preds * weight
            
        return np.clip(ensemble_preds, 1.0, 5.0)
