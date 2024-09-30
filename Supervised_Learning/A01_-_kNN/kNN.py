import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

class knn_model(KNeighborsClassifier):
    def __init__(self, n_neighbors, **kwargs):
        super().__init__(n_neighbors=n_neighbors, **kwargs)

    def set_training_file(self, training_file):
        training_frame = pd.read_csv(training_file, sep='\t')
        self.training_atributes = training_frame.drop(columns=['ID', 'class']).values.tolist()
        self.training_target_attributes = training_frame['class'].values.tolist()
        self.fit(self.training_atributes, self.training_target_attributes)

    def set_testing_file(self, testing_file):
        testing_frame = pd.read_csv(testing_file, sep='\t')
        self.testing_atributes = testing_frame.drop(columns=['ID', 'class']).values.tolist()
        self.testing_target_attributes = testing_frame['class'].values.tolist()

    def get_k_neighbors(self, X):
        neighbors_distance, neighbors_ids = self.kneighbors(X)
        neighbors_dataframe = pd.DataFrame({
            'ID': X,
            'k-Neighbor ID': neighbors_ids.tolist(),
            'Distance': neighbors_distance.tolist(),
            'Prediction': self.predict(X),
        })
        return neighbors_dataframe

    def get_accuracy(self):
        return self.score(self.testing_atributes, self.testing_target_attributes)

    def get_model_results(self):
        print(f'\nModel Predictions with k = {self.n_neighbors}:\n')
        print(self.get_k_neighbors(self.testing_atributes))
        print(f'\nModel Accuracy:\n\n     {self.get_accuracy() * 100}%\n')