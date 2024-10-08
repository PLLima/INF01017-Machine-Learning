import numpy as np
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
            'Testing Atributes': X,
            'k-Neighbor ID': neighbors_ids.tolist(),
            'Distance': np.round(neighbors_distance, 3).tolist(),
            'Prediction': self.predict(X),
        })
        return neighbors_dataframe

    def get_accuracy(self):
        return self.score(self.testing_atributes, self.testing_target_attributes)

    def get_model_results(self):
        print(self.get_k_neighbors(self.testing_atributes))
        print(f'\nModel Accuracy:\n\n     {self.get_accuracy() * 100}%\n')

for k in range(1, 9, 2):
    original_model = knn_model(n_neighbors=k)
    normalized_model = knn_model(n_neighbors=k)
    original_model.set_training_file('./Original_Data_2Features/TrainingData_2F_Original.txt')
    normalized_model.set_training_file('./Normalized_Data_2Features/TrainingData_2F_Norm.txt')
    original_model.set_testing_file('./Original_Data_2Features/TestingData_2F_Original.txt')
    normalized_model.set_testing_file('./Normalized_Data_2Features/TestingData_2F_Norm.txt')
    print(f'\nModel Predictions for k = {k}:\n')
    print('\nOriginal Data:\n')
    original_model.get_model_results()
    print('\nNormalized Data:\n')
    normalized_model.get_model_results()

normalized_model_2f_undisturbed = knn_model(n_neighbors=5)
normalized_model_2f_undisturbed.set_training_file('./Normalized_Data_2Features/TrainingData_2F_Norm.txt')
normalized_model_2f_undisturbed.set_testing_file('./Normalized_Data_2Features/TestingData_2F_Norm.txt')
normalized_model_2f_disturbed1 = knn_model(n_neighbors=5)
normalized_model_2f_disturbed1.set_training_file('./Normalized_Data_2Features/TrainingData_2F_Norm.txt')
normalized_model_2f_disturbed1.set_testing_file('./Normalized_Data_2Features/TestingData_2F_Norm_Dist1.txt')
normalized_model_2f_disturbed2 = knn_model(n_neighbors=5)
normalized_model_2f_disturbed2.set_training_file('./Normalized_Data_2Features/TrainingData_2F_Norm.txt')
normalized_model_2f_disturbed2.set_testing_file('./Normalized_Data_2Features/TestingData_2F_Norm_Dist2.txt')

print('\nNormalized Data - 2 Factors - Undisturbed:\n')
normalized_model_2f_undisturbed.get_model_results()
print('\nNormalized Data - 2 Factors - Disturbed 1:\n')
normalized_model_2f_disturbed1.get_model_results()
print('\nNormalized Data - 2 Factors - Disturbed 2:\n')
normalized_model_2f_disturbed2.get_model_results()

normalized_model_11f_undisturbed = knn_model(n_neighbors=5)
normalized_model_11f_undisturbed.set_training_file('./Normalized_Data_11Features/TrainingData_11F_Norm.txt')
normalized_model_11f_undisturbed.set_testing_file('./Normalized_Data_11Features/TestingData_11F_Norm.txt')
normalized_model_11f_disturbed1 = knn_model(n_neighbors=5)
normalized_model_11f_disturbed1.set_training_file('./Normalized_Data_11Features/TrainingData_11F_Norm.txt')
normalized_model_11f_disturbed1.set_testing_file('./Normalized_Data_11Features/TestingData_11F_Norm_Dist1.txt')
normalized_model_11f_disturbed2 = knn_model(n_neighbors=5)
normalized_model_11f_disturbed2.set_training_file('./Normalized_Data_11Features/TrainingData_11F_Norm.txt')
normalized_model_11f_disturbed2.set_testing_file('./Normalized_Data_11Features/TestingData_11F_Norm_Dist2.txt')

print('\nNormalized Data - 11 Factors - Undisturbed:\n')
normalized_model_11f_undisturbed.get_model_results()
print('\nNormalized Data - 11 Factors - Disturbed 1:\n')
normalized_model_11f_disturbed1.get_model_results()
print('\nNormalized Data - 11 Factors - Disturbed 2:\n')
normalized_model_11f_disturbed2.get_model_results()