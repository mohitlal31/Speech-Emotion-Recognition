from sklearn.base import ClassifierMixin
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from data_generator import DataGenerator
from data_preprocessor import DataPreprocessor
from data_visualizer import DataVisualizer
from model_trainer import ModelTrainer


def classify(model: ClassifierMixin, model_name: str, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("{} classification:".format(model_name))
    print(classification_report(y_test, y_pred))

def classify_with_machine_learning(x_train, y_train, x_test, y_test):
    models = {"SVC": SVC(), "KNN": KNeighborsClassifier(n_neighbors=40), "LDA": LinearDiscriminantAnalysis()}

    for model_name, model in models.items():
        classify(model, model_name, x_train, y_train, x_test, y_test)


# Generate a dataframe extracting the various attributes from the file names
data_generator = DataGenerator()
data = data_generator.create_dataframe_from_filenames()

# Plot the distribution of the attributes
data_visualizer = DataVisualizer(data)
# data_visualizer.plot_data_distribution(['Emotions', 'Sex', 'Ethnicity', 'Race'])
# data_visualizer.plot_waveplot_spectrogram()

# Extract features from each audio sample and create a features file
data_preprocessor = DataPreprocessor(data)

feature_file = 'features.csv'
data_preprocessor.create_features_file(feature_file, augment_features=False)

augmented_feature_file = 'features_augmented.csv'
data_preprocessor.create_features_file(augmented_feature_file, augment_features=True)

# # Classification with Machine Learning on Augmented features
# # LabelEncode, split and normalize the data
# x_train, y_train, x_test, y_test = data_preprocessor.prepare_training_data(augmented_feature_file)
# print("Classification using augmented features")
# classify_with_machine_learning(x_train, y_train, x_test, y_test)

# # Classification with Machine Learning on Regular features
# # LabelEncode, split and normalize the data
# x_train, y_train, x_test, y_test = data_preprocessor.prepare_training_data(feature_file)
# print("Classification using normal features")
# classify_with_machine_learning(x_train, y_train, x_test, y_test)

# # Classification with Multilayer Perceptron on Regular features
# x_train, y_train, x_test, y_test = data_preprocessor.prepare_training_data(feature_file, one_hot_encode=True)
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# mlp = ModelTrainer(x_train, y_train, x_test, y_test)
# mlp.build_mlp()
# mlp.train_model(num_epochs=50)
# mlp.classify()
# mlp.plot_loss_accuracy()

# # Classification with Multilayer Perceptron on Augmented features
# x_train, y_train, x_test, y_test = data_preprocessor.prepare_training_data(augmented_feature_file, one_hot_encode=True)
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(x_train.mean(), y_train.mean(), x_test.mean(), y_test.mean())
# print(x_train.std(), y_train.std(), x_test.std(), y_test.std())
# mlp = ModelTrainer(x_train, y_train, x_test, y_test)
# mlp.build_mlp()
# mlp.train_model(num_epochs=50)
# mlp.classify()
# mlp.plot_loss_accuracy()

# # Classification with 1D CNN on Regular features
# x_train, y_train, x_test, y_test = data_preprocessor.prepare_training_data(feature_file, one_hot_encode=True)
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# x_train = np.expand_dims(x_train, axis=2)
# x_test = np.expand_dims(x_test, axis=2)
# cnn = ModelTrainer(x_train, y_train, x_test, y_test)
# rlrp = ReduceLROnPlateau(factor=0.2, verbose=0, patience=4, min_lr=0.000001)
# earlyStopping = EarlyStopping(patience=10, restore_best_weights=True)
# cnn.build_cnn()
# cnn.train_model(num_epochs=100, callbacks=[earlyStopping, rlrp])
# cnn.classify()
# cnn.plot_loss_accuracy()

# Classification with 1D CNN on Augmented features
x_train, y_train, x_test, y_test = data_preprocessor.prepare_training_data(augmented_feature_file, one_hot_encode=True)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
cnn = ModelTrainer(x_train, y_train, x_test, y_test)
rlrp = ReduceLROnPlateau(factor=0.2, verbose=0, patience=4, min_lr=0.000001)
earlyStopping = EarlyStopping(patience=20, restore_best_weights=True)
cnn.build_cnn()
cnn.train_model(num_epochs=100, callbacks=[earlyStopping, rlrp])
cnn.classify()
cnn.plot_loss_accuracy()
