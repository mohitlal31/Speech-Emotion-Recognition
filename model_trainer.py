from matplotlib.pyplot import hist
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class ModelTrainer():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.model = None
        self.model_name = None
        self.history = None
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
    def train_model(self, num_epochs=100, callbacks:list=[EarlyStopping(patience=5)]):
        history=self.model.fit(self.x_train, self.y_train, batch_size=64, validation_split=0.25, epochs=num_epochs, callbacks=callbacks)
        # print("Stopped Epoch = ", earlyStopping.stopped_epoch)
        self.history = history
        self.model.save('../Saved_models/{}'.format(self.model_name))
        return history
    
    def build_mlp(self):
        model=Sequential()
        model.add(Dense(units=256, activation='relu', input_dim=self.x_train.shape[1]))
        model.add(Dropout(0.2))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=6, activation='softmax'))

        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt , loss='categorical_crossentropy' , metrics=['accuracy'])
        self.model = model
        self.model_name = 'Multilayer Perceptron'
        return model

    def build_cnn1D(self):
        model_cnn=Sequential()
        model_cnn.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(self.x_train.shape[1], 1)))
        model_cnn.add(BatchNormalization())
        model_cnn.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
        # model_cnn.add(Dropout(0.1))

        # model_cnn.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
        # model_cnn.add(BatchNormalization())
        # model_cnn.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
        # model_cnn.add(Dropout(0.4))

        # model_cnn.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
        # model_cnn.add(BatchNormalization())
        # model_cnn.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
        # model_cnn.add(Dropout(0.4))

        model_cnn.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
        model_cnn.add(BatchNormalization())
        model_cnn.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
        # model_cnn.add(Dropout(0.1))

        model_cnn.add(Flatten())
        model_cnn.add(Dense(units=32, activation='relu'))
        # model_cnn.add(Dropout(0.1))

        model_cnn.add(Dense(units=6, activation='softmax'))
        model_cnn.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

        self.model = model_cnn
        self.model_name = 'CNN-1D'

        return model_cnn
    
    def build_cnn2D(self):
        model_cnn=Sequential()
        model_cnn.add(Conv2D(128, kernel_size=5, strides=2, padding='same', activation='relu', \
            input_shape=(self.x_train.shape[1], self.x_train.shape[2], 1)))
        model_cnn.add(BatchNormalization())
        model_cnn.add(MaxPooling2D())
        model_cnn.add(Dropout(0.3))

        # model_cnn.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
        # model_cnn.add(BatchNormalization())
        # model_cnn.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
        # model_cnn.add(Dropout(0.4))

        model_cnn.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
        model_cnn.add(BatchNormalization())
        model_cnn.add(MaxPooling2D())
        model_cnn.add(Dropout(0.3))

        # model_cnn.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
        # model_cnn.add(BatchNormalization())
        # model_cnn.add(MaxPooling2D())
        # model_cnn.add(Dropout(0.4))

        model_cnn.add(Flatten())
        model_cnn.add(Dense(units=32, activation='relu'))
        model_cnn.add(Dropout(0.3))

        model_cnn.add(Dense(units=6, activation='softmax'))
        model_cnn.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

        self.model = model_cnn
        self.model_name = 'CNN-2D'

        return model_cnn

    def build_lstm(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(None, 13), return_sequences=True))
        model.add(LSTM(64))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(6, activation='softmax'))

        model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
        self.model = model
        self.model_name = 'LSTM_partial'
        return model


    def classify(self):
        # predicting on test data.
        y_pred = self.model.predict(self.x_test)
        print("Accuracy of our model on test data : " , self.model.evaluate(self.x_test, self.y_test)[1]*100 , "%")
    
    def plot_loss_accuracy(self):
        print(self.model.metrics_names)
        epochs = [i for i in range(len(self.history.history['loss']))]
        fig , ax = plt.subplots(1,2)
        train_acc = self.history.history['accuracy']
        train_loss = self.history.history['loss']
        test_acc = self.history.history['val_accuracy']
        test_loss = self.history.history['val_loss']

        fig.set_size_inches(20,6)
        ax[0].plot(epochs , train_loss , label = 'Training Loss')
        ax[0].plot(epochs , test_loss , label = 'Validation Loss')
        ax[0].set_title('Training & Validation Loss')
        ax[0].legend()
        ax[0].set_xlabel("Epochs")

        ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
        ax[1].plot(epochs , test_acc , label = 'Validation Accuracy')
        ax[1].set_title('Training & Validation Accuracy')
        ax[1].legend()
        ax[1].set_xlabel("Epochs")

        plt.suptitle(self.model_name)
        plt.show()



    