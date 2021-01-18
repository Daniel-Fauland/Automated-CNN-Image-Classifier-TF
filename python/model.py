import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


class Model():
    def __init__(self, mode):
        self.checkpoint_dir = "./checkpoints"

        if mode == "2":
            # --- prevent TF from using more VRAM than the GPU actually has ---
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)
        elif mode == "3":
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force CPU Usage, instead of GPU


    # ============================================================
    def model(self, dimx, dimy, x_train, x_val, y_train, y_val, epochs, batch_size):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(dimx, dimy, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(43))

        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(x_val, y_val))
        model.save(self.checkpoint_dir + "/model.h5")
        print("training model done.")
        return history, model, x_val, y_val


    # ============================================================
    def results(self, history):
        acc = str(round(max(history.history["val_accuracy"]) * 100, 2))
        epoch_acc = str(history.history["val_accuracy"].index(max(history.history["val_accuracy"])) + 1)
        loss = str(round(min(history.history["val_loss"]), 3))
        epoch_loss = str(history.history["val_loss"].index(min(history.history["val_loss"])) + 1)

        print("\n======================================================================")
        print("The highest acc. ({}%) on the validation data was achieved in epoch {}".format(acc, epoch_acc))
        print("The lowest loss ({}) on the validation data was achieved in epoch {}".format(loss, epoch_loss))
        print("======================================================================")

        # --- plot a graph showing the accuracy over the epochs
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()


    def train_model(self, x_train, x_val, y_train, y_val, dimx, dimy, settings):
        if settings["epochs"] == "":
            epochs = 10
        else:
            epochs = int(settings["epochs"])

        if settings["batch_size"] == "":
            batch_size = 64
        else:
            batch_size = int(settings["batch_size"])

        history, model, x_val, y_val = self.model(dimx, dimy, x_train, x_val, y_train, y_val, epochs, batch_size)
        self.results(history)
