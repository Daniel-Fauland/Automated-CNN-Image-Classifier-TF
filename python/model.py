import os
import tensorflow as tf
import time
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


class Model():
    def __init__(self, mode):
        self.checkpoint_dir = "./checkpoints"

        if mode == "2" or mode == "3":
            # --- prevent TF from using more VRAM than the GPU actually has ---
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)
        elif mode == "4":
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force CPU Usage, instead of GPU


    # ============================================================
    def model(self, dimx, dimy, channels, num_n_1, strides_n_1, m_pool_1, dim_out, x_train, x_val, y_train, y_val, epochs, batch_size):
        model = models.Sequential()
        model.add(layers.Conv2D(num_n_1, strides_n_1, activation='relu', input_shape=(dimx, dimy, channels)))
        model.add(layers.MaxPooling2D(m_pool_1))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        #
        model.add(layers.Flatten())
        # model.add(layers.Dropout(0.2))
        # model.add(layers.Dense(64, activation='relu'))
        # model.add(layers.Dropout(0.2))
        model.add(layers.Dense(dim_out))

        with open('python/model_summary.txt', 'w') as ms:
            model.summary(print_fn=lambda x: ms.write(x + '\n'))

        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(x_val, y_val))
        model.save(self.checkpoint_dir + "/model.h5")
        print("training model done.")
        return history, model, x_val, y_val


    # ============================================================
    def results(self, history, s_time):
        acc = str(round(max(history.history["val_accuracy"]) * 100, 2))
        epoch_acc = str(history.history["val_accuracy"].index(max(history.history["val_accuracy"])) + 1)
        loss = str(round(min(history.history["val_loss"]), 4))
        epoch_loss = str(history.history["val_loss"].index(min(history.history["val_loss"])) + 1)

        end_time = time.time()
        duration = end_time - s_time
        if duration <= 60:
            duration = "The total runtime was {} seconds".format(round(duration, 2))
        else:
            duration = "The total runtime was {} minutes".format(round(duration/ 60, 2))

        print("\n=======================================================================")
        print("The highest acc ({}%) on the validation data was achieved in epoch {}".format(acc, epoch_acc))
        print("The lowest loss ({}) on the validation data was achieved in epoch {}".format(loss, epoch_loss))
        print(duration)
        print("=======================================================================")

        # --- plot a graph showing the accuracy over the epochs
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()


    # ============================================================
    def train_model(self, x_train, x_val, y_train, y_val, dimx, dimy, dim_out, settings):
        s_time = settings["s_time"]
        if settings["epochs"] == "":
            epochs = 10
        else:
            epochs = int(settings["epochs"])

        if settings["batch_size"] == "":
            batch_size = 64
        else:
            batch_size = int(settings["batch_size"])

        if settings["channels"] == "2":
            channels = 3
        else:
            channels = 1

        if settings["num_neurons_1"] == "":
            num_n_1 = 32
        else:
            num_n_1 = int(settings["num_neurons_1"])

        if settings["strides_neurons_1"] == "":
            strides_n_1 = (3, 3)
        else:
            s1 = int(settings["strides_neurons_1"].split(' ')[0])
            s2 = int(settings["strides_neurons_1"].split(' ')[1])
            strides_n_1 = (s1, s2)

        if settings["max_pool_1"] == "":
            m_pool_1 = (2, 2)
        else:
            p1 = int(settings["max_pool_1"].split(' ')[0])
            p2 = int(settings["max_pool_1"].split(' ')[1])
            m_pool_1 = (p1, p2)

        history, model, x_val, y_val = self.model(dimx, dimy, channels, num_n_1, strides_n_1, m_pool_1, dim_out, x_train, x_val, y_train, y_val, epochs, batch_size)
        self.results(history, s_time)
