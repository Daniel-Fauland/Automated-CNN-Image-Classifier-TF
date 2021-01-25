import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import tensorflow as tf
import time
from tensorflow.keras import layers, models
from datetime import datetime
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
    def model(self, dimx, dimy, channels, num_l, num_n, strides_n, activation, pool_layers, m_pool, dim_out, dropout_1,
              dropout_2, num_hidden_l, num_hidden_n, hidden_activation, x_train, x_val, y_train, y_val,
              epochs, batch_size, predefined):
        # ---Model configuration---
        model = models.Sequential()
        model.add(layers.Conv2D(num_n[0], strides_n[0], activation=activation[0], input_shape=(dimx, dimy, channels)))
        if pool_layers[0] == "y":
            model.add(layers.MaxPooling2D(m_pool[0]))
            pool = 0
        else:
            pool = -1

        try:
            for i in range(num_l-1):
                model.add(layers.Conv2D(num_n[i+1], strides_n[i+1], activation=activation[i+1]))
                if pool_layers[i+1] == "y":
                    pool += 1
                    model.add(layers.MaxPooling2D(m_pool[pool]))

            model.add(layers.Flatten())
            if dropout_1 != 0:
                model.add(layers.Dropout(dropout_1))

            for i in range(num_hidden_l):
                model.add(layers.Dense(num_hidden_n[i], activation=hidden_activation[i]))

            if dropout_2 != 0:
                model.add(layers.Dropout(dropout_2))

            model.add(layers.Dense(dim_out))  # Num neurons in last layer will always depend on the number of your categories
            # ---End of model configuration---

            if predefined != "y":
                with open('python/model_summary.txt', 'w') as ms:  # save model summary in txt file
                    model.summary(print_fn=lambda x: ms.write(x + '\n'))
            else:
                with open('python/predefined_model_summary.txt', 'w') as ms:  # # save predefined model summary in txt file
                    model.summary(print_fn=lambda x: ms.write(x + '\n'))

            model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        except:
            print("="*100)
            print("ERROR! You used to many 'Conv2D' and/or 'MaxPooling' layer which led to a negative output shape.\nTry to reduce the amount "
                  "of 'Conv2D' and/or 'MaxPooling' layers.")
            print("="*100)
            sys.exit(1)

        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))

        now = datetime.now()
        timestamp = now.strftime("%d_%m_%Y__%H_%M_%S")
        model.save(self.checkpoint_dir + "/model_" + timestamp + ".h5")  # save model with current timestamp
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
            duration = "The total runtime was {} seconds".format(round(duration, 2))  # get runtime in seconds
        else:
            duration = "The total runtime was {} minutes".format(round(duration / 60, 2))  # get runtime in minutes

        print("\n========================================================================")
        print("The highest acc ({}%) on the validation data was achieved in epoch {}".format(acc, epoch_acc))  # print highest val acc
        print("The lowest loss ({}) on the validation data was achieved in epoch {}".format(loss, epoch_loss))  # print lowest val loss
        print(duration)
        print("========================================================================")

        # --- plot a graph showing the accuracy over the epochs ---
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()

        # --- plot a graph showing the loss over the epochs ---
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim([0, 1])
        plt.legend(loc='upper right')
        plt.show()


    # ============================================================
    def train_model(self, x_train, x_val, y_train, y_val, dimx, dimy, dim_out, settings):
        s_time = settings["s_time"]
        # --- process all necessary user inputs for the model ---
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

        # Model layers options
        num_n = []
        strides_n = []
        m_pool = []
        activation = []
        num_l = settings["count_layers"]
        pool = 0
        pool_layers = settings["pooling_layers"]
        for i in range(num_l):
            if i == 0:
                if settings["num_neurons_" + str(i+1)] == "":
                    num_n.append(32)
                else:
                    num_n.append(int(settings["num_neurons_" + str(i+1)]))
            else:
                if settings["num_neurons_" + str(i+1)] == "":
                    num_n.append(64)
                else:
                    num_n.append(int(settings["num_neurons_" + str(i+1)]))

            if settings["strides_neurons_" + str(i+1)] == "":
                strides_n.append((3, 3))
            else:
                s1 = int(settings["strides_neurons_" + str(i+1)].split(' ')[0])
                s2 = int(settings["strides_neurons_" + str(i+1)].split(' ')[1])
                strides_n.append((s1, s2))

            if pool_layers[i] == "y":
                pool += 1
                if settings["max_pool_" + str(pool)] == "":
                    m_pool.append((2, 2))
                else:
                    p1 = int(settings["max_pool_" + str(pool)].split(' ')[0])
                    p2 = int(settings["max_pool_" + str(pool)].split(' ')[1])
                    m_pool.append((p1, p2))

            if settings["activation_type_" + str(i+1)] == "2":
                activation.append("sigmoid")
            elif settings["activation_type_" + str(i+1)] == "3":
                activation.append("softmax")
            elif settings["activation_type_" + str(i+1)] == "4":
                activation.append("softplus")
            elif settings["activation_type_" + str(i+1)] == "5":
                activation.append("softsign")
            elif settings["activation_type_" + str(i+1)] == "6":
                activation.append("sanh")
            elif settings["activation_type_" + str(i+1)] == "7":
                activation.append("selu")
            elif settings["activation_type_" + str(i+1)] == "8":
                activation.append("elu")
            elif settings["activation_type_" + str(i+1)] == "9":
                activation.append("exponential")
            else:
                activation.append("relu")

        try:
            if settings["dropout_1"] == "":
                dropout_1 = 0.25
            else:
                dropout_1 = int(settings["dropout_1"]) / 100
        except:
            dropout_1 = 0

        try:
            if settings["dropout_2"] == "":
                dropout_2 = 0.25
            else:
                dropout_2 = int(settings["dropout_2"]) / 100
        except:
            dropout_2 = 0

        num_hidden_n = []
        hidden_activation = []
        num_hidden_l = int(settings["num_hidden_layers"])
        for i in range(num_hidden_l):
            if settings["hidden_layer_" + str(i+1)] == "":
                num_hidden_n.append(64)
            else:
                num_hidden_n.append(int(settings["hidden_layer_" + str(i+1)]))

            if settings["hidden_layer_activation_" + str(i+1)] == "2":
                hidden_activation.append("sigmoid")
            elif settings["hidden_layer_activation_" + str(i+1)] == "3":
                hidden_activation.append("softmax")
            elif settings["hidden_layer_activation_" + str(i+1)] == "4":
                hidden_activation.append("softplus")
            elif settings["hidden_layer_activation_" + str(i+1)] == "5":
                hidden_activation.append("softsign")
            elif settings["hidden_layer_activation_" + str(i+1)] == "6":
                hidden_activation.append("sanh")
            elif settings["hidden_layer_activation_" + str(i+1)] == "7":
                hidden_activation.append("selu")
            elif settings["hidden_layer_activation_" + str(i+1)] == "8":
                hidden_activation.append("elu")
            elif settings["hidden_layer_activation_" + str(i+1)] == "9":
                hidden_activation.append("exponential")
            else:
                hidden_activation.append("relu")
        # --- end of processing all necessary user inputs for the model ---

        predefined = settings["predefined_model"]
        history, model, x_val, y_val = self.model(dimx, dimy, channels, num_l, num_n, strides_n, activation, pool_layers, m_pool, dim_out, dropout_1,
                                                  dropout_2, num_hidden_l, num_hidden_n, hidden_activation, x_train, x_val, y_train, y_val,
                                                  epochs, batch_size, predefined)
        self.results(history, s_time)
