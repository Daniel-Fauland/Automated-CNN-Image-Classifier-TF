import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ignore all messages
import sys
import cv2
import re
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tabulate import tabulate


class Predict():
    def __init__(self):
        self.path_data = "predict_data"
        self.checkpoint_dir = "checkpoints"
        params = "python/predict_params.csv"
        self.df = pd.read_csv(params)

    # ============================================================
    def get_images(self, shape):
        if os.path.exists(self.path_data + "/insert your own images here that you want to predict.txt"):
            os.remove(self.path_data + "/insert your own images here that you want to predict.txt")

        dimx = shape[1]
        dimy = shape[2]
        images = []
        src_images = []
        data = os.listdir(self.path_data)
        if ".DS_Store" in data:  # Only necessary for MacOS
            os.remove(self.path_data + "/" + ".DS_Store")
            time.sleep(1)
            data = os.listdir(self.path_data)
        for file in data:
            try:
                image = cv2.imread(self.path_data + "/" + file)
                src_img = plt.imread(self.path_data + "/" + file)
                image = cv2.resize(image, (dimx, dimy))
                images.append(image)
                src_images.append(src_img)
            except:
                print("=" * 100)
                print("ERROR! OpenCV could not open the file '{}'\n"
                      "This is probably due to an invalid character in the file name or the file is corrupted in some way.\nRename "
                      "the file and try again.".format(file))
                print("=" * 100)
                sys.exit(1)
        return images, src_images, data

    # ============================================================
    def preprocess_data(self, images, shape):
        def normalize(img, img_normalize, channels):
            if channels == 3:
                pass
            else:
                img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)  # Grayscale image

            # img = cv2.equalizeHist(np.uint8(img))                  # Optimize Lightning

            if img_normalize == "2":
                pass
            else:
                img = img / 255.0  # Normalize px values between 0 and 1
            return img

        img_normalize = str(self.df["img_normalize"][0])
        channels = shape[3]

        for x in range(len(images)):
            images[x] = normalize(images[x], img_normalize, channels)

        images = np.array(images)  # TF needs numpy array
        # Reshape images back to its original form of width * height. (Open-cv flips width and height when resizing an image)
        images = images.reshape(images.shape[0], images.shape[2], images.shape[1], channels)
        return images

    # ============================================================
    def predict_data(self, src_images, predictions, data, file):
        labels_file_name = self.df["csv_name"][0]
        df_labels = pd.read_csv("labels/" + labels_file_name)
        label_names = df_labels[self.df["csv_column"][0]].tolist()
        prediction_list = []
        label_names_str = []
        for label in label_names:
            label_names_str.append(str(label))

        for i in range(len(src_images)):
            image = src_images[i]
            plt.imshow(image)
            plt.title("Prediction: " + label_names_str[np.argmax(predictions[i])])
            prediction_list.append(label_names_str[np.argmax(predictions[i])])
            plt.show()

        df = {"File": data, "Prediction": prediction_list}
        df = pd.DataFrame(df)
        print("\nPredictions for model file:", file)
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))


    # ============================================================
    def initialize(self):
        def sorted_nicely(l):
            """ Sort the given iterable in the way that humans expect."""
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(l, key=alphanum_key)

        if str(self.df["mode"][0]) == "3":
            # --- prevent TF from using more VRAM than the GPU actually has ---
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)
        elif str(self.df["mode"][0]) == "2" or str(self.df["mode"][0]) == "4":
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force CPU Usage, instead of GPU

        # --- load the model using the load_model function from keras ---
        data = os.listdir(self.checkpoint_dir)
        if ".DS_Store" in data:  # Only necessary for MacOS
            os.remove(self.checkpoint_dir + "/" + ".DS_Store")
            time.sleep(1)
            data = os.listdir(self.checkpoint_dir)

        if len(data) > 1:
            data = sorted_nicely(data)
            for i in range(len(data)):
                print("[{}]: '{}'".format(i + 1, data[i]))
            inp = input("There are multiple files in this directory. (Choose file with number between "
                        "'1' and '{}'; default = '{}'): ".format(len(data), len(data)))
            if inp == "":
                file = data[len(data) - 1]
            else:
                file = data[int(inp) - 1]
            model = tf.keras.models.load_model(self.checkpoint_dir + "/" + file)
        else:
            file = data[0]
            model = tf.keras.models.load_model(self.checkpoint_dir + "/" + file)

        config = model.get_config()
        shape = config["layers"][0]["config"]["batch_input_shape"]

        images, src_images, data = self.get_images(shape)
        images = self.preprocess_data(images, shape)
        predictions = model.predict(images)
        self.predict_data(src_images, predictions, data, file)
