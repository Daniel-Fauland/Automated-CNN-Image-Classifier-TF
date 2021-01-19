import os
import sys
import cv2
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from python.model import Model


class Preprocess():
    def __init__(self):
        self.path_data = "training_data"


    # ============================================================
    def load_data(self, validation, dimx, dimy):
        def sorted_nicely(data):
            """ Sort the given iterable in the way that humans expect."""
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(data, key=alphanum_key)

        if os.path.exists(self.path_data + "/Insert your training data in this directory.txt"):
            os.remove(self.path_data + "/Insert your training data in this directory.txt")

        data = os.listdir(self.path_data)
        data = sorted_nicely(data)
        count = 0
        images, categories = [], []
        for folder in data:
            f = os.listdir(self.path_data + "/" + folder)
            for file in f:
                image = cv2.imread(self.path_data + "/" + folder + "/" + file)
                image = cv2.resize(image, (dimx, dimy))
                images.append(image)
                categories.append(count)

            count += 1
            sys.stdout.write('\r' + "Loaded folder {}/{}".format(count, len(data)))


        # --- split trainingData into train and validation ---
        x_train, x_val, y_train, y_val = train_test_split(images, categories, test_size=validation)
        print("")
        return x_train, x_val, y_train, y_val, len(data)


    # ============================================================
    def preprocess_data(self, x_train, x_val, y_train, y_val, img_normalize):
        def normalize(img, img_normalize):
            img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)  # Grayscale image
            # img = cv2.equalizeHist(np.uint8(img))                  # Optimize Lightning
            if img_normalize == "2":
                pass
            else:
                img = img / 255.0  # Normalize px values between 0 and 1
            return img


        for x in range(len(x_train)):
            x_train[x] = normalize(x_train[x], img_normalize)

        for x in range(len(x_val)):
            x_val[x] = normalize(x_val[x], img_normalize)

        # --- transform the data to be accepted by the model ---
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        x_train = np.array(x_train)
        x_val = np.array(x_val)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
        print("Preprocessing training data complete.")
        return x_train, x_val, y_train, y_val



    # ============================================================
    def initialize(self, settings):
        if settings["validation"] == "":
            validation = 0.2
        else:
            validation = int(settings["validation"]) / 100
        img_normalize = settings["normalize"]

        if settings["dim"] == "":
            data = os.listdir(self.path_data)
            for folder in data:
                f = os.listdir(self.path_data + "/" + folder)
                for file in f:
                    if file.endswith(".txt"):
                        continue
                    image = cv2.imread(self.path_data + "/" + folder + "/" + file)
                    dimx = image.shape[0]
                    dimy = image.shape[1]
                    print("\nAutomatically detected shape of {}x{} pixel for training images.".format(dimx, dimy))
                    break
                break
        else:
            dimx = int(settings["dim"].split(' ')[0])
            dimy = int(settings["dim"].split(' ')[1])

        x_train, x_val, y_train, y_val, dim_out = self.load_data(validation, dimx, dimy)
        x_train, x_val, y_train, y_val = self.preprocess_data(x_train, x_val, y_train, y_val, img_normalize)

        df = {"dimx": [dimx], "dimy": [dimy], "csv_name": [settings["csv_name"]], "csv_column": [settings["csv_column"]],
              "img_normalize": [img_normalize], "mode": [settings["mode"]]}
        df = pd.DataFrame(df)
        df.to_csv("python/predict_params.csv")
        mode = settings["mode"]
        model = Model(mode)
        model.train_model(x_train, x_val, y_train, y_val, dimx, dimy, dim_out, settings)
