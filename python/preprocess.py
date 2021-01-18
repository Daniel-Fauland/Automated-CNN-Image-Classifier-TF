import os
import sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from python.model import Model


class Preprocess():
    def __init__(self):
        self.path_data = "./training_data"


    # ============================================================
    def load_data(self, validation):
        count = 0
        images, categories = [], []
        data = os.listdir(self.path_data)

        for x in range(len(data)):
            folder = os.listdir(self.path_data + "/" + str(count))
            for file in folder:
                image = cv2.imread(self.path_data + "/" + str(count) + "/" + file)
                images.append(image)
                categories.append(count)
            count += 1
            # print("Loaded folder {}/{}".format(count, len(data)), end='\r')
            sys.stdout.write('\r' + "Loaded folder {}/{}".format(count, len(data)))

        # --- split trainingData into train and validation ---
        x_train, x_val, y_train, y_val = train_test_split(images, categories, test_size=validation)
        print("")
        return x_train, x_val, y_train, y_val


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
        print("Preprocessing training data done.")
        return x_train, x_val, y_train, y_val



    # ============================================================
    def initialize(self, settings):
        if settings["validation"] == "":
            validation = 0.2
        else:
            validation = int(settings["validation"]) / 100
        img_normalize = settings["normalize"]
        x_train, x_val, y_train, y_val = self.load_data(validation)
        x_train, x_val, y_train, y_val = self.preprocess_data(x_train, x_val, y_train, y_val, img_normalize)

        mode = settings["mode"]
        model = Model(mode)
        model.train_model(x_train, x_val, y_train, y_val, settings)
