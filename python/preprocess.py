import os
import sys
import cv2
import re
import time
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from python.model import Model


class Preprocess():
    def __init__(self):
        self.path_data = "training_data"
        self.checkpoint_dir = "./checkpoints"


    # ============================================================
    def load_data(self, validation, dimx, dimy, os_mode):
        def sorted_nicely(data):
            """ Sort the given iterable in the way that humans expect."""
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(data, key=alphanum_key)

        if os.path.exists(self.path_data + "/Insert your training data in this directory.txt"):
            os.remove(self.path_data + "/Insert your training data in this directory.txt")

        count = 0
        images, categories = [], []
        error = []
        data = os.listdir(self.path_data)
        if os_mode == "y":  # If OS is MacOS
            if ".DS_Store" in data:  # Only necessary for MacOS
                os.remove(self.path_data + "/" + ".DS_Store")
                time.sleep(1)
                data = os.listdir(self.path_data)
                data = sorted_nicely(data)
            data = sorted_nicely(data)
            for folder in data:  # Iterate over each folder in dir 'training_data'
                f = os.listdir(self.path_data + "/" + folder)
                if ".DS_Store" in f:  # Only necessary for MacOS
                    os.remove(self.path_data + "/" + folder + "/" + ".DS_Store")
                    time.sleep(1)
                    f = os.listdir(self.path_data + "/" + folder)
                for file in f:  # Iterate over each file in each folder
                    try:
                        image = cv2.imread(self.path_data + "/" + folder + "/" + file)  # read image with open-cv
                        image = cv2.resize(image, (dimx, dimy))  # resize image
                        images.append(image)  # append image to array
                        categories.append(count)  # append category/label to array
                    except:
                        error.append(folder + "/" + file)  # Adds file to error list if open-cv could not open it for some reason

                count += 1
                sys.stdout.write('\r' + "Loaded folder {}/{}".format(count, len(data)))

        else:
            data = sorted_nicely(data)
            for folder in data:  # Iterate over each folder in dir 'training_data'
                f = os.listdir(self.path_data + "/" + folder)
                for file in f:  # Iterate over each file in each folder
                    try:
                        image = cv2.imread(self.path_data + "/" + folder + "/" + file)  # read image with open-cv
                        image = cv2.resize(image, (dimx, dimy))  # resize image
                        images.append(image)  # append image to array
                        categories.append(count)  # append category/label to array
                    except:
                        error.append(folder + "/" + file)  # Adds file to error list if open-cv could not open it for some reason

                count += 1
                sys.stdout.write('\r' + "Loaded folder {}/{}.".format(count, len(data)))

        if len(error) > 0:
            print()
            print("\n" + "=" * 100)
            print("WARNING: {} file(s) could not be read for some reason. They were skipped instead.".format(len(error)))
            print("These files are:")
            for i in error:
                print("'" + i + "'")
            print("=" * 100)
        # --- split trainingData into train and validation ---
        x_train, x_val, y_train, y_val = train_test_split(images, categories, test_size=validation)  # Split into train and validation
        print()
        return x_train, x_val, y_train, y_val, len(data)


    # ============================================================
    def preprocess_data(self, x_train, x_val, y_train, y_val, img_normalize, channels):
        def normalize(img, img_normalize, channels):
            if channels == 3:
                pass
            else:
                img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)  # Grayscale image

            if img_normalize == "2":
                pass
            else:
                img = img / 255.0  # Normalize px values between 0 and 1
            return img


        for x in range(len(x_train)):
            x_train[x] = normalize(x_train[x], img_normalize, channels)  # preprocess all training images

        for x in range(len(x_val)):
            x_val[x] = normalize(x_val[x], img_normalize, channels)  # preprocess all validation images

        # --- transform the data to be accepted by the model ---
        y_train = np.array(y_train)  # TF only accepts numpy arrays
        y_val = np.array(y_val)  # TF only accepts numpy arrays
        x_train = np.array(x_train)  # TF only accepts numpy arrays
        x_val = np.array(x_val)  # TF only accepts numpy arrays
        # Reshape images back to its original form of width * height. (Open-cv flips width and height when resizing an image)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1], channels)
        x_val = x_val.reshape(x_val.shape[0], x_val.shape[2], x_val.shape[1], channels)
        print("Preprocessing training data complete.")
        return x_train, x_val, y_train, y_val



    # ============================================================
    def initialize(self, settings):
        def sorted_nicely(l):
            """ Sort the given iterable in the way that humans expect."""
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(l, key=alphanum_key)

        if os.path.exists(self.checkpoint_dir + "/your model will be saved in this directory.txt"):
            os.remove(self.checkpoint_dir + "/your model will be saved in this directory.txt")

        if settings["model_save"] == "2" or settings["model_save"] == "3":
            data = os.listdir(self.checkpoint_dir)
            data = sorted_nicely(data)
            if settings["model_save"] == "2":
                if len(data) > 0:
                    shutil.rmtree(self.checkpoint_dir + "/" + data[0])
                    time.sleep(1)
            if settings["model_save"] == "3":
                if len(data) > 0:
                    for i in data:
                        shutil.rmtree(self.checkpoint_dir + "/" + i)
                    time.sleep(1)

        if settings["validation"] == "":
            validation = 0.2
        else:
            validation = int(settings["validation"]) / 100

        if settings["dim"] == "":
            data = os.listdir(self.path_data)
            for folder in data:
                f = os.listdir(self.path_data + "/" + folder)
                for file in f:
                    if file.endswith(".txt") or file.endswith(".DS_Store"):
                        continue
                    image = cv2.imread(self.path_data + "/" + folder + "/" + file)
                    dimx = image.shape[0]  # width of image
                    dimy = image.shape[1]  # height of image
                    print("Automatically detected shape of {}x{} pixel for training images.".format(dimx, dimy))
                    break
                break
        else:
            dimx = int(settings["dim"].split(' ')[0])  # width
            dimy = int(settings["dim"].split(' ')[1])  # height
            print("Resizing all training images to {}x{} pixel.".format(dimx, dimy))

        if settings["channels"] == "2":
            channels = 3  # 3 --> RGB image
        else:
            channels = 1  # 1 --> Grayscale image
        img_normalize = settings["normalize"]

        x_train, x_val, y_train, y_val, dim_out = self.load_data(validation, dimx, dimy, settings["os"])
        x_train, x_val, y_train, y_val = self.preprocess_data(x_train, x_val, y_train, y_val, img_normalize, channels)

        df = {"csv_name": [settings["csv_name"]], "csv_column": [settings["csv_column"]],
              "img_normalize": [img_normalize], "mode": [settings["mode"]]}
        df = pd.DataFrame(df)  # create pandas dataframe for 'predict_data' file
        df.to_csv("python/predict_params.csv")  # save df in folder 'python'
        mode = settings["mode"]  # execution mode
        model = Model(mode)
        model.train_model(x_train, x_val, y_train, y_val, dimx, dimy, dim_out, settings)
