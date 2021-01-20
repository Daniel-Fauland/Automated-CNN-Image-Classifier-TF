import os
import sys
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Predict():
    def __init__(self):
        self.path_data = "predict_data"
        params = "python/predict_params.csv"
        self.df = pd.read_csv(params)


    # ============================================================
    def get_images(self):
        if os.path.exists(self.path_data + "/insert your own images here that you want to predict.txt"):
            os.remove(self.path_data + "/insert your own images here that you want to predict.txt")

        dimx = self.df["dimx"][0]
        dimy = self.df["dimy"][0]
        images = []
        src_images = []
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
                      "This is probably due to an invalid character in the file name\nRename "
                      "the file and try again.".format(file))
                print("=" * 100)
                sys.exit(1)
        return images, src_images


    # ============================================================
    def preprocess_data(self, images):
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
        channels = self.df["channels"][0]

        for x in range(len(images)):
            images[x] = normalize(images[x], img_normalize, channels)

        images = np.array(images)
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], channels)
        return images


    # ============================================================
    def load_model(self, images):

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
        model = tf.keras.models.load_model("checkpoints/model.h5")
        prediction = model.predict(images)
        return prediction


    # ============================================================
    def predict_data(self, src_images, predictions):
        labels_file_name = self.df["csv_name"][0]
        df_labels = pd.read_csv("labels/" + labels_file_name)
        label_names = df_labels[self.df["csv_column"][0]].tolist()
        label_names_str = []
        for label in label_names:
            label_names_str.append(str(label))

        for i in range(len(src_images)):
            image = src_images[i]
            plt.imshow(image)
            plt.title("Prediction: " + label_names_str[np.argmax(predictions[i])])
            plt.show()


    # ============================================================
    def initialize(self):
        images, src_images = self.get_images()
        images = self.preprocess_data(images)
        predictions = self.load_model(images)
        self.predict_data(src_images, predictions)

