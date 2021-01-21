import os
import sys
import cv2
import re
import tensorflow as tf
import numpy as np


class Augmentation():
    def __init__(self):
        self.path_data = "training_data"


    # ============================================================
    def initialize(self, augmentation_inp):
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
        count_src = 0
        count_img = 0
        for folder in data:
            f = os.listdir(self.path_data + "/" + folder)
            count_src += len(f)
            for file in f:
                image = cv2.imread(self.path_data + "/" + folder + "/" + file)
                if "2" in augmentation_inp:
                    aug_img_rot1 = tf.image.rot90(image, k=1)
                    cv2.imwrite(self.path_data + "/" + folder + "/" + "augmented_rotation1_" + file, np.float32(aug_img_rot1))
                    count_img += 1
                if "3" in augmentation_inp:
                    aug_img_rot2 = tf.image.rot90(image, k=2)
                    cv2.imwrite(self.path_data + "/" + folder + "/" + "augmented_rotation1_" + file, np.float32(aug_img_rot2))
                    count_img += 1
                if "4" in augmentation_inp:
                    aug_img_rot3 = tf.image.rot90(image, k=3)
                    cv2.imwrite(self.path_data + "/" + folder + "/" + "augmented_rotation1_" + file, np.float32(aug_img_rot3))
                    count_img += 1
                if "5" in augmentation_inp:
                    aug_img_right_left = tf.image.random_flip_left_right(image)
                    cv2.imwrite(self.path_data + "/" + folder + "/" + "augmented_flip_left_right_" + file, np.float32(aug_img_right_left))
                    count_img += 1
                if "6" in augmentation_inp:
                    aug_img_up_down = tf.image.random_flip_up_down(image)
                    cv2.imwrite(self.path_data + "/" + folder + "/" + "augmented_flip_up_down_" + file, np.float32(aug_img_up_down))
                    count_img += 1
                if "7" in augmentation_inp:
                    aug_img_r_hue = tf.image.random_hue(image, 0.08)
                    cv2.imwrite(self.path_data + "/" + folder + "/" + "augmented_random_hue_" + file, np.float32(aug_img_r_hue))
                    count_img += 1
                if "8" in augmentation_inp:
                    aug_img_r_saturation = tf.image.random_saturation(image, 0.6, 1.6)
                    cv2.imwrite(self.path_data + "/" + folder + "/" + "augmented_random_saturation_" + file, np.float32(aug_img_r_saturation))
                    count_img += 1
                if "9" in augmentation_inp:
                    aug_img_random_brightness = tf.image.random_brightness(image, 0.05)
                    cv2.imwrite(self.path_data + "/" + folder + "/" + "augmented_random_brightness_" + file, np.float32(aug_img_random_brightness))
                    count_img += 1
                if "10" in augmentation_inp:
                    aug_img_random_contrast = tf.image.random_contrast(image, 0.7, 1.3)
                    cv2.imwrite(self.path_data + "/" + folder + "/" + "augmented_random_contrast_" + file, np.float32(aug_img_random_contrast))
                    count_img += 1
            count += 1
            sys.stdout.write('\r' + "Augmented folder {}/{}".format(count, len(data)))
        sys.stdout.write('\r' + "Successfully created {} augmented images.".format(count_img))
        print("\nIncreased training data from {} images to {} images.\n".format(count_src, count_src + count_img))
        return


    def delete_augmentations(self):
        def sorted_nicely(data):
            """ Sort the given iterable in the way that humans expect."""
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(data, key=alphanum_key)

        if os.path.exists(self.path_data + "/Insert your training data in this directory.txt"):
            os.remove(self.path_data + "/Insert your training data in this directory.txt")

        print()
        data = os.listdir(self.path_data)
        data = sorted_nicely(data)
        count = 0
        count_src = 0
        count_img = 0
        for folder in data:
            f = os.listdir(self.path_data + "/" + folder)
            count_src += len(f)
            for file in f:
                image = self.path_data + "/" + folder + "/" + file
                if "augmented" in image:
                    os.remove(image)
                    count_img += 1
            count += 1
            sys.stdout.write('\r' + "Deleted augmented images in folder {}/{}.".format(count, len(data)))
        sys.stdout.write('\r' + "Successfully deleted all {} augmented images.".format(count_img))
        print("\nDecreased training data from {} images to its original size of {} images.".format(count_src, count_src - count_img))

        print("\n[1]: Continue training without augmented images\n[2]: Create new augmentations and continue training\n[3]: Exit")
        inp = input("Type either '1', '2' or '3' (default = '3'): ")
        if inp == "1":
            augmentation_inp = ["1"]
            return augmentation_inp
        if inp == "2":
            print("\n[1]: Don't augment any images\n[2]: Rotate by 90 degrees\n[3]: Rotate by 180 degrees\n[4]: Rotate by 270 degrees\n[5]: Randomly "
                  "flip left or right\n[6]: Randomly flip up or down\n[7]: Randomly change hue\n[8]: Randomly change saturation "
                  "\n[9]: Randomly change brightness\n[10]: Randomly change contrast")
            augmentation_inp = input("Augment your training data. Type e.g. '2 4 10' for option [2], [4] and [10] (default = '1'): ").split(' ')
            if "1" not in augmentation_inp and augmentation_inp != [""]:
                self.initialize(augmentation_inp)
        else:
            sys.exit(1)


