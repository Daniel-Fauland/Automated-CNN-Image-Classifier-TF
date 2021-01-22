from python.labels import Labels
from python.augmentation import Augmentation
from python.predefined_model import Predefined_model
import sys
import time


class User_inputs():
    def __init__(self):
        self.checkpoint_dir = "./checkpoints"

    def initialize(self):
        print("="*39 + " Automated-CNN-Image-Classifier " + "="*39)
        print("The default values can also be chosen by hitting enter")
        print("Read the GitHub (https://github.com/Daniel-Fauland/Automated-CNN-Image-Classifier-TF) for further instructions")
        print("="*110 + "\n")

        settings = {}
        # ==========================================================
        option = " augmentation options "
        print("*" * 30 + option + "*" * 30)
        print("[1]: Don't augment any images\n[2]: Rotate by 90 degrees\n[3]: Rotate by 180 degrees\n[4]: Rotate by 270 degrees\n[5]: Randomly "
              "flip left or right\n[6]: Randomly flip up or down\n[7]: Randomly change hue\n[8]: Randomly change saturation "
              "\n[9]: Randomly change brightness\n[10]: Randomly change contrast\n[11]: Delete all current augmentations")
        augmentation_inp = input("Augment your training data. Type e.g. '2 4 10' for option [2], [4] and [10] (default = '1'; type '11' to delete "
                    "all current augmented images): ").split(' ')
        if "11" in augmentation_inp:
            augmentation = Augmentation()
            augmentation_inp = augmentation.delete_augmentations()

        labels = Labels()
        settings["csv_name"], settings["csv_column"] = labels.initialize()

        # ==========================================================
        option = " preprocess options "
        print("\n" + "*" * 30 + option + "*" * 30)
        settings["dim"] = input("Resize all images to a specific size. Type e.g. '64 32' for 64px width and 32px height"
                                " (default = resize images to shape of the first image found): ")
        print("\n[1]: Grayscale images (makes all images black and white)\n"
              "[2]: Don't grayscale (color of pictures will be unchanged)")
        settings["channels"] = input("Type either '1' or '2' (default = '1'): ")
        print("\n[1]: Normalize the pixel values between 0 and 1 (recommended option. Can drastically increase model accuracy)\n"
                                "[2]: Don't normalize. Pixel values will be between 0 and 255")
        settings["normalize"] = input("Type either '1' or '2' (default = '1'): ")

        # ==========================================================
        option = " training options "
        print("\n" + "*" * 30 + option + "*" * 30)
        settings["validation"] = input("Choose validation size in % (default = '20'): ")
        settings["epochs"] = input("Choose number of Epochs (default = '10'): ")
        settings["batch_size"] = input("Choose batch size (default = '64'; higher batch size can improve model quality but requires more ram): ")

        option = " model options "
        print("\n" + "*" * 30 + option + "*" * 30)
        print("[1]: Don't delete any previous model files\n[2]: Delete oldest existing model file\n[3]: Delete all existing model files")
        settings["model_save"] = input("Type either '1', '2' or '3' (default = '1'): ")

        print("\n[1]: Customize model structure yourself\n[2]: Use predefined model structure (can be viewed in 'python/predefined_model_summary.txt')")
        inp = input("Type either '1' or '2' (default = '1'): ")
        settings["predefined_model"] = "n"
        if inp != '2':
            # --- These are the options to customize the model ---
            settings["count_layers"] = 1
            c_pool = 0
            settings["pooling_layers"] = []
            settings["num_neurons_1"] = input("Choose number of neurons for the first 'Conv2D' layer (default = '32'): ")
            settings["strides_neurons_1"] = input("Choose strides for the first 'Conv2D' layer (default = '3 3'): ")
            print("\n[1]: Relu\n[2]: Sigmoid\n[3]: Softmax\n[4]: Softplus\n[5]: Softsign\n[6]: Tanh\n[7]: Selu\n"
                  "[8]: Elu\n[9]: Exponential")
            settings["activation_type_1"] = input("Choose activation function for the first 'Conv2D' layer (default = '1'; "
                                                  "Warning: Not all activations may be valid for this layer): ")
            print("\n[1]: Add a 'MaxPooling2D' layer\n[2]: Don't add a 'MaxPooling2D' layer")
            inp2 = input("Type either '1' or '2' (default = '1'): ")
            if inp2 != "2":
                c_pool += 1
                settings["pooling_layers"].append("y")
                settings["max_pool_"+str(c_pool)] = input("Choose pooling size for the first 'MaxPooling2D' layer (default = '2 2'): ")
            else:
                settings["pooling_layers"].append("n")

            print("\n[1]: Add a second 'Conv2D' layer\n[2]: Don't add any more layers")
            inp = input("Type either '1' or '2' (default = '1'): ")
            if inp != "2":
                settings["count_layers"] += 1
                settings["num_neurons_2"] = input("Choose number of neurons for the second 'Conv2D' layer (default = '64'): ")
                settings["strides_neurons_2"] = input("Choose strides for the second 'Conv2D' layer (default = '3 3'): ")
                print("\n[1]: Relu\n[2]: Sigmoid\n[3]: Softmax\n[4]: Softplus\n[5]: Softsign\n[6]: Tanh\n[7]: Selu\n"
                      "[8]: Elu\n[9]: Exponential")
                settings["activation_type_2"] = input("Choose activation function for the second 'Conv2D' layer "
                                                      "(default = '1'; Warning: Not all activations may be valid for this layer): ")
                print("\n[1]: Add a second 'MaxPooling2D' layer\n[2]: Don't add a second 'MaxPooling2D' layer")
                inp2 = input("Type either '1' or '2' (default = '1'): ")
                if inp2 != "2":
                    c_pool += 1
                    settings["pooling_layers"].append("y")
                    settings["max_pool_"+str(c_pool)] = input("Choose pooling size for the second 'MaxPooling2D' layer (default = '2 2'): ")
                else:
                    settings["pooling_layers"].append("n")

                print("\n[1]: Add a third 'Conv2D' layer\n[2]: Don't add any more layers")
                inp = input("Type either '1' or '2' (default = '2'): ")
                if inp == "1":
                    settings["count_layers"] += 1
                    settings["num_neurons_3"] = input("Choose number of neurons for the third 'Conv2D' layer (default = '64'): ")
                    settings["strides_neurons_3"] = input("Choose strides for the third 'Conv2D' layer (default = '3 3'): ")
                    print("\n[1]: Relu\n[2]: Sigmoid\n[3]: Softmax\n[4]: Softplus\n[5]: Softsign\n[6]: Tanh\n[7]: Selu\n"
                          "[8]: Elu\n[9]: Exponential")
                    settings["activation_type_3"] = input("Choose activation function for the third 'Conv2D' layer "
                                                          "(default = '1'; Warning: Not all activations may be valid for this layer): ")
                    print("\n[1]: Add a third 'MaxPooling2D' layer\n[2]: Don't add a third 'MaxPooling2D' layer")
                    inp2 = input("Type either '1' or '2' (default = '2'): ")
                    if inp2 == "1":
                        c_pool += 1
                        settings["pooling_layers"].append("y")
                        settings["max_pool_" + str(c_pool)] = input("Choose pooling size for the third 'MaxPooling2D' layer (default = '2 2'): ")
                    else:
                        settings["pooling_layers"].append("n")

                    print("\n[1]: Add a fourth 'Conv2D' layer\n[2]: Don't add any more layers")
                    inp = input("Type either '1' or '2' (default = '2'): ")
                    if inp == "1":
                        settings["count_layers"] += 1
                        settings["num_neurons_4"] = input(
                            "Choose number of neurons for the fourth 'Conv2D' layer (default = '64'): ")
                        settings["strides_neurons_4"] = input(
                            "Choose strides for the fourth 'Conv2D' layer (default = '3 3'): ")
                        print(
                            "\n[1]: Relu\n[2]: Sigmoid\n[3]: Softmax\n[4]: Softplus\n[5]: Softsign\n[6]: Tanh\n[7]: Selu\n"
                            "[8]: Elu\n[9]: Exponential")
                        settings["activation_type_4"] = input("Choose activation function for the fourth 'Conv2D' layer "
                                                              "(default = '1'; Warning: Not all activations may be valid for this layer): ")
                        print("\n[1]: Add a fourth 'MaxPooling2D' layer\n[2]: Don't add a fourth 'MaxPooling2D' layer")
                        inp2 = input("Type either '1' or '2' (default = '2'): ")
                        if inp2 == "1":
                            c_pool += 1
                            settings["pooling_layers"].append("y")
                            settings["max_pool_" + str(c_pool)] = input("Choose pooling size for the fourth 'MaxPooling2D' layer (default = '2 2'): ")
                        else:
                            settings["pooling_layers"].append("n")

            print("\n[1]: Add dropout layer\n[2]: Don't add dropout layer")
            inp = input("Type either '1' or '2' (default = '1'): ")
            if inp != "2":
                settings["dropout_1"] = input("Choose dropout ratio in % (default = '25'): ")
            settings["num_hidden_layers"] = input("Choose the amount of hidden layers (default = '1'): ")
            if settings["num_hidden_layers"] == "":
                settings["num_hidden_layers"] = 1
            for i in range(int(settings["num_hidden_layers"])):
                settings["hidden_layer_" + str(i+1)] = input("Choose number of neurons for hidden layer '{}' (default = '64'): ".format(i+1))
                print("\n[1]: Relu\n[2]: Sigmoid\n[3]: Softmax\n[4]: Softplus\n[5]: Softsign\n[6]: Tanh\n[7]: Selu\n"
                    "[8]: Elu\n[9]: Exponential")
                settings["hidden_layer_activation_" + str(i + 1)] = input("Choose activation function for hidden layer '{}' "
                                                                          "(default = '1'; Warning: Not all activations may "
                                                                          "be valid for this layer): ".format(i+1))

            print("\n[1]: Add second dropout layer before output layer\n[2]: Don't add another dropout layer")
            inp = input("Type either '1' or '2' (default = '1'): ")
            if inp != "2":
                settings["dropout_2"] = input("Choose dropout ratio in % (default = '25'): ")
        else:
            predefined_model = Predefined_model()
            settings = predefined_model.initialize(settings)
            settings["predefined_model"] = "y"

        # --- Choose execution mode ---
        option = " execution options "
        print("\n" + "*" * 30 + option + "*" * 30)
        print("[1]: Automatic (Use GPU or CPU depending on your installation. IMPORTANT: If you have a NVIDIA GPU and you get a TF "
              "error of any kind use one of the other options)\n"
              "[2]: Use GPU for training and CPU for predicting with memory growth enabled for the GPU (recommended if you have"
              " a NVIDIA GPU and CUDA installed. This feature prevents TF from allocating more VRAM than the GPU actually has)\n"
              "[3]: Use GPU for training and predicting (Note: Predicting with GPU is slower than CPU in most cases "
              "because of initializing duration for cuda)\n"
              "[4]: Force CPU for training and predicting (CPU will be used for training the model even if you have a GPU available)")
        settings["mode"] = input("Choose execution mode. Type either '1', '2', '3', or '4' (default = '1'): ")

        print("\n[1]: Start training\n[2]: Exit program")
        inp = input("Type either '1' or '2' (default = '1'): ")
        if inp == "2":
            sys.exit(1)
        settings["s_time"] = time.time()
        option = " START "
        print("\n" + "*" * 30 + option + "*" * 30)
        print()
        if "1" not in augmentation_inp and augmentation_inp != [""]:
            augmentation = Augmentation()
            augmentation.initialize(augmentation_inp)
        return settings


