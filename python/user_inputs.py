from python.labels import Labels
import sys
import time


class User_inputs():
    def __init__(self):
        pass

    def initialize(self):
        settings = {}
        labels = Labels()
        option = " label options "
        print("*" * 30 + option + "*" * 30)
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
        option = " model and training options "
        print("\n" + "*" * 30 + option + "*" * 30)
        settings["validation"] = input("Choose validation size in % (default = '20'): ")
        settings["epochs"] = input("Choose number of Epochs (default = '10'): ")
        settings["batch_size"] = input("Choose batch size (default = '64'; higher batch size can improve model quality but requires more ram): ")

        settings["count_layers"] = 1
        c_pool = 1
        settings["pooling_layers"] = ["y"]
        settings["num_neurons_1"] = input("Choose number of neurons for the first 'Conv2D' layer (default = '32'): ")
        settings["strides_neurons_1"] = input("Choose strides for the first 'Conv2D' layer (default = '3 3'): ")
        print("\n[1]: Relu\n[2]: Sigmoid\n[3]: Softmax\n[4]: Softplus\n[5]: Softsign\n[6]: Tanh\n[7]: Selu\n"
              "[8]: Elu\n[9]: Exponential")
        settings["activation_type_1"] = input("Choose activation function for the first 'Conv2D' layer (default = '1'; "
                                              "Warning: Not all activations may be valid for this layer): ")
        settings["max_pool_"+str(c_pool)] = input("Choose pooling size for the first 'MaxPooling2D' layer (default = '2 2'): ")

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


        print("\n[1]: Automatic (Use GPU or CPU depending on your installation. If you have a NVIDIA GPU and you get a TF "
              "error of any kind use one of the other options)\n"
              "[2]: Use GPU for training and CPU for predicting with memory growth enabled for the GPU (recommended if you have"
              " a NVIDIA GPU and CUDA installed. This feature prevents TF from allocating more VRAM than the GPU actually has\n"
              "[3]: Use GPU for training and predicting (Note: Predicting with GPU is slower than CPU in most cases "
              "because of initializing time of cuda)\n"
              "[4]: Force CPU for training and predicting (CPU will be used for training the model even if you have a GPU available)")
        settings["mode"] = input("Choose execution mode. Type either '1', '2', '3', or '4' (default = '1'): ")

        print(settings)
        print("\n[1]: Start training\n[2]: Exit program")
        inp = input("Type either '1' or '2' (default = '1'): ")
        if inp == "2":
            sys.exit(1)
        settings["s_time"] = time.time()
        print()
        return settings
