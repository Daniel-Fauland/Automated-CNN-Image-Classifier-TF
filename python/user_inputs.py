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
        settings["num_neurons_1"] = input("Choose number of neurons for the first Conv2D layer (default = '32'): ")
        settings["strides_neurons_1"] = input("Choose strides for the first Conv2D layer (default = '3 3'): ")
        settings["max_pool_1"] = input("Choose pooling size for the first MaxPooling2D layer (default = '2 2'): ")
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
        inp = input("Type '1' or click enter to start the training or type '2' to exit (default = '1'): ")
        if inp == "2":
            sys.exit(1)
        settings["s_time"] = time.time()
        print()
        return settings
