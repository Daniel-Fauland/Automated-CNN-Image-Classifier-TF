class Predefined_model():
    def initialize(self, settings):
        # --- These are the settings for the predefined model ---
        settings["count_layers"] = 2  # Choose number of convolutional layers (min number: 1)
        settings["num_neurons_1"] = "64"  # Choose number of neurons for first convolutional layer
        settings["num_neurons_2"] = "64"  # Choose number of neurons for second convolutional layer
        settings["activation_type_1"] = "1"  # Choose activation function for first convolutional layer
        settings["activation_type_2"] = "1"  # Choose activation function for second convolutional layer
        settings["strides_neurons_1"] = "3 3"  # Choose strides for first convolutional layer
        settings["strides_neurons_2"] = "3 3"  # Choose strides for second convolutional layer

        # This array has to be the same length as number of conv. layer. Specify with 'y' or 'n' the amount and position of the MaxPooling layers
        settings["pooling_layers"] = ["y", "y"]
        settings["max_pool_1"] = "2 2"  # Choose pooling size for the first MaxPooling layer
        settings["max_pool_2"] = "2 2"  # Choose pooling size for the second MaxPooling layer
        settings["dropout_1"] = "20"  # Choose dropout ratio for first dropout layer. If you don't want a dropout layer set the value to "0"
        settings["num_hidden_layers"] = 1  # Choose number of hidden/dense layers (min number: 0)
        settings["hidden_layer_1"] = "64"  # Choose number of neurons for the first hidden layer
        settings["hidden_layer_activation_1"] = "1"  # Choose activation function for the first hidden layer
        settings["dropout_2"] = "20"  # Choose dropout ratio for second dropout layer. If you don't want a dropout layer set the value to "0"
        return settings
