from python.labels import Labels
from python.preprocess import Preprocess
import time

settings = {}
labels = Labels()
settings["csv_name"], settings["csv_column"] = labels.initialize()

# print(settings["csv_name"])
# print(settings["csv_column"])

print("\n" + "=" * 70)
settings["validation"] = input("Choose validation size in % (default = '20'): ")
print("=" * 70 + "\n")

print("=" * 70)
settings["dim"] = input("Resize all images to a specific size. Type e.g. '64 32' for 64px width and 32px height"
                        " (NOTE: If left blank all images will be resized to the shape of the first image in "
                        "the training folder): ")
print("=" * 70 + "\n")


print("=" * 70 + "\n[1]: Normalize the pixel values between 0 and 1 (recommended. Can drastically increase training accuracy)\n"
                        "[2]: Don't normalize. Pixel values will be between 0 and 255\n" + "=" * 70)
settings["normalize"] = input("Type either '1' or '2' (default = '1'): ")
# print("=" * 70 + "\n")


print("\n" + "=" * 70)
settings["epochs"] = input("Choose number of Epochs (default = '10'): ")
settings["batch_size"] = input("Choose batch size (default = '64'; higher batch size can improve model quality but requires more ram): ")
print("=" * 70 + "\n")


print("=" * 70 + "\n[1]: Automatic (Use GPU or CPU depending on your installation. If you have a NVIDIA GPU and you get"
                 " a TF error of any kind use one of the other options)\n"
                 "[2]: Use GPU for training and CPU for predicting with memory growth enabled for the GPU (recommended if you have a NVIDIA GPU and CUDA "
                 "installed. This feature prevents TF from allocating more VRAM than the GPU actually has\n"
                 "[3]: Use GPU for training and predicting (Note: Predicting with GPU is slower than CPU in most cases)\n"
                 "[4]: Force CPU for training and predicting (CPU will be used for training the model even if you have a GPU available)\n" + "=" * 70)
settings["mode"] = input("Choose execution mode. Type either '1', '2', '3', or '4' (default = '1'): ")
# print("=" * 70 + "\n")

settings["s_time"] = time.time()
preprocess = Preprocess()
preprocess.initialize(settings)
