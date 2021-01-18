from python.preprocess import Preprocess

settings = {}
print("=" * 70)
settings["validation"] = input("Choose validation size in % (default = '20'): ")
print("=" * 70 + "\n")


print("=" * 70 + "\n[1]: Normalize the pixel values between 0 and 1 (can increase training accuracy)\n"
                        "[2]: Don't normalize. Pixel values will be between 0 and 255")
settings["normalize"] = input("Type either '1' or '2' (default = '1'): ")
print("=" * 70 + "\n")


print("=" * 70)
settings["epochs"] = input("Choose number of Epochs (default = '10'): ")
settings["batch_size"] = input("Choose batch size (default = '64'): ")
print("=" * 70 + "\n")


print("=" * 70 + "\n[1]: Automatic (Use GPU or CPU depending on your installation)\n"
                        "[2]: Use GPU with memory growth enabled (recommended if you have a NVIDIA GPU and CUDA "
                 "installed. This feature prevents TF from allocating more VRAM than the GPU actually has\n"
                        "[3]: Force CPU usage (CPU will be used for training the model)")
settings["mode"] = input("Type either '1' or '2' or '3' (default = '1'): ")
print("=" * 70 + "\n")

preprocess = Preprocess()
preprocess.initialize(settings)
