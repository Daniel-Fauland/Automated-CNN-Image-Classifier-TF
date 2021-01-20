from python.user_inputs import User_inputs
from python.preprocess import Preprocess

user_inputs = User_inputs()
settings = user_inputs.initialize()
preprocess = Preprocess()
preprocess.initialize(settings)
