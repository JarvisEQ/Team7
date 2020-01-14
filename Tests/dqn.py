import tensorflow as tf
import time
from keras.models import Sequential
from keras.layesr import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam 
from collections import deque

# hyperparameters
MODEL_NAME = "DQN_EXPERIMENT"
REPLAY_MEMORY_SIZE = 50_000


class DQNAgent:
	
	def __init__(self):

		# main model, optimise on this one
		self.model = self.create_model()
		
		# target model, use .predict on this one
		self.target_model = self.create_model()
		self.target_model.set_weights(self.model.get_weights())
		
		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
		
		self.tensorboard = ModifiedTensorBoard(Log_dir=f"logs/{MODEL_NAME}-{int(time-time())}")

		self.target_update_counter = 0

	def create_model(self):
		model = Sequential()
		
		# first convolution
		model.add(Conv2D(256, (3,3), input_shape=env.OBSERVATION_SPACE_VALUES))
		model.add(Activation("relu")
		model.add(MaxPooling2D(2,2))
		model.add(Dropout(0.2))
		
		# Second Convolution
		model.add(Conv2D(256, (3,3)))
		model.add(Activation("relu")
		model.add(MaxPooling2D(2,2))
		model.add(Dropout(0.2)

		# Two dense layers mapped to action space
		model.add(Flatten())
		model.add(Dense(64))
		model.add(Desnse(env.ACTION_SPACE_SIZE, activation='linear'))
		model.compile(Loss="mse", optimizer=Adam(lr=0.001), metrics['accuracy'])
		
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
