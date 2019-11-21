import gym
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda


class DQNNetwork(Model):
    """
    Class for DQN model architecture.
    """
    
    def __init__(self, num_actions: int, agent_history_length: int):
        super(DQNNetwork, self).__init__()
        self.normalize = Lambda(lambda x: x / 255.0)
        self.conv1 = Conv2D(filters=32, kernel_size=8, strides=4, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu", input_shape=(None, 84, 84, agent_history_length))
        self.conv2 = Conv2D(filters=64, kernel_size=4, strides=2, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.conv3 = Conv2D(filters=64, kernel_size=3, strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.flatten = Flatten()
        self.dense1 = Dense(512, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation='relu')
        self.dense2 = Dense(num_actions, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="linear")

    @tf.function
    def call(self, x):
        normalized = self.normalize(x)
        h1 = self.conv1(normalized)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.flatten(h3)
        h5 = self.dense1(h4)
        out = self.dense2(h5)
        return out
