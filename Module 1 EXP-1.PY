# Experiment 1
# Aim: Installation and Verification of Reinforcement Learning Development Environment

import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import tensorflow as tf
import keras

print("="*60)
print("REINFORCEMENT LEARNING ENVIRONMENT VERIFICATION")
print("="*60)

print("\n1. Python Version")
print(sys.version)

print("\n2. NumPy Version")
print(np.__version__)

print("\n3. Matplotlib Version")
print(plt.matplotlib.__version__)

print("\n4. Gymnasium Version")
print(gym.__version__)

print("\n5. TensorFlow Version")
print(tf.__version__)

print("\n6. Keras Version")
print(keras.__version__)

print("\nCreating CartPole Environment...")

env = gym.make("CartPole-v1")

state, info = env.reset()

print("\nInitial State:")
print(state)

print("\nAction Space:", env.action_space)
print("Observation Space:", env.observation_space)

action = env.action_space.sample()

print("\nRandom Action Selected:", action)

next_state, reward, terminated, truncated, info = env.step(action)

print("\nNext State:")
print(next_state)

print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)

env.close()

print("\nTesting Matplotlib...")

x = [1,2,3,4,5]
y = [5,8,6,10,9]

plt.plot(x, y, marker='o')
plt.title("Environment Verification")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

print("\nEnvironment setup completed successfully.")