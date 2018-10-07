from dopamine.atari.preprocessing import AtariPreprocessing, FrameStackPreprocessing
import gym
import matplotlib.pyplot as plt

env = gym.make('PongNoFrameskip-v0')
env = env.env

env = AtariPreprocessing(env)
env = FrameStackPreprocessing(env)

s = env.reset()
print(s)
print(s.shape)
s2, reward, terminal, info = env.step(0)
print(s2)
# plt.imshow(s.reshape(84, 84))
# plt.show()
print('hello')
# plt.imshow(s)
# plt.show()
