import os

import matplotlib.pyplot as plt
import numpy as np
from dopamine.colab import utils as colab_utils


raw_data = colab_utils.load_statistics('/tmp/endorphin/ppo/CartPole/logs')
summarized_data = colab_utils.summarize_data(raw_data[0], ['eval_episode_returns'])

plt.plot(summarized_data['eval_episode_returns'], label='episode returns')
plt.plot()
plt.title("ppo training - CartPole")
plt.xlabel('Iteration')
plt.ylabel('Return')
plt.legend()
plt.show()