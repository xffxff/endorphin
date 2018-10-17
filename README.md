# dopamine-X
[Dopamine](https://github.com/google/dopamine) is my favorite research framework of reinforcement learning, for
its very readable code and easy to try out new ideas. This 
project is mainly inspired by dopamine, and can also be said I am learning code from dopamine.  
  
My design princles are:  
- Flexible development: Make it easy to try out research ideas.  
- Concentration: Only focus on policy gradient or actor critic style algorithms.

Also inspired by:
- [openai/baselines](https://github.com/openai/baselines)

### Prerequisites
#### Ubuntu
`sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev`

### Requirements
- Python3
- [PyTorch](https://pytorch.org/)
- [Openai/Gym](https://gym.openai.com/)
- [Stable-Baselines](https://github.com/hill-a/stable-baselines)

### Training models
`python -m dopamine.atari.train --agent_name=<name of the agent> --base_dir=<name of the base directory> --game_name=<name of the game>`
#### Examples
To train a fully-connected network controlling Classic Control using ppo  

`python -m dopamine.classic.train --agent_name='ppo' --base_dir=/tmp/dopamine/ppo/CartPole --game_name='CartPole-v0' `

To train a conv network controlling Atari using a2c  

`python -m dopamine.atari.train --agent_name='a2c' --base_dir=/tmp/dopamine/a2c/Breakout --game_name='Breakout'`
