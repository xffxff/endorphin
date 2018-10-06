
from absl import app
from absl import flags

from dopamine.atari import run_experiment
from dopamine.agents.a2c2 import a2c_agent

# flags.DEFINE_string('agent_name', None, 'Name of the agent')
# # flags.DEFINE_string('base_dir', None, 'Base directory to host all required sub-directories')
# flags.DEFINE_string(
#     'schedule', 'continuous_train',
#     'The schedule with which to run the experiment and choose an appropriate '
#     'Runner. Supported choices are '
#     '{continuous_train, continuous_train_and_eval}.')

# FLAGS = flags.FLAGS


def create_agent(environment):
    return a2c_agent.A2CAgent(environment.action_space.n)

def create_runner(create_agent_fn, base_dir):
    return run_experiment.Runner(create_agent_fn, base_dir)

def launch_experiment(create_runner_fn, create_agent_fn, base_dir):
    runner = create_runner_fn(create_agent_fn, base_dir)
    runner.run_experiment()

def main(unused_argv):
    base_dir = '/tmp/dopamine/a2c2'
    launch_experiment(create_runner, create_agent, base_dir)

if __name__ == '__main__':
    app.run(main)