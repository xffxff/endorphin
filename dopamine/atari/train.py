
from absl import app
from absl import flags

from dopamine.atari import run_experiment
from dopamine.agents.atari.a2c import a2c_agent

flags.DEFINE_string('agent_name', 'a2c', 'Name of the agent')
flags.DEFINE_string('base_dir', '/tmp/dopamine/a2c/Pong', 'Base directory to host all required sub-directories')
flags.DEFINE_string('game_name', 'Breakout', 'Name of the game')
# flags.DEFINE_string(
#     'schedule', 'continuous_train_and_eval',
#     'The schedule with which to run the experiment and choose an appropriate '
#     'Runner. Supported choices are '
#     '{continuous_train, continuous_train_and_eval}.')

FLAGS = flags.FLAGS


def create_agent(environment, n_env):
    if FLAGS.agent_name == 'a2c':
        return a2c_agent.A2CAgent(environment.action_space.n, n_env=n_env)

def create_runner(create_agent_fn):
    return run_experiment.Runner(create_agent_fn, FLAGS.base_dir, game_name=FLAGS.game_name)

def launch_experiment(create_runner_fn, create_agent_fn):
    runner = create_runner_fn(create_agent_fn)
    runner.run_experiment()

def main(unused_argv):
    launch_experiment(create_runner, create_agent)

if __name__ == '__main__':
    app.run(main)