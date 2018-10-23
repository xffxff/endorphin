
from absl import app
from absl import flags

from endorphin.atari import run_experiment
from endorphin.agents.atari import a2c_agent, ppo_agent

flags.DEFINE_string('agent_name', 'ppo', 'Name of the agent')
flags.DEFINE_string('base_dir', '/tmp/endorphin/a2c/Breakout', 'Base directory to host all required sub-directories')
flags.DEFINE_string('game_name', 'Breakout', 'Name of the game')
# flags.DEFINE_string(
#     'schedule', 'continuous_train_and_eval',
#     'The schedule with which to run the experiment and choose an appropriate '
#     'Runner. Supported choices are '
#     '{continuous_train, continuous_train_and_eval}.')

FLAGS = flags.FLAGS


def create_agent(environment, n_env):
    """Select an agent"""
    if FLAGS.agent_name == 'a2c':
        return a2c_agent.A2CAgent(environment.action_space.n, n_env=n_env)
    if FLAGS.agent_name == 'ppo':
        return ppo_agent.PPOAgent(environment.action_space.n, n_env=n_env)

def create_runner(create_agent_fn):
    """Create an experiment Runner"""
    return run_experiment.Runner(create_agent_fn, FLAGS.base_dir, game_name=FLAGS.game_name)

def launch_experiment(create_runner_fn, create_agent_fn):
    """Launches the experiment."""
    runner = create_runner_fn(create_agent_fn)
    runner.run_experiment()

def main(unused_argv):
    """Main method.

    Args:
        unused_argv: Arguments (unused).
    """
    launch_experiment(create_runner, create_agent)

if __name__ == '__main__':
    app.run(main)