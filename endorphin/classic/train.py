
from absl import app, flags

from endorphin.agents.classic import a2c_agent, ppo_agent
from endorphin.classic import run_experiment

flags.DEFINE_string('agent_name', 'a2c', 'Name of the agent')
flags.DEFINE_string('base_dir', 'tmp/endorphin/a2c/CartPole', 'Base directory to host all required sub-directories')
flags.DEFINE_string('game_name', 'CartPole-v0', 'Name of the game')
# flags.DEFINE_string(
#     'schedule', 'continuous_train',
#     'The schedule with which to run the experiment and choose an appropriate '
#     'Runner. Supported choices are '
#     '{continuous_train, continuous_train_and_eval}.')

FLAGS = flags.FLAGS


def create_agent(environment, n_env):
    """Select an agent"""
    if FLAGS.agent_name == 'a2c':
        return a2c_agent.A2CAgent(environment.action_space.n, n_env)
    if FLAGS.agent_name == 'ppo':
        return ppo_agent.PPOAgent(environment.action_space.n, n_env)

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
    print(f'agent: {FLAGS.agent_name}')
    print(f'game: {FLAGS.game_name}\n')
    launch_experiment(create_runner, create_agent)

if __name__ == '__main__':
    app.run(main)
