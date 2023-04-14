import gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from wrpsolver.bc.gym_env import GridWorldEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.check_freq = check_freq
        # self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        pass
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.model.save('dqn_policy')

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    env_id = 'IL/GridWorld-v0'
    num_cpu = 8  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000,log_dir = 'dqn_log/')

    model =DQN("CnnPolicy", env,buffer_size=100000,batch_size=1024,gradient_steps=2,train_freq=(256,'step'))
    model.policy = A2C("CnnPolicy",env=env).policy.load('bc_policy_ppo1')
    model.learn(total_timesteps=32768*100,callback = callback,log_interval=1)
    print("finish")
    model.save('a2c_policy')
