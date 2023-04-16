import gym
import numpy as np
import os
from stable_baselines3 import PPO
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
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

def make_env(env_id, rank, logFile = None,seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        # env.seed(seed + rank)
        # return Monitor(env,logFile)
        return env
        # set_random_seed(seed)
        env.seed(seed + rank)
        return env
    return _init

if __name__ == "__main__":
    env_id = 'IL/GridWorld-v0'
    num_cpu = 8  # Number of processes to use
    log_dir = "dqn_log"
    os.makedirs(log_dir, exist_ok=True)
    env = SubprocVecEnv([make_env(env_id, i,log_dir) for i in range(num_cpu)])
    # env = Monitor(gym.make(env_id),log_dir)
    # env = gym.make(env_id)
    model =DQN("CnnPolicy", env,gamma=0.999)
    model.policy =PPO("CnnPolicy", env).policy.load('bc_policy')
    # model.policy = model.policy.load('bc_policy')
    # callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
    for i in range(1000):
      # model.learn(total_timesteps=2048,callback = callback,log_interval=1)
      model.learn(total_timesteps=8192)
      testEnv = gym.make(env_id)
      rewardList = []
      print(f"Test begins after {i} times train")
      for _ in range(5):
        observation = testEnv.reset()
        action,state = model.predict(observation)
        Done = False
        rewardSum = 0
        while not Done:
          observation,reward,Done,_ = testEnv.step(int(action))
          action,state = model.predict(observation,state)
          rewardSum += reward
        rewardList.append(rewardSum)
      print(f"rewardList is {rewardList} after train {i} times")
      model.save('dqn_policy')
