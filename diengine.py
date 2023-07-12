
import torch
from wrpsolver.bc.gym_env_ding import GridWorldEnv
from ditk import logging
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.envs import DingEnvWrapper, SubprocessEnvManagerV2
from ding.envs.env_wrappers import  ScaledFloatFrameWrapper,EvalEpisodeReturnEnv, TimeLimitWrapper,MaxAndSkipWrapper,FrameStackWrapper,WarpFrameWrapper
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    eps_greedy_handler, CkptSaver, nstep_reward_enhancer
from ding.utils import set_pkg_seed
from wrpsolver.bc.ding_config import main_config, create_config
def wrapped_grid_env():
    return DingEnvWrapper(
        GridWorldEnv(),
        cfg={
            'env_wrapper': [
                lambda env: MaxAndSkipWrapper(env, skip=4),
                lambda env: WarpFrameWrapper(env, size=200),
                lambda env: ScaledFloatFrameWrapper(env),
                lambda env: TimeLimitWrapper(env, max_limit=4000),
                lambda env: FrameStackWrapper(env, n_frames=4),
                lambda env: EvalEpisodeReturnEnv(env),
            ]
        }
    )
def main():
    filename = '{}/log.txt'.format(main_config.exp_name)
    logging.getLogger(with_files=[filename]).setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = SubprocessEnvManagerV2(
            env_fn=[wrapped_grid_env for _ in range(cfg.env.collector_env_num)],
            cfg=cfg.env.manager
        )
        evaluator_env = SubprocessEnvManagerV2(
            env_fn=[wrapped_grid_env for _ in range(cfg.env.evaluator_env_num)],
            cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = DQN(**cfg.policy.model)
        # state_dict = torch.load('/remote-home/ums_qipeng/WatchRouteProblem/GridWorld_dqn_seed0_230507_041521/ckpt/iteration_31000.pth.tar', map_location='cpu') # 从模型文件加载模型参数
        # model.load_state_dict(state_dict['model']) # 将模型参数载入模型
        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = DQNPolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(nstep_reward_enhancer(cfg))
        task.use(data_pusher(cfg, buffer_))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=1000))
        task.run()

if __name__ == "__main__":
    main()