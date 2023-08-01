import torch # 载入 PyTorch 库，用于加载 Tensor 模型，定义计算网络
import cv2
import os
from tensorboardX import SummaryWriter
from wrpsolver.bc.gym_env_hwc_100_pp_ding import GridWorldEnv
from easydict import EasyDict # 载入 EasyDict，用于实例化配置文件
from ding.config import compile_config # 载入DI-engine config 中配置相关组件
from ding.envs import SyncSubprocessEnvManager,BaseEnvManager,DingEnvWrapper # 载入DI-engine env 中环境相关组件
from ding.policy import DQNPolicy, single_env_forward_wrapper # 载入DI-engine policy 中策略相关组件
from ding.model import DQN # 载入DI-engine model 中模型相关组件
from ding.envs.env_wrappers import  ScaledFloatFrameWrapper,EvalEpisodeReturnEnv, TimeLimitWrapper,MaxAndSkipWrapper,FrameStackWrapper,WarpFrameWrapper
from ding_dqn_config import main_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer

def wrapped_grid_env():
    return DingEnvWrapper(
        env = GridWorldEnv(),
        cfg={
            'env_wrapper': [
                lambda env: MaxAndSkipWrapper(env, skip=4),
                lambda env: ScaledFloatFrameWrapper(env),
                lambda env: FrameStackWrapper(env, n_frames=1),
                lambda env: EvalEpisodeReturnEnv(env),
            ]
        }
    )
def main(main_config: EasyDict, create_config: EasyDict, ckpt_path: str):
    main_config.exp_name = 'ding_test' # 设置本次部署运行的实验名，即为将要创建的工程文件夹名
    cfg = main_config
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        save_cfg=True
    )
    cfg.policy.load_path = ckpt_path
    env = BaseEnvManager(
            env_fn=[wrapped_grid_env for _ in range(1)],
            cfg=cfg.env.manager
        )
    model = DQN(**cfg.policy.model) # 导入模型配置，实例化DQN模型
    policy = DQNPolicy(cfg.policy, model=model) # 导入策略配置，导入模型，实例化DQN策略，并选择评价模式
    policy.eval_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location='cpu'))

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator.eval()

if __name__ == "__main__":
    main(main_config=main_config, create_config=None, ckpt_path='/remote-home/ums_qipeng/WatchRouteProblem/ding_dqn_stack_230725_031510/ckpt/iteration_880000.pth.tar')