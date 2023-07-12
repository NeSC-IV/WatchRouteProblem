from easydict import EasyDict

grid_dqn_config = dict(
    exp_name='GridWorld_dqn_seed1',
    env=dict(
        collector_env_num=4,
        evaluator_env_num=4,
        n_evaluator_episode=8,
        stop_value=100000,
        channels_first=True
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[4, 200, 200],
            action_shape=8,
            encoder_hidden_size_list=[128, 128, 256],
            dueling=True,
        ),
        nstep=3,
        discount_factor=0.97,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=5000,
        ),
        collect=dict(n_sample=96, ),
        eval=dict(evaluator=dict(eval_freq=200, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.2,
                decay=250000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
grid_dqn_config = EasyDict(grid_dqn_config)
main_config = grid_dqn_config
gird_dqn_create_config = dict(
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
gird_dqn_create_config = EasyDict(gird_dqn_create_config)
create_config = gird_dqn_create_config