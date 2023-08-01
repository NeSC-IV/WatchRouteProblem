from easydict import EasyDict

mario_dqn_config = dict(
    exp_name='ding_dqn_stack',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=1,
        stop_value=100000,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[1, 100, 100],
            action_shape=8,
            encoder_hidden_size_list=[128, 128, 256],
            dueling=True,
        ),
        nstep=10,
        discount_factor=0.95,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
        ),
        collect=dict(n_sample=96, ),
        eval=dict(evaluator=dict(eval_freq=2000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
mario_dqn_config = EasyDict(mario_dqn_config)
main_config = mario_dqn_config
mario_dqn_create_config = dict(
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
mario_dqn_create_config = EasyDict(mario_dqn_create_config)
create_config = mario_dqn_create_config