import paddle
import paddle.nn as nn


from .gym_env import GridWorldEnv
class PolicyNet(paddle.nn.Layer):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super(PolicyNet,self).__init__()
        self.online = nn.Sequential(
            nn.Conv2D(in_channels=input_dim, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(28224, 512),
            nn.ReLU(),
            nn.Linear(512, 8),
        )


    def forward(self, input):
        # return paddle.argmax(self.online(input),axis=1)
        out = self.online(input)
        return out

def paddle_gather(x, axis, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if axis < 0:
        axis = len(x.shape) + axis
    nd_index = []
    for k in range(len(x.shape)):
        if k == axis:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            axis_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(axis_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out
class BehaviorClone:
    def __init__(self, state_dim, action_dim, lr):
        self.policy = PolicyNet(state_dim, action_dim)
        self.optimizer = paddle.optimizer.Adam(parameters=self.policy.parameters(),learning_rate=lr)

    def learn(self, states, actions):
        states = paddle.to_tensor(states, dtype="float32").reshape([1,1,200,200])
        actions = paddle.to_tensor(actions).reshape([-1, 1]) 
        log_probs = paddle.log(paddle_gather(self.policy(states),1,actions)) 
        bc_loss = paddle.mean(-log_probs)  # 最大似然估计

        self.optimizer.clear_grad()
        bc_loss.backward()
        self.optimizer.step()

    def take_action(self, state):
        state = paddle.to_tensor(state, dtype="float32").reshape([1,1,200,200])
        probs = self.policy(state)
        action_dist = paddle.distribution.Categorical(probs)
        action = action_dist.sample([1])
        return action.numpy()[0][0]
    
def test_agent(agent, polygon):
    episode_return = 0
    env = GridWorldEnv(polygon)
    state = env.reset()
    done = False
    while not done:
        action = agent.take_action(state)
        next_state, reward, done= env.step(action)
        print(reward)
        state = next_state
        episode_return += reward
    return episode_return