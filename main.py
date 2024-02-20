import os
import csv
import gymnasium as gym
from gymnasium.utils import seeding
import argparse
import numpy as np
import pickle
from dqn_agent import DQNAgent
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fill_memory(env, dqn_agent, num_memory_fill_eps):

    for _ in range(num_memory_fill_eps):
        done = False
        state = env.reset()[0]

        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            dqn_agent.memory.store(state=state, 
                                action=action, 
                                next_state=next_state, 
                                reward=reward, 
                                done=terminated)

def train(env, dqn_agent, num_train_eps, num_memory_fill_eps, update_frequency, batchsize, results_basepath, render=False):
    """
    env: gym.Env
        Instance of the environment used for training
    dqn_agent: DQNAgent
        Agent to be trained
    num_train_eps: int
        Number of episodes of training to be performed
    num_memory_fill_eps: int
        Number of episodes of random interaction to be performed
    update_frequency: int
        Number of steps after which the target models must be updated
    batchsize: int
        Number of transitions to be sampled from the replay buffer to perform a learning step
    results_basepath: str
        Location where models and other result files are saved
    render: bool
        Whether to create a pop-up window display the interaction of the agent with the environment
    """

    fill_memory(env, dqn_agent, num_memory_fill_eps)
    print('Memory filled. Current capacity: ', len(dqn_agent.memory))
    
    reward_history = []
    epsilon_history = []
    step_cnt = 0
    best_score = -np.inf

    for ep_cnt in range(num_train_eps):
        epsilon_history.append(dqn_agent.epsilon)

        done = False
        state = env.reset()[0]

        ep_score = 0

        while not done:
            if render:
                env.render()

            action = dqn_agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # done = terminated
            dqn_agent.memory.store(state=state, action=action, next_state=next_state, reward=reward, done=terminated)

            dqn_agent.learn(batchsize=batchsize)

            if step_cnt % update_frequency == 0:
                dqn_agent.update_target_net()

            state = next_state
            ep_score += reward
            step_cnt += 1

        dqn_agent.update_epsilon()

        reward_history.append(ep_score)
        current_avg_score = np.mean(reward_history[-100:]) # moving average of last 100 episodes

        print('Ep: {}, Total Steps: {}, Ep: Score: {}, Avg score: {}; Epsilon: {}'
              .format(ep_cnt, step_cnt, ep_score, current_avg_score, epsilon_history[-1]))
        
        if current_avg_score >= best_score:
            dqn_agent.save_model('{}/dqn_model'.format(results_basepath))
            best_score = current_avg_score

    with open('{}/train_reward_history.pkl'.format(results_basepath), 'wb') as f:
        pickle.dump(reward_history, f)

    with open('{}/train_epsilon_history.pkl'.format(results_basepath), 'wb') as f:
        pickle.dump(epsilon_history, f)


def test(env, dqn_agent, num_test_eps, seed, results_basepath, render=True):
    """
    env: gym.Env
        Instance of the environment used for training
    dqn_agent: DQNAgent
        Agent to be trained
    num_test_eps: int
        Number of episodes of testing to be performed
    seed: int
        Value of the seed used for testing
    results_basepath: str
        Location where models and other result files are saved
    render: bool
        Whether to create a pop-up window display the interaction of the agent with the environment
    """

    step_cnt = 0
    reward_history = []

    for ep in range(num_test_eps):
        score = 0
        done = False
        state = env.reset()
        while not done:

            if render:
                env.render()

            action = dqn_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            score += reward
            state = next_state
            step_cnt += 1

        reward_history.append(score)
        print('Ep: {}, Score: {}'.format(ep, score))

    with open('{}/test_reward_history_{}.pkl'.format(results_basepath, seed), 'wb') as f:
        pickle.dump(reward_history, f)
        
"""
if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
num-train-eps = 2000
num-test-eps = 100
num-memory-fill-eps = 20
update-frequency = 1000
train-seed = 12321
test-seed = [456, 12, 985234, 123, 3202]
discount = 0.99
lr = 1e-3
eps-max = 1.0
eps-min = 0.01
eps-decay = 0.995
batchsize = 64
memory-capacity = 10000
results-folder = str
env-name = 'LunarLander-v2'
train = 
render = 
    parser.add_argument('--num-train-eps', type=int, default=2000, help='specify the max episodes to train for (counts even the period of memory initialisation)')
    parser.add_argument('--num-test-eps', type=int, default=100, help='specify the max episodes to test for')
    parser.add_argument('--num-memory-fill-eps', type=int, default=20, help='number of timesteps after which learning should start (used to initialise the memory)')
    parser.add_argument('--update-frequency', type=int, default=1000, help='how frequently should the target network by updated')
    parser.add_argument('--train-seed', type=int, default=12321, help='seed to use while training the model')
    parser.add_argument('--test-seed', type=int, nargs='+', default=[456, 12, 985234, 123, 3202], help='seeds to use while testing the model')
    parser.add_argument('--discount', type=float, default=0.99, help='discounting value to determine how far-sighted the agent should be')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--eps-max', type=float, default=1.0, help='max value for epsilon')
    parser.add_argument('--eps-min', type=float, default=0.01, help='min value for epsilon')
    parser.add_argument('--eps-decay', type=float, default=0.995, help='amount by which to decay the epsilon value for annealing strategy')
    parser.add_argument('--batchsize', type=int, default=64, help='number of samples to draw from memory for learning')
    parser.add_argument('--memory-capacity', type=int, default=10000, help='define the capacity of the replay memory')
    parser.add_argument('--results-folder', type=str, help='folder where the models and results of the current run must by stored')
    parser.add_argument('--env-name', type=str, default='LunarLander-v2', help='environment in which to train the agent')
    parser.add_argument('--train', action='store_true', help='train the agent')
    parser.add_argument('--render', action='store_true', help='render the interaction')
    args = parser.parse_args()

    if args.train:

        os.environ['PYTHONHASHSEED']=str(args.train_seed)
        np.random.seed(args.train_seed)
        torch.manual_seed(args.train_seed)

        env = gym.make(args.env_name)
        env.seed(args.train_seed)
        env.action_space.np_random.seed(args.train_seed)

        if args.results_folder is None:
            args.results_folder = "results/{}_epsmax{}_epsmin{}_epsdec{}_batchsize{}_memcap{}_updfreq{}".format(args.env_name, args.eps_max, args.eps_min, args.eps_decay, args.batchsize, args.memory_capacity, args.update_frequency)

        os.makedirs(args.results_folder, exist_ok=True)

        dqn_agent = DQNAgent(device, 
                                env.observation_space.shape[0], 
                                env.action_space.n, 
                                discount=args.discount, 
                                eps_max=args.eps_max, 
                                eps_min=args.eps_min, 
                                eps_decay=args.eps_decay,
                                memory_capacity=args.memory_capacity,
                                lr=args.lr,
                                train_mode=True)

        train(env=env, 
                dqn_agent=dqn_agent, 
                results_basepath=args.results_folder, 
                num_train_eps=args.num_train_eps, 
                num_memory_fill_eps=args.num_memory_fill_eps, 
                update_frequency=args.update_frequency,
                batchsize=args.batchsize)

        env.close()
    
    else:
        for idx, seed in enumerate(args.test_seed):
            print("Testing {}/{}, seed = {}".format(idx+1, len(args.test_seed), seed))
            os.environ['PYTHONHASHSEED']=str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            env = gym.make(args.env_name)
            env.seed(seed)
            env.action_space.np_random.seed(seed)

            dqn_agent = DQNAgent(device, 
                                env.observation_space.shape[0], 
                                env.action_space.n, 
                                discount=args.discount, 
                                eps_max=0.0, # epsilon values should be zero to ensure no exploration in testing mode
                                eps_min=0.0, 
                                eps_decay=0.0,
                                train_mode=False)
            dqn_agent.load_model('{}/dqn_model'.format(args.results_folder))

            test(env=env, dqn_agent=dqn_agent, num_test_eps=args.num_test_eps, 
                 seed=seed, results_basepath=args.results_folder, render=args.render)

            env.close()

"""

num_train_eps = 2000
num_test_eps = 100
num_memory_fill_eps = 20
update_frequency = 1000
train_seed = 12321
test_seed = [456, 12, 985234, 123, 3202]
discount = 0.99
lr = 1e-3
eps_max = 1.0
eps_min = 0.01
eps_decay = 0.995
batchsize = 64
memory_capacity = 10000
results_folder = 'simulation'
env_name = 'LunarLander-v2'
#train = True
render = False

if __name__ ==  '__main__':

    os.environ['PYTHONHASHSEED']=str(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)

    env = gym.make(env_name)
    seeding.np_random(train_seed)
    env.reset(seed=train_seed)
    env.action_space.seed(train_seed)

    os.makedirs(results_folder, exist_ok=True)

    dqn_agent = DQNAgent(device, 
                                env.observation_space.shape[0], 
                                env.action_space.n, 
                                discount=discount, 
                                eps_max=eps_max, 
                                eps_min=eps_min, 
                                eps_decay=eps_decay,
                                memory_capacity=memory_capacity,
                                lr=lr,
                                train_mode=True)

    train(env=env, 
                dqn_agent=dqn_agent, 
                results_basepath=results_folder, 
                num_train_eps=num_train_eps, 
                num_memory_fill_eps=num_memory_fill_eps, 
                update_frequency=update_frequency,
                batchsize=batchsize)

    env.close()
