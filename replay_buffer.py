import random
import numpy as np
import torch

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer_state = []
        self.buffer_action = []
        self.buffer_next_state = []
        self.buffer_reward = []
        self.buffer_done = []
        self.idx = 0

    def store(self, state, action, next_state, reward, done):

        if len(self.buffer_state) < self.capacity:
            self.buffer_state.append(state)
            self.buffer_action.append(action)
            self.buffer_next_state.append(next_state)
            self.buffer_reward.append(reward)
            self.buffer_done.append(done)
        else:
            self.buffer_state[self.idx] = state
            self.buffer_action[self.idx] = action
            self.buffer_next_state[self.idx] = next_state
            self.buffer_reward[self.idx] = reward
            self.buffer_done[self.idx] = done

        self.idx = (self.idx+1)%self.capacity # for circular memory

    def sample(self, batch_size, device):
        
        indices_to_sample = random.sample(range(len(self.buffer_state)), batch_size)

        states_np = np.array(self.buffer_state)[indices_to_sample]
        actions_np = np.array(self.buffer_action)[indices_to_sample]
        next_states_np = np.array(self.buffer_next_state)[indices_to_sample]
        rewards_np = np.array(self.buffer_reward)[indices_to_sample]
        dones_np = np.array(self.buffer_done)[indices_to_sample]

        states = torch.as_tensor(states_np, dtype=torch.float32).to(device)
        actions = torch.as_tensor(actions_np, dtype=torch.int64).unsqueeze(-1).to(device) # just index
        next_states = torch.as_tensor(next_states_np, dtype=torch.float32).to(device)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32).to(device)
        dones = torch.as_tensor(dones_np, dtype=torch.bool).to(device)

        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.buffer_state)