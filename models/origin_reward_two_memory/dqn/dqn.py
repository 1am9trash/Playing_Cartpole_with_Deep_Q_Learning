import gym
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import random


class DQN:
    def __init__(self,
                 action_num, state_shape,
                 learning_rate=0.01, reward_decay=0.9,
                 e_greedy_min=0.01, e_greedy_init=0.5, e_greedy_step=100000,
                 memory_size=4096, batch_size=32):
        self.action_num = action_num
        self.state_shape = state_shape
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.e_greedy = e_greedy_init if e_greedy_step is not None else e_greedy_min
        self.e_greedy_min = e_greedy_min
        self.e_greedy_init = e_greedy_init
        self.e_greedy_step = e_greedy_step
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.pos_memory = deque(maxlen=int(0.5*memory_size))
        self.neg_memory = deque(maxlen=memory_size-int(0.5*memory_size))
        self.learning_step = 0
        self.eval_nn = self.__build_nn()

    def __build_nn(self):
        model = Sequential([
            Dense(64, activation="relu", input_shape=self.state_shape),
            Dense(32, activation="relu"),
            Dense(self.action_num, activation="relu"),
        ])
        model.compile(loss="mean_squared_error",
                      optimizer="RMSprop", metrics=["accuracy"])
        return model

    def get_action(self, state, mode):
        if mode == "best":
            actions = self.eval_nn.predict(state)
            return actions.argmax()
        if mode == "e_greedy":
            if self.e_greedy_step is not None and self.e_greedy > self.e_greedy_min:
                self.e_greedy -= (self.e_greedy_init - self.e_greedy_min) / self.e_greedy_step
            else:
                self.e_greedy = self.e_greedy_min
            if np.random.uniform() > self.e_greedy:
                actions = self.eval_nn.predict(state)
                return actions.argmax()
            else: 
                return np.random.randint(0, self.action_num)
        print("ERROR: mode doesn't exist.")
        return 0

    def store(self, state, action, reward, next_state, done):
        if done:
            self.neg_memory.append((state, action, reward, next_state, done))
        else:
            self.pos_memory.append((state, action, reward, next_state, done))

    def replay(self):
        mini_batch = [[], [], [], [], []]
        for data in random.sample(self.pos_memory, int(0.5*self.batch_size)):
            for i in range(5):
                mini_batch[i].append(data[i])
        for data in random.sample(self.neg_memory, self.batch_size-int(0.5*self.batch_size)):
            for i in range(5):
                mini_batch[i].append(data[i])
        state = np.array(mini_batch[:][0])
        action = np.array(mini_batch[:][1]).astype(np.int)
        reward = np.array(mini_batch[:][2])
        next_state = np.array(mini_batch[:][3])
        done = np.array(mini_batch[:][4])
        return state, action, reward, next_state, done

    def learn(self):
        state, action, reward, next_state, done = self.replay()
        q_eval = self.eval_nn.predict(state)
        q_next = self.eval_nn.predict(next_state)
        q_target = q_eval.copy()

        for i in range(self.batch_size):
            if done[i]:
                q_target[i][action[i]] = reward[i]
            else:
                q_target[i][action[i]] = reward[i] + \
                    self.reward_decay * np.max(q_next[i])

        self.eval_nn.fit(state, q_target, epochs=10, verbose=False)

    def load_weights(self, name):
        self.eval_nn.load_weights(name, by_name=True)


random.seed(0)
tf.random.set_seed(0)
np.random.seed(0)
game_name = "CartPole-v0"
env = gym.make(game_name)
env.seed(0)

model = DQN(env.action_space.n, env.observation_space.shape)
learn_cnt = 0
step = 0

for episode in range(1000):
    state = env.reset()
    total_reward = 0
    while True:
        action = model.get_action(state[None, :], "e_greedy")
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        if total_reward < 200:
            model.store(state, action, reward, next_state, done)
        if step > 1000 and episode >= 16 and step % 5 == 0:
            learn_cnt += 1
            model.learn()
        if done:
            break
        state = next_state
        step += 1
    state = env.reset()
    total_reward = 0
    while True:
        # env.render()
        action = model.get_action(state[None, :], "best")
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        if done:
            break
        state = next_state
    if total_reward > 199:
        model.eval_nn.save_weights("./" + str(episode) + ".h5")
    print(episode, ", ", total_reward, ", ", learn_cnt, sep='')

env.close()
