import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque
import copy

num_episodes = 200              # 游戏训练的总episode数量
num_exploration_episodes = 100  # 探索过程所占的episode数量
max_len_episode = 1000          # 每个episode的最大回合数
batch_size = 32                 # 批次大小
learning_rate = 1e-3            # 学习率
gamma = 0.99                      # 折扣因子
beta_entropy = 0.01

class PGN(tf.keras.Model):
    def __init__(self,actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=actions)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        
        return x

def cal_q(rewards):
    res = []
    s = 0.0
    for r in reversed(rewards):
        s*=gamma
        s+=r
        res.append(s)
    return list(reversed(res))

if __name__ == '__main__':
    env = gym.make('CartPole-v1')       
    action_nums = env.action_space.n
    net = PGN(actions= action_nums)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for episode_id in range(num_episodes):
        states, actions, rewards =[],[],[]
        state = env.reset()  
        for t in range(max_len_episode):
            env.render()
            action = tf.nn.softmax(net(np.expand_dims(state, axis=0))).numpy()[0]
            action = np.random.choice(action_nums, p=action)
            next_s,r,done,_ = env.step(action)
            r = -10 if done else r
            states.append(state)
            actions.append(action)
            rewards.append(r)
            if done:
                print("episode %4d, score %4d" % (episode_id, t))
                break
            state = next_s
        qs = cal_q(rewards)
        with tf.GradientTape() as tape:
            s = tf.constant(states)
            a = tf.constant(actions)
            q = tf.constant(qs)
            log_prob = tf.nn.log_softmax(net(s))
            l = []
            for i in range(len(s)):
                l.append(log_prob[i,a[i]])
            loss_policy = -tf.reduce_mean(q*l)
            prob = tf.nn.softmax(net(s))
            loss_entropy = tf.reduce_mean(tf.reduce_sum(prob*log_prob,axis=1))
            loss = loss_policy + beta_entropy * loss_entropy
        grads = tape.gradient(loss,net.variables)
        opt.apply_gradients(grads_and_vars=zip(grads, net.variables)) 

