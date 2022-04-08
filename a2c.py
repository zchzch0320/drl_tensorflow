from turtle import st
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

class Actor(tf.keras.Model):
    def __init__(self,actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense_act1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense_act2 = tf.keras.layers.Dense(units=actions)

    def call(self, inputs):
        x = self.dense1(inputs)
        policy = self.dense_act1(x)
        policy = self.dense_act2(policy)
        return policy

class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense_cri1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense_cri2 = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        x = self.dense1(inputs)
        value = self.dense_cri1(x)
        value = self.dense_cri2(value)        
        return value

if __name__ == '__main__':
    env = gym.make('CartPole-v1')       
    action_nums = env.action_space.n
    actor = Actor(actions= action_nums)
    critic = Critic()
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for episode_id in range(num_episodes):
        states, actions, rewards, v_list =[],[],[],[]
        state = env.reset()             # 初始化环境，获得初始状态
        for t in range(max_len_episode):
            env.render()                                # 对当前帧进行渲染，绘图到屏幕
            inputs = np.expand_dims(state, axis=0)
            action = actor(inputs)
            v = critic(inputs)
            action = tf.nn.softmax(action).numpy()[0]
            action = np.random.choice(action_nums, p=action)
            next_s,r,done,_ = env.step(action)
            r = -10 if done else r



            y = r+gamma*critic(np.expand_dims(next_s, axis=0))*(1-done)
            with tf.GradientTape() as tape:
                value_v = critic(np.expand_dims(state, axis=0))
                value_v = tf.squeeze(value_v)
                l = tf.subtract(y,value_v)
                loss_value = tf.reduce_mean(l**2)
            grads = tape.gradient(loss_value,critic.variables)
            opt.apply_gradients(grads_and_vars=zip(grads, critic.variables)) 


            with tf.GradientTape() as tape:
                value_v = critic(np.expand_dims(state, axis=0))
                logits_v = actor(np.expand_dims(state, axis=0))
                value = tf.stop_gradient(value_v)
                log_prob = tf.nn.log_softmax(logits_v)
                adv_v = y - value
                loss_policy = -tf.reduce_mean(adv_v*log_prob[0][action])
                prob = tf.nn.softmax(logits_v)
                loss_entropy = tf.reduce_mean(tf.reduce_sum(prob*log_prob,axis=1))
                loss = loss_policy + beta_entropy * loss_entropy
            grads = tape.gradient(loss,actor.variables)
            opt.apply_gradients(grads_and_vars=zip(grads, actor.variables)) 






            states.append(state)
            actions.append(action)
            rewards.append(r)
            v_list.append(v[0][0].numpy())
            if done:
                print("episode %4d, score %4d" % (episode_id, t))
                break
            state = next_s
        # y_list = []
        # for i in range(1,len(states)):
        #     a = rewards[i-1]+gamma*v_list[i]
        #     y_list.append(a)
        # y_list.append(rewards[-1])
        # y_list = np.array(y_list,dtype=np.float32)


        
        # with tf.GradientTape() as tape:
        #     s = tf.constant(states)
        #     y = tf.constant(y_list)
        #     value_v = critic(np.expand_dims(s, axis=0))
        #     value_v = tf.squeeze(value_v)
        #     l = y - value_v
        #     print(l.shape)
        #     loss_value = tf.reduce_mean(l**2)
        # grads = tape.gradient(loss_value,critic.variables)
        # opt.apply_gradients(grads_and_vars=zip(grads, critic.variables)) 


        # with tf.GradientTape() as tape:
        #     s = tf.constant(states)
        #     a = tf.constant(actions)
        #     y = tf.constant(y_list)
        #     value_v = critic(np.expand_dims(s, axis=0))
        #     logits_v = actor(np.expand_dims(s, axis=0))
        #     value = tf.stop_gradient(value_v)
        #     log_prob = tf.nn.log_softmax(logits_v)
        #     adv_v = y - value
        #     l = []
        #     for i in range(len(s)):
        #         l.append(log_prob[0][i,a[i]])
        #     loss_policy = -tf.reduce_mean(adv_v*l)
        #     print(loss_policy)
        #     prob = tf.nn.softmax(logits_v)
        #     loss_entropy = tf.reduce_mean(tf.reduce_sum(prob*log_prob,axis=1))
        #     loss = loss_policy + beta_entropy * loss_entropy
        # grads = tape.gradient(loss,actor.variables)
        # opt.apply_gradients(grads_and_vars=zip(grads, actor.variables)) 
