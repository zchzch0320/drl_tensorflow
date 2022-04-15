import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque
import math
import pybullet_envs

learning_rate = 3e-4
buffer_size = 100000
batch_size = 128
gamma = 0.99
tau = 0.01
episodes_num = 2000
max_episode_len = 2048
tar_steps = 10
alpha = 0.1
reward_scale = 1
update_fre = 4

class Actor(tf.keras.Model):
    def __init__(self,actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense_mu1 = tf.keras.layers.Dense(units=actions, activation=tf.nn.tanh)
        self.dense_var1 = tf.keras.layers.Dense(units=actions, activation=tf.nn.tanh)
        

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        mu = self.dense_mu1(x)
        var = self.dense_var1(x)
        return mu,var

    def act(self, inputs):
        mu, std = self.call(inputs)
        std = tf.clip_by_value(std,clip_value_min=-20,clip_value_max=2)
        var = tf.exp(std)
        action = tf.tanh(tf.random.normal(shape=[action_shape],mean=mu,stddev=var))
        return action

    def act_and_logprob(self,state):
        mu, std = self.call(state)
        std = tf.clip_by_value(std,clip_value_min=-20,clip_value_max=2)
        var = tf.exp(std)
        noise = tf.random.normal(shape=[action_shape])
        action = mu + noise * var
        act_t = tf.tanh(action)
        log_prob = cal_log_prob(action,mu,std)
        log_prob -= np.log(1-act_t**2+1e-6)
        return act_t,log_prob

class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=400, activation=tf.nn.relu)
        self.dense_cri1 = tf.keras.layers.Dense(units=300, activation=tf.nn.relu)
        self.dense_cri2 = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        x = self.dense1(inputs)
        value = self.dense_cri1(x)
        value = self.dense_cri2(value)        
        return value

class Twin_q(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense_cri1 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense_cri2 = tf.keras.layers.Dense(units=1)

        self.dense2 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense_cri3 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense_cri4 = tf.keras.layers.Dense(units=1)

    def call(self, s,a):
        inputs = tf.concat([s,a],axis=1)
        x = self.dense1(inputs)
        value1 = self.dense_cri1(x)
        value1 = self.dense_cri2(value1)

        x = self.dense2(inputs)
        value2 = self.dense_cri3(x)
        value2 = self.dense_cri4(value2)
        return value1, value2

class replay_buffer():
    def __init__(self,size) -> None:
        self.buffer = deque(maxlen=buffer_size)
        self.iter = 0


    def store(self,exp):
        if self._len() < buffer_size:
            self.buffer.append(exp)
        else :
            self.buffer[self.iter] = exp
        self.iter = (self.iter+1)%buffer_size
    

    def _len(self):
        return len(self.buffer)

    def append(self,exp):
        self.buffer.append(exp)

    def sample(self,batch_size):
        exp = random.sample(self.buffer,batch_size)
        batch_state = np.array([x[0] for x in exp])
        batch_action = np.array([x[1] for x in exp])
        batch_reward = np.array([x[2] for x in exp])
        batch_next_state = np.array([x[3] for x in exp])
        batch_done = np.array([x[4] for x in exp])



        return batch_state, batch_action, batch_reward, batch_next_state, batch_done

def test_net(count=2):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        state = env.reset()
        while True:
            inputs = np.expand_dims(state, axis=0)
            action,_ = actor.act_and_logprob(inputs)
            action = action.numpy().squeeze()
            state,r,done,_ = env.step(action)
            rewards += r
            steps +=1
            if done:
                break
    return rewards / count, steps / count

def cal_log_prob(x,mu,var):
    p1 = ((x-mu)**2) / (2*np.exp(var))
    p2 = np.log(np.sqrt(2*math.pi)) + var
    return p1 + p2

env = gym.make("HalfCheetahBulletEnv-v0")
action_shape = env.action_space.shape[0]
actor = Actor(actions=action_shape)
critic = Critic()
twin_q = Twin_q()
tar_critic = Critic()

tar_critic.set_weights(critic.get_weights())


opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
buffer = replay_buffer(size = buffer_size)
action_range = [env.action_space.low, env.action_space.high]
#print(action_range)

for episode in range(episodes_num):
    state = env.reset()
    for i in range(max_episode_len):
        env.render()
        mu = actor.act(tf.expand_dims(state,0))
        action = mu.numpy().squeeze()
        #action =  action * (action_range[1] - action_range[0]) / 2.0 +  (action_range[1] + action_range[0]) / 2.0
        
        next_s,r,is_done,_ = env.step(action=action)
        
        
        buffer.store((state,action,r*reward_scale,next_s,is_done))
        if is_done:
            r,steps = test_net()
            print("episode %4d, reward %.3f, step %4d" % (episode, r, steps)) 
            break
        if (buffer._len() > batch_size) and (i%update_fre == 0):
            batch_state, batch_action, batch_reward, batch_next_state, batch_done= buffer.sample(batch_size)

            with tf.GradientTape() as tape:
                y_pred = critic(batch_state)
                action,batch_prob = actor.act_and_logprob(batch_state)
                q1,q2 = twin_q(batch_state,action)
                q = tf.concat([q1,q2],axis = 1)
                print(q1.shape,q2.shape,q.shape)
                q = tf.reduce_min(q,axis = 1)
                batch_prob = tf.reduce_sum(batch_prob,axis=1)
                print(q.shape)
                #print('aa',q,batch_prob)
                q -= batch_prob * alpha
                l = ((y_pred-q)**2)/2
                loss = tf.reduce_mean(l)
            grads = tape.gradient(loss,critic.variables)
            opt.apply_gradients(grads_and_vars=zip(grads, critic.variables)) 

            with tf.GradientTape() as tape:
                y_pred1,y_pred2 = twin_q(batch_state,batch_action)
                y = np.expand_dims(batch_reward,axis=1) + (1-batch_done) * gamma * tar_critic(batch_next_state)
                l = ((y-y_pred1)**2+(y-y_pred2)**2)/4
                loss = tf.reduce_mean(l)
            grads = tape.gradient(loss,twin_q.variables)
            opt.apply_gradients(grads_and_vars=zip(grads, twin_q.variables)) 

            with tf.GradientTape() as tape:
                action,batch_prob = actor.act_and_logprob(batch_state)
                q,_ = twin_q(batch_state,action)
                l = alpha * batch_prob - q
                loss = - tf.reduce_mean(l)
            grads = tape.gradient(loss,actor.variables)
            opt.apply_gradients(grads_and_vars=zip(grads, actor.variables)) 
            
            if i % tar_steps ==0:
                net_w = critic.get_weights()
                tar_w = tar_critic.get_weights()
                for i,(nw,tarw) in enumerate(zip(net_w,tar_w)):
                    tar_w[i] = tau * nw + (1-tau)*tarw
                tar_critic.set_weights(tar_w)

                
        state = next_s
