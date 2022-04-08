import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque
import copy
import pybullet_envs

learning_rate = 1e-4
buffer_size = 100000
batch_size = 64
gamma = 0.99
tau = 0.1
episodes_num = 200
max_episode_len = 256
tar_steps = 10
tar_actor_steps = 20
beta_start = 0.4
beta_change = 10000
ou_theta = 0.15
ou_sigma = 0.2
ou_mu = 0.



class Actor(tf.keras.Model):
    def __init__(self,actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=400, activation=tf.nn.relu)
        self.dense_act1 = tf.keras.layers.Dense(units=300, activation=tf.nn.relu)
        self.dense_act2 = tf.keras.layers.Dense(units=actions, activation=tf.nn.tanh)

    def call(self, inputs):
        x = self.dense1(inputs)
        policy = self.dense_act1(x)
        policy = self.dense_act2(policy)
        return policy

class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=400, activation=tf.nn.relu)
        self.dense_cri1 = tf.keras.layers.Dense(units=300, activation=tf.nn.relu)
        self.dense_cri2 = tf.keras.layers.Dense(units=1)

    def call(self, inputs,a):
        x = self.dense1(inputs)
        x = tf.concat([x,a],axis=1)
        value = self.dense_cri1(x)
        value = self.dense_cri2(value)        
        return value



class replay_buffer():
    def __init__(self,size) -> None:
        self.buffer = deque(maxlen=buffer_size)
        self.iter = 0
        self.pri = np.zeros((buffer_size,))
        self.beta = beta_start

    def update_beta(self,i):
        self.beta = min(1.0,beta_start+i*(1.0-beta_start)/beta_change)

    def store(self,exp):
        max_pri = self.pri.max() if self.buffer else 1.
        if self._len() < buffer_size:
            self.buffer.append(exp)
        else :
            self.buffer[self.iter] = exp
        self.pri[self.iter] = max_pri
        self.iter = (self.iter+1)%buffer_size
    
    def update_pio(self,loss,indices):
        for idx,pri in zip(indices,loss):
            self.pri[idx] = 1e-5+pri

    def _len(self):
        return len(self.buffer)

    def append(self,exp):
        self.buffer.append(exp)

    def sample(self,batch_size):
        probs = copy.deepcopy(self.pri[:self._len()])
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size,p = probs)
        exp = [self.buffer[idx] for idx in indices]
        batch_state = np.array([x[0] for x in exp])
        batch_action = np.array([x[1] for x in exp])
        batch_reward = np.array([x[2] for x in exp])
        batch_next_state = np.array([x[3] for x in exp])
        batch_done = np.array([x[4] for x in exp])
        total = self._len()
        weights = (total * probs[indices])**(-self.beta)
        weights/=weights.max()



        return indices,batch_state, batch_action, batch_reward, batch_next_state, batch_done ,weights

def test_net(count=2):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        state = env.reset()
        while True:
            inputs = np.expand_dims(state, axis=0)
            action = actor(inputs).numpy().squeeze()
            state,r,done,_ = env.step(action)
            rewards += r
            steps +=1
            if done:
                break
    return rewards / count, steps / count

env = gym.make("MinitaurBulletEnv-v0")

actor = Actor(actions=env.action_space.shape[0])
critic = Critic()
tar_actor = Actor(actions=env.action_space.shape[0])
tar_critic = Critic()

tar_actor.set_weights(actor.get_weights())
tar_critic.set_weights(critic.get_weights())


opt = tf.keras.optimizers.Adam(learning_rate=0.01)
per_buffer = replay_buffer(size = buffer_size)
sum=0
for episode in range(episodes_num):
    state = env.reset()
    for i in range(max_episode_len):
        per_buffer.update_beta(i)
        env.render()
        action = tar_actor(tf.expand_dims(state,0)).numpy().squeeze()
        action += ou_theta * (ou_mu - action)
        action += ou_sigma * np.random.normal(size=action.shape)
        action = np.clip(action,-1,1)

        next_s,r,is_done,_ = env.step(action=action)
        
        
        per_buffer.store((state,action,r,next_s,is_done))
        if is_done:
            r,steps = test_net()
            print("episode %4d, reward %.3f, step %4d" % (episode, r, steps)) 
            break
        if per_buffer._len() >= batch_size:
            indices,batch_state, batch_action, batch_reward, batch_next_state, batch_done, weights = per_buffer.sample(batch_size)
            with tf.GradientTape() as tape:
                y_pred = critic(batch_state,batch_action)
                next_a = tar_actor(batch_state)
                next_q = tar_critic(batch_next_state,next_a)
                y = np.expand_dims(batch_reward,axis=1)  + gamma * next_q
                l = (y-y_pred)**2
                loss = tf.multiply(l,weights)
            grads = tape.gradient(loss,critic.variables)
            opt.apply_gradients(grads_and_vars=zip(grads, critic.variables)) 
            per_buffer.update_pio(abs(y-y_pred),indices)


            with tf.GradientTape() as tape:
                a = actor(batch_state)
                loss = - tf.reduce_mean(critic(batch_state,a))
            
            grads = tape.gradient(loss,actor.variables)
            opt.apply_gradients(grads_and_vars=zip(grads, actor.variables)) 

            if i % tar_steps ==0:
                net_w = critic.get_weights()
                tar_w = tar_critic.get_weights()
                for i,(nw,tarw) in enumerate(zip(net_w,tar_w)):
                    tar_w[i] = tau * nw + (1-tau)*tarw
                tar_critic.set_weights(tar_w)

            if i % tar_actor_steps ==0:
                net_w = actor.get_weights()
                tar_w = tar_actor.get_weights()
                for i,(nw,tarw) in enumerate(zip(net_w,tar_w)):
                    tar_w[i] = tau * nw + (1-tau)*tarw
                tar_actor.set_weights(tar_w)
                
        state = next_s

print(sum)


