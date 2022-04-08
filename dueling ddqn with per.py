from sympy import Lambda
import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque
import copy


learning_rate = 1e-3
buffer_size = 1024
batch_size = 32
gamma = 1.
tau = 0.1
initial_epsilon = 1.
final_epsilon = 0.01
episodes_num = 200
explore_episodes_num = 100
max_episode_len = 256
tar_steps = 10
beta_start = 0.4
beta_change = 10000

class qnet(tf.keras.Model):
    def __init__(self,action_dim) -> None:
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units = 16,activation = tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units = 16,activation = tf.nn.relu)
        
        self.dense3 = tf.keras.layers.Dense(units = 1)
        self.dense4 = tf.keras.layers.Dense(units = action_dim)


        #self.dense3 = tf.keras.layers.Dense(units = action_dim)

    def call(self,x):
        x=self.dense1(x)
        x=self.dense2(x)

        v_s = self.dense3(x)
        #v_s = tf.expand_dims(x[:,0],-1) (v_s)
        a_sa = self.dense4(x)
        a_sa -= tf.reduce_mean(a_sa)
        x = a_sa + v_s


        #x=self.dense3(x)
        return x

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



env = gym.make('CartPole-v1')
net = qnet(action_dim=env.action_space.n)
tar_net = qnet(action_dim=env.action_space.n)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
per_buffer = replay_buffer(size = buffer_size)
eps = initial_epsilon
sum=0
for episode in range(episodes_num):
    eps = max(final_epsilon , initial_epsilon *( explore_episodes_num-episode) / explore_episodes_num)
    state = env.reset()
    for i in range(max_episode_len):
        per_buffer.update_beta(i)
        env.render()              
        if np.random.random()<eps:
            action = env.action_space.sample()
        else:
            action = tf.argmax(net(np.expand_dims(state, axis=0)),axis=-1).numpy()[0]

        next_s,r,is_done,_ = env.step(action=action)
        
        if is_done:
            r = -10
        else:
            if np.random.random()<eps:
                action2 = env.action_space.sample()
            else:
                action2 = tf.argmax(net(np.expand_dims(next_s, axis=0)),axis=-1).numpy()[0]
            next_s2,r2,is_done2,_ = env.step(action=action2)
            if is_done2:
                r2 = -10
                is_done = is_done2
            r += gamma * r2
            next_s = next_s2


        per_buffer.store((state,action,r,next_s,is_done))
        if is_done:
            print("episode %4d, eps %2f, score %4d" % (episode, eps, i))
            sum += i
            break
        if per_buffer._len() >= batch_size:
            indices,batch_state, batch_action, batch_reward, batch_next_state, batch_done, weights = per_buffer.sample(batch_size)
            #y = batch_reward + gamma **2 *  tf.reduce_max(tar_net(batch_next_state),axis=1)* (1 - batch_done)
            with tf.GradientTape() as tape:
                max_a = tf.one_hot( tf.argmax(net(batch_next_state),axis= 1), depth=2)
                gammaq = tf.reduce_sum(tar_net(batch_next_state)* max_a,axis=1)
                y = batch_reward + gamma **2 *  gammaq* (1 - batch_done)
                y_pred = tf.reduce_sum(net(batch_state) * tf.one_hot(batch_action, depth=2), axis=1)
                l = (y-y_pred)**2
                loss = l *weights
            
            grads = tape.gradient(loss,net.variables)
            opt.apply_gradients(grads_and_vars=zip(grads, net.variables)) 
            per_buffer.update_pio(abs(y-y_pred),indices)

            if i % tar_steps ==0:
                net_w = net.get_weights()
                tar_w = tar_net.get_weights()
                for i,(nw,tarw) in enumerate(zip(net_w,tar_w)):
                    tar_w[i] = tau * nw + (1-tau)*tarw
                tar_net.set_weights(tar_w)
                
        state = next_s

print(sum)
