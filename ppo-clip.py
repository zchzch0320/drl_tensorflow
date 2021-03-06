import tensorflow as tf
import numpy as np
import gym
import math
import pybullet_envs

num_episodes = 200            
num_exploration_episodes = 100 
max_len_episode = 2048   
batch_size = 64  
actor_learning_rate = 3e-5  
critic_learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
beta_entropy = 0.01
ppo_epoches = 3
ppo_eps = 0.2
buffer_size = 2048

def test_net(count=5):
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

def calc_adv_ref(rewards,states,dones):
    states = tf.constant(states)
    values = critic(states).numpy().squeeze()
    last_gae = 0.
    advs = []
    refs = []
    gae_gamma = gamma*gae_lambda
    for v,r,next_v,done in zip(reversed(values[:-1]), reversed(rewards[:-1]),reversed(values[1:]),reversed(dones[:-1])):
        if done:
            last_gae = delta = r-v
        else:
            delta = r + gamma * next_v - v
            last_gae = delta + gae_gamma * last_gae
        advs.append(last_gae)
        refs.append(last_gae+v)
    return list(reversed(advs)),list(reversed(refs))

def calc_logprob(mu,actions,logstd):
    p1 = -(mu-actions)**2 / (2* tf.clip_by_value(tf.exp(logstd),clip_value_min=1e-3,clip_value_max=1e5))
    p2 = - tf.math.log((2 * math.pi * tf.exp(logstd))**(1/2))
    return p1+p2



class Actor(tf.keras.Model):
    def __init__(self,actions):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=64, activation=tf.nn.tanh)
        self.dense_act1 = tf.keras.layers.Dense(units=64, activation=tf.nn.tanh)
        self.dense_act2 = tf.keras.layers.Dense(units=actions)
        self.logstd = tf.Variable(tf.zeros(actions))

    def call(self, inputs):
        x = self.dense1(inputs)
        policy = self.dense_act1(x)
        policy = self.dense_act2(policy)
        return policy

class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense_cri1 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense_cri2 = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        x = self.dense1(inputs)
        value = self.dense_cri1(x)
        value = self.dense_cri2(value)        
        return value

if __name__ == '__main__':
    env = gym.make("HalfCheetahBulletEnv-v0")
    action_nums = env.action_space.shape[0]
    actor = Actor(actions= action_nums)
    critic = Critic()
    cri_opt = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)
    act_opt = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)

    states, actions, rewards, dones =[],[],[],[]
    for episode_id in range(num_episodes):
        
        state = env.reset()
        for t in range(max_len_episode):
            env.render()
            inputs = np.expand_dims(state, axis=0)
            action = actor(inputs).numpy().squeeze()
            action += np.random.normal(size=action.shape)
            action = np.clip(action, -1, 1)
            next_s,r,done,_ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(r)
            dones.append(done)
            if done:
                r, step = test_net()
                print("episode %4d,reward %.3f, steps %d" % (episode_id, r, step))
                break
            state = next_s

            if len(states)<buffer_size:
                continue


            adv_v, ref_v = calc_adv_ref(rewards,states,dones)
            mu_v = actor(tf.constant(states))
            old_logprob = calc_logprob(mu_v,tf.constant(actions),actor.logstd)
            adv_v = tf.constant(adv_v)
            adv_v -= tf.reduce_mean(adv_v)
            adv_v /= tf.math.reduce_std(adv_v)
            states = states[:-1]
            actions = actions[:-1]
            old_logprob = old_logprob[:-1]
            for i in range(ppo_epoches):
                for batch_ofs in range(0,len(states),batch_size):
                    batch_l = batch_ofs + batch_size
                    state_v = states[batch_ofs:batch_l]
                    
                    actions_v = actions[batch_ofs:batch_l]
                    batch_adv = adv_v[batch_ofs:batch_l]
                    batch_ref = ref_v[batch_ofs:batch_l]
                    batch_old_logprob = old_logprob[batch_ofs:batch_l]

                    with tf.GradientTape() as tape:
                        s = tf.constant(state_v)
                        r = tf.constant(batch_ref,dtype=np.float32)
                        v = critic(s)
                        v = tf.squeeze(v)
                        loss = tf.reduce_mean((v-r)**2)
                    grads = tape.gradient(loss,critic.variables)
                    cri_opt.apply_gradients(grads_and_vars=zip(grads, critic.variables))
                    
                    with tf.GradientTape() as tape:
                        s = tf.constant(state_v)
                        a = tf.constant(actions_v)
                        mu = actor(s)
                        logprob_pi = calc_logprob(mu,a,actor.logstd)
                        ratio = tf.exp(logprob_pi - batch_old_logprob)
                        clip_ratio = tf.clip_by_value(ratio,1-ppo_eps,1+ppo_eps)
                        ratio = tf.cast(ratio,dtype=np.float64)
                        clip_ratio = tf.cast(clip_ratio,dtype=np.float64)
                        batch_adv = tf.constant(batch_adv)
                        batch_adv = tf.expand_dims(batch_adv,axis=1)
                        a1 = batch_adv*ratio
                        a2 = batch_adv*clip_ratio
                        loss_policy = - tf.reduce_mean(tf.reduce_min(tf.stack([a1,a2]),0))
                        prob_pi = tf.exp(logprob_pi)
                        loss_entropy = tf.reduce_mean(tf.reduce_sum(prob_pi*logprob_pi,axis=1))
                        loss_entropy = tf.cast(loss_entropy,dtype=np.float64)
                        loss = loss_policy + beta_entropy * loss_entropy

                    grads = tape.gradient(loss,actor.variables)
                    act_opt.apply_gradients(grads_and_vars=zip(grads, actor.variables))

            states.clear()
            actions.clear()
            rewards.clear()
            dones.clear()




