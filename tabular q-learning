import gym
from tensorboardX import SummaryWriter
import collections
env_name = "FrozenLake-v1"
gamma = 0.9
alpha = 0.2
test_episodes = 20
class Agent:
    def __init__(self) -> None:
        self.env = gym.make(env_name)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)
    
    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    def get_best_action(self,state):
        best_q, best_action = None, None
        for action in range(self.env.action_space.n):
            q = self.values[(state,action)]
            if best_q is None or q > best_q:
                best_q = q
                best_action = action


        return best_q,best_action

    def q_update(self,s,a,r,next_s):
        best_q,_ =self.get_best_action(next_s)
        self.values[(s,a)]=(1-alpha)*self.values[(s,a)] +alpha*(r + gamma*best_q)

    def play_episode(self,env):
        sum = 0
        state = env.reset()
        while True:
            _, action = self.get_best_action(state)
            next_s,r,is_done,_= env.step(action)
            sum += r
            if is_done:
                break
            else:
                state = next_s
        return sum



if __name__ =='__main__':
    env = gym.make(env_name)
    agent = Agent()
    writter = SummaryWriter(comment='tabular q-learning')
    i=0
    best_reward = 0
    while True:
        i +=1
        s,a,r,next_s = agent.sample_env()
        agent.q_update(s,a,r,next_s)

        r = 0
        for _  in range(test_episodes):
            r += agent.play_episode(env)
        r /= test_episodes
        writter.add_scalar('reward',r,i)
        if r>best_reward:
            best_reward = r
            print(r,i)
        if r >0.8:
            print('solved in %d iterations!'% i)
            break 
    
    writter.close()
