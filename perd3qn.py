# coding=utf-8

import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
import pandas as pd

# Hyper Parameters for DQN
GAMMA = 0.99 # discount factor for target Q
INITIAL_EPSILON = 0.9 # starting value of epsilon
FINAL_EPSILON = 0.00002 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch
#REPLACE_TARGET_FREQ = 10 # frequency to update target Q network
TAU = 0.05

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.3  # [0~1] convert the importance of TD error to priority
    beta = 0.1  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001    #最开始是0.001
    abs_err_upper = 3.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
 
class DQN():                                                                        #按照模板根据自己的方法写的class DQN
  # DQN Agent
  def __init__(self, env):
    # init experience replay
    self.replay_total = 0
    # init some parameters
    self.time_step = 0
    self.epsilon = INITIAL_EPSILON
    self.state_dim = env.observation_space.shape[0]
    self.action_dim = env.action_space.n
    self.memory = Memory(capacity=REPLAY_SIZE)

    self.create_Q_network()
    self.create_training_method()

    # Init session
    self.session = tf.InteractiveSession()
    self.session.run(tf.global_variables_initializer())

  def create_Q_network(self):
    # input layer
    self.state_input = tf.placeholder("float", [None, self.state_dim])
    self.ISWeights = tf.placeholder(tf.float32, [None, 1])
    # network weights
    with tf.variable_scope('current_net'):
        W1 = self.weight_variable([self.state_dim,50])
        b1 = self.bias_variable([50])

        # hidden layer 1
        h_layer_1 = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)

        # hidden layer  for state value
        with tf.variable_scope('Value'):
          W21= self.weight_variable([50,1])
          b21 = self.bias_variable([1])
          self.V = tf.matmul(h_layer_1, W21) + b21

        # hidden layer  for action value
        with tf.variable_scope('Advantage'):
          W22 = self.weight_variable([50,self.action_dim])
          b22 = self.bias_variable([self.action_dim])
          self.A = tf.matmul(h_layer_1, W22) + b22

          # Q Value layer
          self.Q_value = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))

    with tf.variable_scope('target_net'):
        W1t = self.weight_variable([self.state_dim,50])
        b1t = self.bias_variable([50])

        # hidden layer 1
        h_layer_1t = tf.nn.relu(tf.matmul(self.state_input,W1t) + b1t)

        # hidden layer  for state value
        with tf.variable_scope('Value'):
          W2v = self.weight_variable([50,1])
          b2v = self.bias_variable([1])
          self.VT = tf.matmul(h_layer_1t, W2v) + b2v

        # hidden layer  for action value
        with tf.variable_scope('Advantage'):
          W2a = self.weight_variable([50,self.action_dim])
          b2a = self.bias_variable([self.action_dim])
          self.AT = tf.matmul(h_layer_1t, W2a) + b2a

          # Q Value layer
          self.target_Q_value = self.VT + (self.AT - tf.reduce_mean(self.AT, axis=1, keep_dims=True))

    t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')       
    e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')

    self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)  # 将原来的硬更新改写成自己的软更新方式   TAU设置为0.05  TAU也是调参出来发现0.05 效果很好             
    for t, e in zip(t_params , e_params )]

  def create_training_method(self):
    self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
    self.y_input = tf.placeholder("float",[None])
    Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
    self.cost = tf.reduce_mean(self.ISWeights *(tf.square(self.y_input - Q_action)))
    self.abs_errors =tf.abs(self.y_input - Q_action)
    self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

  def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, r, s_, done))
        self.memory.store(transition)    # have high priority for newly arrived transition

  def perceive(self,state,action,reward,next_state,done):
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1
    #print(state,one_hot_action,reward,next_state,done)
    self.store_transition(state,one_hot_action,reward,next_state,done)
    self.replay_total += 1
    if self.replay_total > BATCH_SIZE:
        self.train_Q_network()

  def train_Q_network(self):
    self.time_step += 1
    # Step 1: obtain random minibatch from replay memory
    tree_idx, minibatch, ISWeights = self.memory.sample(BATCH_SIZE)
    state_batch = minibatch[:,0:2]
    action_batch =  minibatch[:,2:5]
    reward_batch = [data[5] for data in minibatch]
    next_state_batch = minibatch[:,5:7]
    # Step 2: calculate y
    y_batch = []
    current_Q_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
    max_action_next = np.argmax(current_Q_batch, axis=1)
    target_Q_batch = self.target_Q_value.eval(feed_dict={self.state_input: next_state_batch})

    for i in range(0,BATCH_SIZE):
      done = minibatch[i][7]
      if done:
        y_batch.append(reward_batch[i])
      else :
        target_Q_value = target_Q_batch[i, max_action_next[i]]
        y_batch.append(reward_batch[i] + GAMMA * target_Q_value)

    self.optimizer.run(feed_dict={
      self.y_input:y_batch,
      self.action_input:action_batch,
      self.state_input:state_batch,
      self.ISWeights: ISWeights
      })
    _, abs_errors, _ = self.session.run([self.optimizer, self.abs_errors, self.cost], feed_dict={
                          self.y_input: y_batch,
                          self.action_input: action_batch,
                          self.state_input: state_batch,
                          self.ISWeights: ISWeights
                          })
    self.memory.batch_update(tree_idx, abs_errors)  # update priority

  def egreedy_action(self,state,n):             #def egreedy_action(self,state,n):  按照动态贪婪自己写的代码 具体的调参数也是在里面进行的
    Q_value = self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0]
    self.epsilon = (pow((n+25),3)/120)*(pow(np.e,(-np.sqrt(n+25))))  
    if random.random() <= self.epsilon:   
        return random.randint(0,self.action_dim - 1)
    else:
        return np.argmax(Q_value)

  def action(self,state):
    return np.argmax(self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0])

  def update_target_q_network(self):                    #修改成软更新方式
    self.session.run(self.soft_replace)

  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)

  def bias_variable(self,shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)
# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'MountainCar-v0'
EPISODE = 400 # Episode limitation
STEP = 2000 # Step limitation in an episode

number_ddpqn = []
step_lis = []

def main():                                      #def main函数按照修订的方法自己写的  查资料 修改了给出的条件   添加合适的条件  这样子mountaincar才可以收敛
  # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    agent = DQN(env)
    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        #print('state = ',state[0])
        # Train
        for step in range(STEP):
            #print('step = ',step)
            action = agent.egreedy_action(state,episode) # e-greedy action for train
            next_state,reward,done,_ = env.step(action)
            position,velocity = next_state
            reward = abs(position-(-0.5))
            agent.perceive(state,action,reward,next_state,done)
            agent.update_target_q_network()
            state = next_state
            if (done==True) | (step == STEP-1):
                step_lis.append(step+1)
                #print('step = ',step)
                print ('episode: ',episode,'step : ',step+1)
            if done:
                break
        number_ddpqn.append(episode)
    print('step_lis = ',step_lis)
    print(number_ddpqn)

if __name__ == '__main__':
   main()

number_ddpqnpd = pd.DataFrame(number_ddpqn)
step_lis_ddpqnpd =pd.DataFrame(step_lis)

number_ddpqnpd.to_csv('dataddpqn1.csv')
step_lis_ddpqnpd.to_csv('dataddpqn2.csv')












