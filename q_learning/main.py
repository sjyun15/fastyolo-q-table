import numpy as np
import sys
import random
import time
from environment import Environment
from agent import Agent

def print_q_table(q_table, seg, state_size):
    count = 0
    print '======================================================================================================='
    while(True):
        if count >= len(q_table):
            break
        else:
            printing = q_table[count:count+seg]
            for idx in range(seg):
                for aidx in range(state_size):
                    print "[{:.4f}, {:.4f}]   ".format(printing[idx][aidx][0], printing[idx][aidx][1]),
                print
            print '---------------------------------------------------------------------------------------------------'
        count += seg
    print '======================================================================================================='


if sys.argv[1] == 'train':
    env = Environment()
    agent = Agent()
    start = time.time()
    for e in range(200):
        state_list  = env.reset()
        if e % 10 == 0 and e >= 50:
            if agent.learning_rate > 0.001:
                agent.learning_rate /= 10
        state = state_list[0]
        done = False
        print ("training phase {}. current epsilon: {:.2f}".format(e, agent.eps_max))
        print ("current Q-table")
        print_q_table(agent.q_table, 5, 5)
        i = 0 ## counting image
        while True:
            action = agent.select_action(state)
            next_state_list, reward, done = env.step(state_list, action)
            if done:
                end = time.time()
                break
            else:
                next_state = next_state_list[0]
                agent.learn(state, action, reward, next_state)
                #print("{} th image, state:{}, action:{}, reward:{:.3f}, next_state:{}".format(i, state, action, reward, next_state))
                state = next_state
                state_list = next_state_list
                i += 1
        print ("total runtime: {:.2f}".format(end-start))
    np.save('q_table.npy', agent.q_table)

elif sys.argv[1] == 'test':
    env = Environment(train=False)
    agent = Agent(train=False)
    state_list = env.reset()
    state = state_list[0]
    while True:
        action = agent.select_action(state)
        agent.action_list.append(action)
        next_state_list, reward, done = env.step(state_list, action)
        if done:
            break
        else:
            next_state = next_state_list[0]
            state = next_state
            state_list = next_state_list
    np.save('action_list.npy', np.array(agent.action_list))
else:
    print("Wrong command!")
    print("please type \'python main.py test\' or \'python main.py train\'")
