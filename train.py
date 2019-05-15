#train
import numpy as np
from env import chessboard
from agent import DQN

def run_game():
    step = 0
    for episode in range(10000):
        observation = GAME.init()    
        while True:
            observation = GAME.maintain()
            #刷新

            action = RL.choose_action(observation)
            #根据估值函数选择action

            observation_, reward, choosed_action, done = GAME.step(action[0]) 
            #action应为一个四元组


            RL.store_transition(observation, choosed_action, reward, observation_) 
            #记忆储存， 储存这个 行动——局面——回报 组

            if (step > 150) and (step % 5 == 0):  
                #控制学习起始时间和频率：先记忆再学习
                RL.learn()
            
            if (done == False): 
                break
            step += 1

        #end of game
        #print("game over")
        #print(observation, '\n')
        result.append(np.max(observation))

if __name__ == "__main__":
    GAME = chessboard("GAME")
    result = []

    RL = DQN(GAME.n_actions, GAME.n_features,
                      learning_rate=0.01,
                      reward_decay=0.93,
                      e_greedy=0.93,
                      replace_target_iter=100,
                      memory_size=20000,
                      # output_graph=True
                      )
   
    run_game()
    RL.show_cost()
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(result)), result)
    plt.ylabel('max')
    plt.xlabel('games')
    plt.show() 


        