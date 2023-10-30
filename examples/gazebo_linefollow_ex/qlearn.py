import random
import pickle


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions
        self.highest_reward = float('-inf')  # Initialize with negative infinity

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        with open(filename + ".pickle", 'rb') as file:
            self.q = pickle.load(file)


        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        with open(filename + ".pickle", 'wb') as file:
             pickle.dump(self.q, file)


        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q_values)
            count = q_values.count(maxQ)

            if count > 1:
                best = [i for i in range(len(self.actions)) if q_values[i] == maxQ]
                i = random.choice(best)
            else:
                i = q_values.index(maxQ)
            action = self.actions[i]

        if return_q:
                return action, self.getQ(state, action)
        return action



    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        integer_sum = sum([s for s in state1 if isinstance(s, int)])
        if integer_sum == len(state1):
            reward = 0
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        oldv = self.q.get((state1, action1), None)
        if oldv is None:
            self.q[(state1, action1)] = reward
        else:
            self.q[(state1, action1)] = oldv + self.alpha * (reward + self.gamma * maxqnew - oldv)
        
        if reward > self.highest_reward:
            self.saveQ("best_policy")
            self.highest_reward = reward
            print("New highest reward achieved. Policy saved.")
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

    def loadBestPolicy(self):
        self.loadQ("best_policy")

