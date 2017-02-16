import argparse
from copy import copy
import operator
import pickle
import random
import datetime

import gym

import gym_2048

# TODO: Consider if can get stuck always picking the best action as an uninitialised value if don't use epsilon greedy.
class Node(object):
    """This represents a state of what to do from a particular state."""
    def __init__(self, state):
        # Remembers:
        # The state as a tuple
        self.state = state
        # How good each action is (based on the final score)
        # Initialise to a high value to encourage exploration
        self.action_quality = [0.] * 4
        # How many times each action has been taken (to handle averaging)
        self.action_count = [0] * 4

    def update_action(self, action, score):
        """Update the average for that action with the new information about the score."""
        self.action_count[action] += 1
        self.action_quality[action] += (score - self.action_quality[action]) / self.action_count[action]

    def visited(self):
        """Report how many times this state has been visited."""
        return sum(self.action_count)

    def __str__(self):
        s = "State {} was visitied {} times. I know (action:count:score)".format(self.state, self.visited())
        for i in range(4):
            s += " {}:{}:{:.1f}".format(i, self.action_count[i], self.action_quality[i])
        return s

    def best_action(self):
        """Report which the best action would be based on score."""
        joint_best_actions = []
        max_quality = max(self.action_quality)
        for i, quality in enumerate(self.action_quality):
            if quality == max_quality:
                joint_best_actions.append(i)
        return random.choice(joint_best_actions)
        #return self.action_quality.index(max(self.action_quality))

class Knowledge(object):
    def __init__(self):
        self.nodes = dict()

    def add(self, state, action, score):
        if state not in self.nodes:
            self.nodes[state] = Node(state)
        self.nodes[state].update_action(action, score)

    def get_node_for_state(self, state):
        try:
            return self.nodes[state]
        except KeyError:
            return None

    def size(self):
        return len(self.nodes)

    def __str__(self):
        return "I know about {} states".format(self.size())

    def dump(self, limit=0):
        """Dump knowledge, sorted by visits, limited by count."""
        count = 0
        for n, nod in sorted(self.nodes.items(), key=lambda x: x[1].visited(), reverse=True):
            print self.nodes[n]
            count += 1
            if limit and count > limit:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--reportfrequency', type=int, default=100)
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('input', nargs='?', default=None)
    args = parser.parse_args()
    env = gym.make('2048-v0')
    seen_states = dict()
    knowledge = Knowledge()
    if args.input:
        pkl_file = open(args.input, 'rb')
        knowledge = pickle.load(pkl_file)
    # Epsilon is probability of using best action
    epsilon = 0.95
    # Lambda is the decay value for return on action
    llambda = 0.9
    previous_knowledge_size = 0
    start = datetime.datetime.now()
    print("Episode,Steps,Cumulative reward,Highest tile,Best actions used,States known,States learnt since previous report")
    total_moves = 0
    for i_episode in range(args.episodes):
        #print "New episode"
        observation = env.reset()
        cumulative_reward = 0.
        history = list()
        best_actions_used = 0
        for t in range(1000):
            #env.render()
            #print(observation)
            action = None
            if (random.uniform(0, 1) < epsilon):
                if knowledge.get_node_for_state(tuple(observation)):
                    #print "Picking best known action"
                    state_node = knowledge.get_node_for_state(tuple(observation))
                    action = state_node.best_action()
                    #print state_node
                    best_actions_used += 1
                else:
                    #print "Picking a random action due to lack of knowledge"
                    action = env.action_space.sample()
            else:
                #print "Picking a random action due to epsilon"
                action = env.action_space.sample()
            #print "Action: {}".format(action)
            # Record what we did in a particular state
            last_observation = tuple(observation)
            #print "Observation: {}".format(last_observation)
            observation, reward, done, info = env.step(action)
            #print "New Observation: {}, reward: {}, done: {}, info: {}".format(observation, reward, done, info)
            history.append((last_observation, action, reward))
            cumulative_reward += reward
            if done:
                #print("Episode finished after {} timesteps. Cumulative reward {}".format(t+1, cumulative_reward))
                if (i_episode % args.reportfrequency) == 0:
                    print("{},{},{},{},{},{},{}".format(i_episode, t + 1, cumulative_reward, env.highest(), best_actions_used, knowledge.size(), knowledge.size() - previous_knowledge_size))
                    previous_knowledge_size = knowledge.size()
                total_moves += (t + 1)
                break
        # Go through history creating or updating knowledge
        # Calculate TD lambda estimates, actual value + lambda of next one plus lambda squared of the next one etc.
        td_lambda_estimate = list()
        # TD0 lambda estimate
        #print len(history)
        for h in history:
            td_lambda_estimate.append(h[2])
        # Add TD1 (e.g. Lambda * reward goes to n-1 action
        for td_factor in range(1, min(10, len(history))):
            #print "Lambda estimate: {}".format(td_lambda_estimate)
            lambda_multiplier = llambda**td_factor
            #print "Lambda multiplier: {}".format(lambda_multiplier)
            for i in range(len(history) - td_factor):
                td_lambda_estimate[i] += history[i+td_factor][2] * lambda_multiplier

        # Update knowledge with estimates
        for idx, h in enumerate(history):
            knowledge.add(h[0], h[1], td_lambda_estimate[idx])
            #knowledge.add(h[0], h[1], cumulative_reward)

    end = datetime.datetime.now()
    taken = end - start
    print "{} moves took {}. {:.1f} moves per second".format(total_moves, taken, total_moves / taken.total_seconds())
    print knowledge

    if args.output:
        with open(args.output, 'w') as f:
            pickle.dump(knowledge, f)

    knowledge.dump(10)
