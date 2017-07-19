import argparse
from copy import copy
import math
import operator
import pickle
import random
import datetime
import numpy as np

import gym

import gym_2048

# TODO: Consider if can get stuck always picking the best action as an uninitialised value if don't use epsilon greedy.
class Node(object):
    """This represents a state of what to do from a particular state."""
    def __init__(self):
        # Remembers:
        # How good each action is (based on the final score)
        # Initialise to a high value to encourage exploration
        self.action_quality = [0.] * 4
        # How many times each action has been taken (to handle averaging)
        self.action_count = [0] * 4

    def get_estimate(self, action):
        """Return estimate of the quality of specified action."""
        return self.action_quality[action]

    def update_action(self, action, score, alpha):
        """Update the average for that action with the new information about the score."""
        self.action_count[action] += 1
        self.action_quality[action] += (score - self.action_quality[action]) * alpha

    def update_action_quality(self, action, new_quality):
        """Update the average for that action with the new information about the score."""
        self.action_count[action] += 1
        self.action_quality[action] = new_quality

    def visited(self):
        """Report how many times this state has been visited."""
        return sum(self.action_count)

    def __str__(self):
        def fmt(action):
            count = self.action_count[action]
            quality = self.action_quality[action]
            return "{quality:.1f} ({count})".format(quality=quality, count=count)

        indent = " " * len(fmt(3))
        s = "Visited {} times.\n".format(self.visited())
        s += "{}  {}\n".format(indent, fmt(0))
        s += "{}  ^\n".format(indent)
        s += "{} < > {}\n".format(fmt(3), fmt(1))
        s += "{}  v\n".format(indent)
        s += "{}  {}\n".format(indent, fmt(2))

        return s

    def ucb1_action(self):
        """Report which the best action would be based on upper confidence
bounds. This consists of the (scaled) quality + an upper confidence bound based
on the number of visits. Don't need random tiebreaking for this as more times
around will make a difference."""
        sum_quality = sum(self.action_quality)
        sum_counts = sum(self.action_count)
        ucb1s = list()
        for i, quality in enumerate(self.action_quality):
            count = self.action_count[i]
            if not count:
                count = 1
            scaled_quality = 0.25 # If there is no quality then default to evens
            if sum_quality:
                scaled_quality = quality / sum_quality
            ucb = math.sqrt((2. * math.log(sum_counts)) / count)
            ucb1s.append(scaled_quality + ucb)
        #print ucb1s
        return ucb1s.index(max(ucb1s))

    def greedy_action(self):
        """Report which the best action would be based on score. In event of a tie break, choose one randomly."""
        joint_best_actions = []
        max_quality = max(self.action_quality)
        for i, quality in enumerate(self.action_quality):
            if quality == max_quality:
                joint_best_actions.append(i)
        return random.choice(joint_best_actions)

class Knowledge(object):
    def __init__(self):
        self.nodes = dict()

    def add(self, state, action, score, alpha):
        if state not in self.nodes:
            self.nodes[state] = Node()
        self.nodes[state].update_action(action, score, alpha)

    def update_state_action_quality(self, state, action, new_quality):
        if state not in self.nodes:
            self.nodes[state] = Node()
        self.nodes[state].update_action_quality(action, new_quality)

    def get_estimate(self, state, action):
        try:
            return self.get_node_for_state(state).get_estimate(action)
        except AttributeError:
            return 0.

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
        for n, nod in sorted(list(self.nodes.items()), key=lambda x: x[1].visited(), reverse=True):
            npa = np.array(n)
            grid = npa.reshape((4, 4))
            print(("State:\n{}\nKnowledge:\n{}\n".format(grid, nod)))
            count += 1
            if limit and count > limit:
                break

def choose_action(env, observation, knowledge, epsilon):
    action = None
    best = False
    if (random.uniform(0, 1) > epsilon):
        if knowledge.get_node_for_state(tuple(observation)):
            #print "Picking best known action"
            state_node = knowledge.get_node_for_state(tuple(observation))
            action = state_node.ucb1_action()
            #print state_node
            best = True
        else:
            #print "Picking a random action due to lack of knowledge"
            action = env.action_space.sample()
    else:
        #print "Picking a random action due to epsilon"
        action = env.action_space.sample()
    return (action, best)


def train():
    # Eligibility trace records how much state, action pairs affect the current reward
    eligibility_trace = dict()
    #print "New episode"
    observation = env.reset()
    history = list()
    # Initialise S, A
    (action, best) = choose_action(env, observation, knowledge, epsilon)
    for t in range(1000):
        #env.render()
        #print(observation)
        #print "Action: {}".format(action)
        #last_observation = tuple(observation)
        #print "Observation: {}".format(last_observation)
        # Take action, observe R, S'
        next_observation, reward, done, info = env.step(action)
        #print "New Observation: {}, reward: {}, done: {}, info: {}".format(next_observation, reward, done, info)
        # Record what we did in a particular state

        history.append((tuple(observation), action, reward))

        if done:
            #total_moves += (t + 1)
            break

        # Choose A' from S'	using policy derived from Q
        (next_action, best) = choose_action(env, observation, knowledge, epsilon)

        # Calculate delta
        delta = reward + gamma * knowledge.get_estimate(tuple(next_observation), next_action) - knowledge.get_estimate(tuple(observation), action)
        #print(delta)
        # Increment eligibility trace for state and action
        sa = (tuple(observation), action)
        if sa in eligibility_trace:
            eligibility_trace[sa] += 1
        else:
            eligibility_trace[sa] = 1

        # Update previous data for states through history
        for h in history:
            state = h[0]
            action = h[1]
            sa = (state, action)
            estimate = knowledge.get_estimate(state, action)
            et = eligibility_trace[sa]
            knowledge.update_state_action_quality(state, action, estimate + alpha * delta * et)

            eligibility_trace[sa] = et * llambda * gamma

        observation = next_observation
        action = next_action

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.1, help="Alpha is the proportion to update your estimate by")
    parser.add_argument('--epsilon', type=float, default=0.1, help="Epsilon is probability of exploration rather than choosing best action")
    parser.add_argument('--gamma', type=float, default=0.9, help="Gamma is the decay constant for SARSA TD Error")
    parser.add_argument('--lambda', dest='llambda', type=float, default=0.9, help="Lambda is the decay value for return on action")
    parser.add_argument('--episodes', type=int, default=1000)
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
    alpha = args.alpha
    epsilon = args.epsilon
    gamma = args.gamma
    llambda = args.llambda

    previous_knowledge_size = 0
    start = datetime.datetime.now()
    print("Episode,Steps,Cumulative reward,High score,Highest tile,States known,States learnt since previous report,mse")
    high_score = 0.
    for i_episode in range(args.episodes):
        train()

        if (i_episode % args.reportfrequency) == 0:
            # Evaluate how good our current knowledge is, with a number of games
            # Make a buffer for averaging MSE over episodes
            cumulative_reward = 0.
            #cumulative_reward += reward
            if cumulative_reward > high_score:
                high_score = cumulative_reward
            mse_vals = np.zeros(100)
            cr_vals = np.zeros(100)
            ht_vals = np.zeros(100)
            cr_vals[i_episode % 100] = cumulative_reward
            ht_vals[i_episode % 100] = env.unwrapped.highest()
            t = 0
            print(("{},{},{},{},{},{},{},{}".format(i_episode, t + 1, cr_vals.mean(), high_score, ht_vals.mean(), knowledge.size(), knowledge.size() - previous_knowledge_size, mse_vals.mean(axis=None))))
            previous_knowledge_size = knowledge.size()

    end = datetime.datetime.now()
    taken = end - start
    print("{} moves took {}. {:.1f} moves per second".format(total_moves, taken, total_moves / taken.total_seconds()))
    print(knowledge)
    print("{} moves took {}. {:.1f} moves per second".format(1, taken, 1 / taken.total_seconds()))
    print(knowledge)

    if args.output:
        with open(args.output, 'w') as f:
            pickle.dump(knowledge, f)

    knowledge.dump(10)
