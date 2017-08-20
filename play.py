from __future__ import print_function

import argparse
import math
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

def get_tile_total_for_state(state):
    """Report the total value of all tiles on the board in this state. State is a tuple."""
    return sum(state)

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

    def report(self):
        """Report how many states I know for each tile total"""
        report = ''
        freq = dict()
        max_tile_total = 0
        for state, node in self.nodes.items():
            tile_total = get_tile_total_for_state(state)
            max_tile_total = max(max_tile_total, tile_total)
            try:
                freq[tile_total] += 1
            except:
                freq[tile_total] = 1
        #print("Tile total,States known")
        for i in range(4, max_tile_total, 2):
            try:
                report += "{},".format(freq[i])
            except KeyError:
                report += "{},".format(0)
        return report

    def dump(self, limit=0, size=4):
        """Dump knowledge, sorted by visits, limited by count."""
        count = 0
        for n, nod in sorted(list(self.nodes.items()), key=lambda x: x[1].visited(), reverse=True):
            npa = np.array(n)
            grid = npa.reshape((size, size))

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
            action = random.randint(0, 3)
    else:
        #print "Picking a random action due to epsilon"
        action = random.randint(0, 3)
    return (action, best)


def train(k):
    # Eligibility trace records how much state, action pairs affect the current reward
    eligibility_trace = dict()
    #print "New episode"
    # Initialise seed for environment
    env.seed()
    # Initialise seed for agent
    random.seed()
    observation = env.reset()
    history = list()
    # Initialise S, A
    (action, best) = choose_action(env, observation, k, epsilon)
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
        (next_action, best) = choose_action(env, observation, k, epsilon)

        # Calculate delta
        delta = reward + gamma * k.get_estimate(tuple(next_observation), next_action) - k.get_estimate(tuple(observation), action)
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
            estimate = k.get_estimate(state, action)
            et = eligibility_trace[sa]
            k.update_state_action_quality(state, action, estimate + alpha * delta * et)

            eligibility_trace[sa] = et * llambda * gamma

        observation = next_observation
        action = next_action
        #print("")

def evaluate(k):
    #print("Evaluating")
    # Don't explore when evaluating, just use best actions
    epsilon = 0

    evaluation_episodes = 10

    cr_vals = np.zeros(10)
    crs = list()

    for eval_episode in range(evaluation_episodes):
        cumulative_reward = 0.
        # Fix seed for environment so we are always evaluating the same game
        env.seed(123 + eval_episode)
        # Fix seed for agent so we are always making the same random choices when required
        random.seed(456 + eval_episode)

        observation = env.reset()
        #env.render()

        (action, best) = choose_action(env, observation, k, epsilon)
        for t in range(1000):
            next_observation, reward, done, info = env.step(action)
            cumulative_reward += reward
            if done:
                break
            # Choose A' from S'	using policy derived from Q
            (next_action, best) = choose_action(env, observation, k, epsilon)
            observation = next_observation
            action = next_action
        #print("Score: {}".format(cumulative_reward))
        crs.append(str(cumulative_reward))
        cr_vals[eval_episode] = cumulative_reward
    crs.append(str(cr_vals.mean()))
    #print("Average score: {}".format(cr_vals.mean()))
    return crs

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
    high_score = 0.
    #print("Episodes,A,B,C,D,E,F,G,H,I,J,Average")
    with open('tile_total_frequency.csv', 'w') as ttf:
        ttf.write("Episode,")
        for i in range(4, 600, 2):
            ttf.write("{},".format(i))
        ttf.write("\n")
        for i_episode in range(args.episodes):

            if (i_episode % args.reportfrequency) == 0:
                ttf.write("{},{}\n".format(i_episode, knowledge.report()))

                # Evaluate how good our current knowledge is, with a number of games
                #s = evaluate(knowledge)
                #print(','.join([str(i_episode)] + s))

            train(knowledge)

        if (i_episode % args.reportfrequency) == 0:
            # Evaluate how good our current knowledge is, with a number of games
            #s = evaluate(knowledge)
            #print(','.join([str(i_episode)] + s))

    # Close the environment
    env.close()

    end = datetime.datetime.now()
    taken = end - start
    print("{} episodes took {}. {:.1f} episodes per second".format(args.episodes, taken, args.episodes / taken.total_seconds()))
    print(knowledge)

    if args.output:
        with open(args.output, 'w') as f:
            pickle.dump(knowledge, f)

    knowledge.dump(10)
