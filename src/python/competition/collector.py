"""
Designed to collect supervisor rollouts and save data to training_data directory
No evaluation takes place here. See evaluator.py
No learning should happen here either
"""

from experiments.episodicexperiment import EpisodicExperiment
from tasks.mariotask import MarioTask
from agents.forwardagent import ForwardAgent
from agents.forwardrandomagent import ForwardRandomAgent
from agents.ahude import Ahude
from agents.dagger import Dagger 
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
from analysis import Analysis
import time

class Collector():

    def __init__(self,agent,exp):
        """Constructor"""
        self.exp = exp
        self.agent = agent

    def collect(self, rounds=20, iterations=35, 
        learning_samples=1, eval_samples=1, directory=''):

        start = time.time()

        for r in range(rounds):
            self.agent.reset()
            for t in range(iterations):
                print "iteration: " + str(t)
                self.exp.collectEpisodes(1, learning_samples)

                print "Iteration Agent states: " + str(self.agent.states.shape)
                print "Iteration Agent actions: " + str(self.agent.actions.shape)
                pickle.dump(self.agent.states, open(directory + 'states_' + 'round' + str(r) + '_iter' + str(t) + '.p', 'w'))
                pickle.dump(self.agent.actions, open(directory + 'actions_' + 'round' + str(r) + '_iter' + str(t) + '.p', 'w'))

                self.agent.reset()

            # print "Round states: " + str(self.agent.states.shape)
            # print "Round actions: " + str(self.agent.actions.shape)
            # pickle.dump(self.agent.states, open('./training_data/all_round_states_' + 'round' + str(r) + '.npy', 'w'))
            # pickle.dump(self.agent.actions, open('./training_data/all_round_actions_' + 'round' + str(r) + '.npy', 'w'))

        print time.time() - start
        self.exp.task.env.setLevelBack()
        return None



















