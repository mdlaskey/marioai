"""
Designed to evaluate supervised learning policy based on training_data
gathered from collector.py.
Shoudl should happen here, and policies should be evaulated at each stage in
the learning.
"""

from experiments.episodicexperiment import EpisodicExperiment
from tasks.mariotask import MarioTask
from agents.forwardagent import ForwardAgent
from agents.forwardrandomagent import ForwardRandomAgent
from agents.ahude import Ahude
from agents.dagger import Dagger 
import IPython
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
from analysis import Analysis
import argparse



class Evaluator():

    def __init__(self,agent,exp):
        """Constructor"""
        self.exp = exp
        self.agent = agent

    def eval(self, rounds=20, iterations=35, 
        learning_samples=1, eval_samples=1, prefix='', directory=''):

        avg_data_r = np.zeros((rounds, iterations))
        avg_losses_r = np.zeros((rounds, iterations))
        acc_data = np.zeros((rounds, iterations))
        test_acc_data = np.zeros((rounds, iterations))
        avg_js_r = np.zeros((rounds, iterations))

        for r in range(rounds):
            self.agent.newModel()
            self.agent.saveModel()
            self.agent.loadModel()
            self.agent.reset()

            # self.agent.reset()

            losses = np.zeros(iterations)
            acc = np.zeros(iterations)
            test_acc = np.zeros(iterations)
            data = np.zeros(iterations)
            js = np.zeros(iterations)
            for t in range(iterations):
                print "iteration: " + str(t)

                rewards, loss, j, _ = self.exp.evalEpisodes(r, t, directory, eval_samples)
                rewards = np.mean(rewards, axis=0)
                loss = np.mean(loss, axis=0)
                j = np.mean(j, axis=0)

                data[t] = rewards[-1]
                losses[t] = loss[-1]
                acc[t] = self.agent.learner.accs
                test_acc[t] = self.agent.learner.test_accs
                js[t] = j[-1]

                self.agent.reset()


            avg_data_r[r, :] = data
            avg_losses_r[r, :] = losses
            acc_data[r, :] = acc
            test_acc_data[r, :] = test_acc
            avg_js_r[r, :] = js

            np.save('./data/' + prefix + '-eval_loss_round' + str(r) + '.npy', losses)
            np.save('./data/' + prefix + '-eval_sl_reward_round' + str(r) + '.npy', data)
            np.save('./data/' + prefix + '-eval_acc_round' + str(r) + '.npy', acc)
            np.save('./data/' + prefix + '-eval_js_round' + str(r) + '.npy', js)
            np.save('./data/' + prefix + '-eval_test_acc_round' + str(r) + '.npy', test_acc)

            a = Analysis()
            a.get_perf(np.array([data]), range(iterations))
            a.plot(names=['Supervised Learning'], label='Rewards', filename='./results/' + prefix + '-eval_return_plot' + str(r) + '.eps')

            a = Analysis()
            a.get_perf(np.array([losses]), range(iterations))
            a.plot(names=['Supervised Learning'], label='Loss', filename='./results/' + prefix + '-eval_loss_plot' + str(r) + '.eps', ylims=[0, 1])

            a = Analysis()
            a.get_perf(np.array([js]), range(iterations))
            a.plot(names=['Supervised Learning'], label='J()', filename='./results/' + prefix + 'eval_js_plot'+ str(r) + '.eps')


        self.exp.task.env.setLevelBack()
        return avg_data_r, None, acc_data, avg_losses_r, avg_js_r, test_acc_data



















