__author__ = "Sergey Karakovskiy, sergey at idsia dot ch"
__date__ = "$Apr 30, 2009 1:46:32 AM$"

import sys

from experiments.episodicexperiment import EpisodicExperiment
from tasks.mariotask import MarioTask
from agents.forwardagent import ForwardAgent
from agents.forwardrandomagent import ForwardRandomAgent
from agents.ahude import Ahude
from agents.dagger import Dagger 
from agents.noisy_supervise import NoisySupervise
from agents.sheath import Sheath
import IPython
import matplotlib.pyplot as plt
import numpy as np
from tester import Tester 
import cPickle as pickle 
from analysis import Analysis
from evaluator import Evaluator
import argparse
#from pybrain.... episodic import EpisodicExperiment
#TODO: reset sends: vis, diff=, lt=, ll=, rs=, mariomode, time limit, pw,
# with creatures, without creatures HIGH.
# send creatures.
ITERATIONS = 35
IT = False


parser = argparse.ArgumentParser()
parser.add_argument('--noisy', action='store_true')
args = vars(parser.parse_args())

def main():
    f = open('try_3.txt','w')
    g = open('accs.txt', 'w')
    g.close()
    task = MarioTask("testbed", initMarioMode = 2)
    task.env.initMarioMode = 2
    task.env.levelDifficulty = 1

    results = [] 
    names = [] 

    with open('type.txt', 'w') as f:
        f.write('dt')
    
    
    iterations = 10
    rounds = 5
    learning_samples = 2
    eval_samples = 5

  
    
    agent = Supervise(IT,useKMM = False)
    exp = EpisodicExperiment(task, agent) 
    E = Evaluator(agent,exp)
    
    if args['noisy']:
        prefix = 'dt-noisy-sup-eval'
    else:
        prefix = 'dt-sup-eval'
    
    sl_data, sup_data, acc, loss = E.eval(rounds = rounds, iterations = iterations, learning_samples=learning_samples, eval_samples=eval_samples, prefix = prefix)

    np.save('./data/' + prefix + '-sl_data.npy', sl_data)
    np.save('./data/' + prefix + '-acc.npy', acc)    
    np.save('./data/' + prefix + '-loss.npy', loss)

    analysis = Analysis()
    analysis.get_perf(sl_data, range(iterations))
    analysis.plot(names=['Supervised Learning'], label='Reward', filename='./results/' + prefix + '-return_plots.eps')#, ylims=[0, 1600])

    acc_a = Analysis()
    acc_a.get_perf(acc, range(iterations))
    acc_a.plot(names=['Supervised Learning Acc.'], label='Accuracy', filename='./results/' + prefix + '-acc_plots.eps')

    loss_a = Analysis()
    loss_a.get_perf(loss, range(iterations))
    loss_a.plot(names=['Supervised Loss'], label='Loss', filename='./results/' + prefix + '-loss_plots.eps')
    

    
       

    #agent.saveModel()
    print "finished"



if __name__ == "__main__":
    main()
else:
    print "This is module to be run rather than imported."
