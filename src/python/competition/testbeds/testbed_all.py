__author__ = "Sergey Karakovskiy, sergey at idsia dot ch"
__date__ = "$Apr 30, 2009 1:46:32 AM$"

import sys

from experiments.episodicexperiment import EpisodicExperiment
from tasks.mariotask import MarioTask
from agents.forwardagent import ForwardAgent
from agents.forwardrandomagent import ForwardRandomAgent
from agents.ahude import Ahude
from agents.dagger import Dagger 
from agents.supervise import Supervise
from agents.sheath import Sheath
import IPython
import matplotlib.pyplot as plt
import numpy as np
from tester import Tester 
import cPickle as pickle 
from analysis import Analysis

#from pybrain.... episodic import EpisodicExperiment
#TODO: reset sends: vis, diff=, lt=, ll=, rs=, mariomode, time limit, pw,
# with creatures, without creatures HIGH.
# send creatures.
ITERATIONS = 35
IT = False

def main():
    f = open('try_3.txt','w')
    g = open('accs.txt', 'w')
    g.close()
    task = MarioTask("testbed", initMarioMode = 2)
    task.env.initMarioMode = 2
    task.env.levelDifficulty = 1

    results = [] 
    names = [] 

    
    iterations = 50
    rounds = 15
     
    agent = Supervise(IT,useKMM = False)
    exp = EpisodicExperiment(task, agent) 
    T = Tester(agent,exp)
    sl_data, sup_data, acc = T.test(rounds = rounds, iterations = iterations)

    np.save('./data/sup_data.npy', sup_data)
    np.save('./data/sl_data.npy', sl_data)
    np.save('./data/acc.npy', acc)    
    
    IPython.embed()

    analysis = Analysis()
    analysis.get_perf(sup_data, range(iterations))
    analysis.get_perf(sl_data, range(iterations))
    analysis.plot(names=['Supervisor', 'Supervised Learning'], label='Reward', filename='./results/return_plots.eps')#, ylims=[0, 1600])

    acc_a = Analysis()
    acc_a.get_perf(acc, range(iterations))
    acc_a.plot(names=['Supervised Learning Acc.'], label='Accuracy', filename='./results/acc_plots.eps')

    print "finished"



if __name__ == "__main__":
    main()
else:
    print "This is module to be run rather than imported."
