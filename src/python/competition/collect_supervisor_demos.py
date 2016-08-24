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
from agents.noisy_supervise import NoisySupervise
from agents.sheath import Sheath
import IPython
import matplotlib.pyplot as plt
import numpy as np
from tester import Tester 
import cPickle as pickle 
from analysis import Analysis
from collector import Collector
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--noisy', action='store_true')
args = vars(parser.parse_args())

IT = False
def main():
    f = open('try_3.txt','w')
    g = open('accs.txt', 'w')
    g.close()
    task = MarioTask("testbed", initMarioMode = 2)
    task.env.initMarioMode = 2
    task.env.levelDifficulty = 1
    task.env.BASE_LEVEL = 500000

    results = [] 
    names = [] 

    
    iterations = 20
    rounds = 15
    learning_samples = 33
    eval_samples = 10
    
    if args['noisy']:
        agent = NoisySupervise(IT, useKMM = False)
        dire = './training_data_noisy/'
    else:
        agent = Supervise(IT, useKMM = False)
        dire = './training_data/'

    exp = EpisodicExperiment(task, agent) 
    C = Collector(agent,exp)
    C.collect(rounds = rounds, iterations = iterations, 
        learning_samples = learning_samples, eval_samples = eval_samples,
        directory=dire)

    print "finished"



if __name__ == "__main__":
    main()
else:
    print "This is module to be run rather than imported."
