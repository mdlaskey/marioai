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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--linear', action='store_true')
args = vars(parser.parse_args())

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
    
    with open('type.txt', 'w') as f:
        f.write('dt')
    
    iterations = 5
    rounds = 2
    learning_samples = 2
    eval_samples = 3


    agent = Dagger(IT,useKMM = False)

    if args['linear']:
        agent.learner.linear = True
        prefix = 'svc-dagger-change-'
    else:
        agent.learner.linear = False
        prefix = 'dt-dagger-change-'

    exp = EpisodicExperiment(task, agent) 
    T = Tester(agent,exp)
    dagger_data, _, acc, loss, js = T.test(rounds = rounds, iterations = iterations,
         learning_samples = learning_samples, eval_samples = eval_samples, prefix = prefix)

    np.save('./data/' + prefix + 'dagger_data.npy', dagger_data)
    np.save('./data/' + prefix + 'acc.npy', acc)    
    np.save('./data/' + prefix + 'loss.npy', loss)
    np.save('./data/' + prefix + 'js.npy', js)
    
    analysis = Analysis()
    analysis.get_perf(dagger_data, range(iterations))
    analysis.plot(names=['DAgger'], label='Reward', filename='./results/' + prefix + 'return_plots.eps')

    acc_a = Analysis()
    acc_a.get_perf(acc, range(iterations))
    acc_a.plot(names=['DAgger Acc.'], label='Accuracy', filename='./results/' + prefix + 'acc_plots.eps', ylims=[0,1])

    loss_a = Analysis()
    loss_a.get_perf(loss, range(iterations))
    loss_a.plot(names=['DAgger Loss'], label='Loss', filename='./results/' + prefix + 'loss_plots.eps', ylims=[0, 1])

    js_a = Analysis()
    js_a.get_perf(js, range(iterations))
    js_a.plot(names=['DAgger'], label='J()', filename='./results/' + prefix + '-js_plots.eps')

    
    print "finished"



if __name__ == "__main__":
    main()
else:
    print "This is module to be run rather than imported."
