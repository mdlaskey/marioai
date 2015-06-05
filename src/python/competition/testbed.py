__author__ = "Sergey Karakovskiy, sergey at idsia dot ch"
__date__ = "$Apr 30, 2009 1:46:32 AM$"

import sys

from experiments.episodicexperiment import EpisodicExperiment
from tasks.mariotask import MarioTask
from agents.forwardagent import ForwardAgent
from agents.forwardrandomagent import ForwardRandomAgent
from agents.ahude import Ahude
from agents.dagger import Dagger 
import IPython
import matplotlib.pyplot as plt
import numpy as np
from tester import Tester 
import cPickle as pickle 



#from pybrain.... episodic import EpisodicExperiment
#TODO: reset sends: vis, diff=, lt=, ll=, rs=, mariomode, time limit, pw,
# with creatures, without creatures HIGH.
# send creatures.
ITERATIONS = 35
IT = False

def main():
    f = open('try_3.txt','w')
    
    task = MarioTask("testbed", initMarioMode = 2)
    task.env.initMarioMode = 2
    task.env.levelDifficulty = 1

    results = [] 
    names = [] 
    
    #test dagger
    agent = Dagger(IT)
    exp = EpisodicExperiment(task, agent) 
    T = Tester(agent,exp)
    dagger_results = T.test()
    results.append(dagger_results)
    names.append('dagger_no_weight')


    
    # # # #test big ahude
    agent = Ahude(IT,f,gamma = 1)
    exp = EpisodicExperiment(task, agent) 
    T = Tester(agent,exp)
    ahude_big_results = T.test()
    results.append(ahude_big_results)
    names.append('ahude_1e-2')

    # pickle.dump(results,open('results.p','wb'))


    # # #test med ahude
    agent = Ahude(IT,f,gamma = 1e-1)
    exp = EpisodicExperiment(task, agent) 
    T = Tester(agent,exp)
    ahude_med_results = T.test()
    results.append(ahude_med_results)
    names.append('ahude_1e-4')
    
    # pickle.dump(results,open('results.p','wb'))

    # # #test small ahude 
    agent = Ahude(IT,f,gamma = 1e-2)
    exp = EpisodicExperiment(task, agent) 
    T = Tester(agent,exp)
    ahude_small_results = T.test() 
    results.append(ahude_small_results)
    names.append('ahude_1e-3')
    
 
    pickle.dump(results,open('results.p','wb'))

    plt.figure(1)
    for i in range(len(results)):
        plt.plot(results[i][1])
    
    
    plt.legend(names,loc='upper left')

    plt.figure(2)
    for i in range(len(results)):
        plt.plot(results[i][0])

    plt.legend(names,loc='upper left')

    plt.figure(3)
    for i in range(len(results)):
        plt.plot(results[i][4])

    plt.legend(names,loc='upper left')


    plt.show()
    
    IPython.embed()
    f.close()           
       

    #agent.saveModel()
    print "finished"



if __name__ == "__main__":
    main()
else:
    print "This is module to be run rather than imported."
