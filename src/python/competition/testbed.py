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
    
    IPython.embed()

    task = MarioTask("testbed", initMarioMode = 2)
    task.env.initMarioMode = 2
    task.env.levelDifficulty = 1
    
    
    #test dagger
    agent = Dagger(IT)
    exp = EpisodicExperiment(task, agent) 
    T = Tester(agent,exp)
    dagger_results = T.test()
    
    #test big ahude
    agent = Ahude(IT,f,gamma = 1e-2)
    exp = EpisodicExperiment(task, agent) 
    T = Tester(agent,exp)
    ahude_big_results = T.test()
    
    #test med ahude
    agent = Ahude(IT,f,gamma = 5e-3)
    exp = EpisodicExperiment(task, agent) 
    T = Tester(agent,exp)
    ahude_med_results = T.test()
    
    #test small ahude 
    agent = Ahude(IT,f,gamma = 1e-3)
    exp = EpisodicExperiment(task, agent) 
    T = Tester(agent,exp)
    ahude_small_results = T.test() 
    
    results = [dagger_results, ahude_big_results, ahude_med_results,ahude_small_results]
    
        
    pickle.dump(results,open('results.p','wb'))
    
    plt.plot(dagger_results[1])
    plt.plot(ahude_big_results[1])
    plt.plot(ahude_med_results[1])
    plt.plot(ahude_small_results[1])
    
    plt.legend(['dagger','ahude_big','ahude_med','ahude_small'],loc='upper left')
    plt.show()
    
    IPython.embed()
    f.close()           
       

    #agent.saveModel()
    print "finished"



if __name__ == "__main__":
    main()
else:
    print "This is module to be run rather than imported."
