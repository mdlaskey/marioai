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


#from pybrain.... episodic import EpisodicExperiment
#TODO: reset sends: vis, diff=, lt=, ll=, rs=, mariomode, time limit, pw,
# with creatures, without creatures HIGH.
# send creatures.
ITERATIONS = 30
IT = False

def main():
    agent = Ahude(IT)
    distances = np.zeros([1])
    data = np.zeros([1])
    num_help = np.zeros([1])
    mis_match = np.zeros([1])


    #task = MarioTask(agent.name, initMarioMode = 2)
    #exp = EpisodicExperiment(task, agent)
    print 'Task Ready'
    #task.env.initMarioMode = 2
    #task.env.levelDifficulty = 1
    task = MarioTask(agent.name, initMarioMode = 2)
    exp = EpisodicExperiment(task, agent)
    task.env.initMarioMode = 2
    task.env.levelDifficulty = 1
    if(agent.initialTraining):
        exp.doEpisodes(5)
        agent.newModel()
    else:
        for i in range(ITERATIONS):
            #if( i == 2):
                #agent.initialTraining = True; 
             #   IPython.embed()
            print agent.initialTraining
          
            rewards = exp.doEpisodes(1)
            
           
           
            agent.updateModel()

            if(agent._getName() == 'Ahude'):
                num_help = np.vstack((num_help,np.array(agent.getNumHelp())))
                mis_match = np.vstack((mis_match,np.array(agent.getMismatch())))
                #agent.off = True
                #rewards = exp.doEpisodes(1)
                #agent.off = False

            size = len(rewards[0])
                
            distances = np.vstack((distances,np.array(rewards[0][size-1])))
            data = np.vstack((data,np.array(agent.getNumData())))
         
            

          
            #agent.notComplete = False
            print "TRACK COMPLETE"
            agent.reset()
        IPython.embed()
        plt.figure(2)
        plt.plot(data,distances)

        if(agent._getName() == 'Ahude'):    
            plt.figure(1)
            plt.plot(data,num_help)
            plt.figure(3)
            plt.plot(mis_match)
        plt.show()
    #agent.saveModel()
    print "finished"

#    clo = CmdLineOptions(sys.argv)
#    task = MarioTask(MarioEnvironment(clo.getHost(), clo.getPort(), clo.getAgent().name))
#    exp = EpisodicExperiment(clo.getAgent(), task)
#    exp.doEpisodes(3)

if __name__ == "__main__":
    main()
else:
    print "This is module to be run rather than imported."
