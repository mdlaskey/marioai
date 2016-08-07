
__author__ = "Michael Laskey"

from experiments.episodicexperiment import EpisodicExperiment
from tasks.mariotask import MarioTask
from agents.forwardagent import ForwardAgent
from agents.forwardrandomagent import ForwardRandomAgent
from agents.ahude import Ahude
from agents.dagger import Dagger 
import IPython
import matplotlib.pyplot as plt
import numpy as np


class Tester:
    """ In fact the Python twin of the
        corresponding Java ForwardAgent.
    """
    action = None
    actionStr = None
    KEY_JUMP = 3
    KEY_SPEED = 4
    levelScene = None
    mayMarioJump = None
    isMarioOnGround = None
    marioFloats = None
    enemiesFloats = None
    isEpisodeOver = False
      
    trueJumpCounter = 0;
    trueSpeedCounter = 0;
    record_action = False; 

  
    def __init__(self,agent,exp):
        """Constructor"""
        self.exp = exp
        self.agent = agent
        
 
    def test(self,rounds=20,iterations=35):
        num_ask_help = np.zeros(iterations) 
        num_mismatch = np.zeros(iterations)
        avg_data = np.zeros(iterations) 
        avg_distance = np.zeros(iterations)
        avg_precision = np.zeros(iterations)
        avg_human_input = np.zeros(iterations)
        
        


        for r in range(rounds):

            self.agent.initialTraining = True
            self.exp.doEpisodes(1)
            if self.agent._name != "supervise":
                self.agent.newModel()
                self.agent.saveModel()
            self.agent.initialTraining = False
            if self.agent._name != "supervise":
                self.agent.loadModel()
                self.agent.reset()


            distances = np.zeros(iterations)
            mis_match = np.zeros(iterations)
            data = np.zeros(iterations)
            num_help = np.zeros(iterations)
            precision = np.zeros(iterations)
            human_input = np.zeros(iterations)
            for t in range(iterations):
                
                rewards = self.exp.doEpisodes(1)
                
                size = len(rewards[0])
                data[t] = rewards[0][size-1]
                self.agent.updateModel()
               

                if(self.agent._getName() == 'Ahude'):
                    num_help[t] = self.agent.getNumHelp()
                    mis_match[t] = self.agent.getMismatch()
                    self.agent.off = True
                    rewards = self.exp.doEpisodes(1)
                    self.agent.off = False

                size = len(rewards[0])
               
                    
                distances[t] = rewards[0][size-1]
               
                precision[t] = self.agent.learner.getPrecision()
                human_input[t] = self.agent.getNumHumanInput()
                self.agent.reset()
             
            if(self.agent._getName() == 'Ahude'):
                num_ask_help = num_ask_help + num_help
                num_mismatch = num_mismatch + mis_match

            avg_data = data+avg_data
            avg_distance = distances+avg_distance 
            avg_precision = precision+avg_precision
            avg_human_input = avg_human_input + human_input
          
            self.exp.task.env.changeLevel()
        
        num_ask_help = num_ask_help/rounds
        num_mismatch = num_mismatch/rounds
        avg_data = avg_data/rounds
        avg_distance = avg_distance/rounds 
        avg_precision = avg_precision/rounds 
        avg_human_input = avg_human_input/rounds
        self.exp.task.env.setLevelBack()
        
        print avg_distance

        return avg_data,avg_distance,num_mismatch,num_ask_help,avg_precision,avg_human_input

             
             
        
  
