import numpy
__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$May 1, 2009 2:46:34 AM$"

from marioagent import MarioAgent
from utils.learner import Learner
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import time
import random
import numpy as np
class NoisySupervise(MarioAgent):
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
    initialTraining = False 
    trueJumpCounter = 0;
    trueSpeedCounter = 0;
    record_action = False; 
    STATE_DIM = 27136 

    def getTraining(self):
        return self.initialTraining
    # def reset(self):
    #     self.isEpisodeOver = False
    #     self.trueJumpCounter = 0;
    #     self.trueSpeedCounter = 0;
        

    def __init__(self,initialTraining,useKMM = False):
        """Constructor"""
        self.trueJumpCounter = 0
        self.trueSpeedCounter = 0
        self.action = numpy.zeros(6, int)
        self.weight = numpy.zeros(1)
        self.initialTraining = initialTraining
        self.actionTaken = 0
        self.actions = numpy.array([0])
        self.action[5] = 0
        self.states  = csr_matrix(numpy.zeros([1,self.STATE_DIM]))
        self.actionStr = ""
        self.learner = Learner()
        self.learner.useKMM = useKMM
        self.count = 0; 
        self.human_input = 0; 
        self.prevMario = 0.0
        self._name = 'supervise'
        self.isLearning = False
        
    def loadModel(self):
        self.learner.Load()

    def getAction(self):
        """ Possible analysis of current observation and sending an action back
        """
#        print "M: mayJump: %s, onGround: %s, level[11,12]: %d, level[11,13]: %d, jc: %d" \
#            % (self.mayMarioJump, self.isMarioOnGround, self.levelScene[11,12], \
#            self.levelScene[11,13], self.trueJumpCounter)
#        if (self.isEpisodeOver):
#            return numpy.ones(5, int)            
        if self.isLearning:
            r = random.random()
            if r > .3:
                self.action = numpy.zeros(6, int)
                self.action[5] = 1
                self.record = True
                # actInt = -1
            else:
                actInt = random.randint(0, 2 ** 5 - 1)
                self.action = self.int2bin(actInt)
                self.record_action = True
                # self.actionTaken = actInt
        elif self.count <= 6:
            self.action = numpy.zeros(6, int)
            self.action[5] = 1
            self.record = True
            # actInt = -1
        # elif self.isLearning:
        #     r = random.random()
        #     if r > .2:
        #         actInt = self.should_take_action
        #         self.action = self.int2bin(actInt)
        #         self.record_action = True
        #         self.actionTaken = actInt
        #     else:
        #         actInt = random.randint(0, 2 ** 5 - 1)
        #         self.action = self.int2bin(actInt)
        #         self.record_action = True
        #         self.actionTaken = actInt
        else:
            actInt = self.learner.getAction(self.obsArray.T)
            self.action = self.int2bin(actInt)
            self.record_action = True
            self.actionTaken = actInt
        return self.action
        
        """if self.count <= 6:
            self.action = numpy.zeros(6, int)
            self.action[5] = 1
            self.record=True
        elif self.isLearning or self.count <= 6:
            #self.action = numpy.zeros(6, int)
            #self.action[5] = 1
            self.action = self.int2bin(self.should_take_action)
            self.record_action = True;
        else:
            actInt = self.learner.getAction(self.obsArray.T)
            self.record_actions.append(actInt[0])
            self.action = self.int2bin(10)
            self.action = self.int2bin(actInt)
            self.record_action = True
            self.actionTaken = actInt
            print "Should taken: " + str(self.should_take_action)
            print "Taken: " + str(self.actionTaken[0])
        return self.action"""


    def integrateObservation(self, obs):
        """This method stores the observation inside the agent"""
        start_time = time.time()
        self.obs = obs
        if (len(obs) != 8):
            self.isEpisodeOver = True
        else:
            self.mayMarioJump, self.isMarioOnGround, self.marioFloats, self.enemiesFloats, self.levelScene, dummy,action,self.obsArray = obs
            self.obsArray = csr_matrix(self.obsArray) # delete this later (if it doesn't work)
            self.should_take_action = action

            if(self.count > 5):
                if self.isLearning:
                    self.actions = numpy.vstack((self.actions,numpy.array([action])))
                    # self.obsArray = csr_matrix(self.obsArray)
                    # self.states = vstack((self.states,self.obsArray.T))
                    self.states = vstack((self.states,self.prev_obs.T))
                    self.human_input += 1
                    self.should_take_action = action
                else:
                    if self.count > 6 and action != self.actionTaken:
                            self.mistakes += 1


                    self.should_take_action = action
                # if(self.initialTraining):
                #     self.actions = numpy.vstack((self.actions,numpy.array([action])))
                #     self.obsArray = csr_matrix(self.obsArray)
                #     self.states = vstack((self.states,self.obsArray.T))
                # else:#elif(self.record_action and self.prevMario != self.marioFloats[0]): 
                #     obsArray_csr = csr_matrix(self.obsArray)
                #     self.kmm_state = vstack((self.kmm_state,obsArray_csr.T))
                #     if True:#if((self.actionTaken != action)):
                #         self.prevMario = self.marioFloats[0]
                #         self.actions = numpy.vstack((self.actions,numpy.array([action])))
                #         self.states = numpy.vstack((self.states,self.obsArray.T))
            self.prev_obs = self.obsArray     
            self.count += 1
            # self.prev_action = action
        # self._write_out(time.time() - start_time)

            #self.printLevelScene()
    def int2bin(self,num):
        action = numpy.zeros(6)
        actStr = numpy.binary_repr(num)
        
        for i in range(len(actStr)):
            action[i] = float(actStr[len(actStr)-1-i])
        return action 
    
    def _write_out(self, time):
        filename = 'supervise_times.txt'
        with open(filename, 'a') as f:
            f.write("time: " + str(time) + "\n")
        return

    def updateModel(self):
        # print self.states
        # print self.actions 

        self.learner.updateModel(self.states,self.actions,None)#self.kmm_state)
        self.dataAdded = self.actions.shape[0]

    def getDataAdded(self):
        return self.dataAdded

    def getNumHumanInput(self):
        return self.human_input

    def newModel(self):
        self.learner.newModel(self.states,self.actions)

    def saveModel(self):
        self.learner.saveModel()

    def getNumData(self): 
        return self.learner.getNumData()
        
    def getName(self):
        return "Supervise"
        
    def reset(self):
        self.actions = numpy.array([0])
        self.states  = numpy.zeros([1,self.STATE_DIM])
        self.kmm_state = numpy.zeros([1,self.STATE_DIM])
        self.weight = numpy.zeros(1)
        
        self.count = 0
        self.mistakes = 0

    def reset_task(self):
        self.count = 0
        self.mistakes = 0

    def get_loss(self):
        try:
            return self.mistakes / float(self.count)
        except ZeroDivisionError:
            return -1.0

    def get_j(self):
        return self.mistakes
            
    def printLevelScene(self):
        ret = ""
        for x in range(22):
            tmpData = ""
            for y in range(22):
                tmpData += self.mapElToStr(self.levelScene[x][y]);
            ret += "\n%s" % tmpData;
        print ret

    def mapElToStr(self, el):
        """maps element of levelScene to str representation"""
        s = "";
        if  (el == 0):
            s = "##"
        s += "#MM#" if (el == 95) else str(el)
        while (len(s) < 4):
            s += "#";
        return s + " "

    def printObs(self):
        """for debug"""
        print repr(self.observation)
