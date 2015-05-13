import numpy
import IPython
__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$May 1, 2009 2:46:34 AM$"

from marioagent import MarioAgent
from utils.learner import Learner

class Dagger(MarioAgent):
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

    def getTraining(self):
        return self.initialTraining
    def reset(self):
        self.isEpisodeOver = False
        self.trueJumpCounter = 0;
        self.trueSpeedCounter = 0;
        
    def __init__(self,initialTraining):
        """Constructor"""
        self.trueJumpCounter = 0
        self.trueSpeedCounter = 0
        self.action = numpy.zeros(6, int)
        self.initialTraining = initialTraining
        self.actions = numpy.array([0])
        self.action[5] = 0
        self.states  = numpy.zeros(489)
        self.actionStr = ""
        self.learner = Learner()
        self.count = 0; 
        if(not self.initialTraining):
            self.learner.Load()
        
 

    def getAction(self):
        """ Possible analysis of current observation and sending an action back
        """
#        print "M: mayJump: %s, onGround: %s, level[11,12]: %d, level[11,13]: %d, jc: %d" \
#            % (self.mayMarioJump, self.isMarioOnGround, self.levelScene[11,12], \
#            self.levelScene[11,13], self.trueJumpCounter)
#        if (self.isEpisodeOver):
#            return numpy.ones(5, int)
       
        if(self.initialTraining):
            self.action[5] = 1
            self.record_action = True; 
        else: 

            actInt = self.learner.getAction(self.obsArray)
            
            self.action = self.int2bin(actInt)
            #print "ACTION TAKEN", actInt," ",self.action
            self.record_action = True
            self.actionTaken = actInt

        return self.action

    def integrateObservation(self, obs):
        """This method stores the observation inside the agent"""
        self.obs = obs
        if (len(obs) != 8):
            self.isEpisodeOver = True
        else:
            self.mayMarioJump, self.isMarioOnGround, self.marioFloats, self.enemiesFloats, self.levelScene, dummy,action,self.obsArray = obs
            if(self.record_action and (action != 18)): 
            #if(self.record_action):
                self.actions = numpy.vstack((self.actions,numpy.array([action])))
                self.states = numpy.vstack((self.states,self.obsArray))


            #self.printLevelScene()
    def int2bin(self,num):
        action = numpy.zeros(6)
        actStr = numpy.binary_repr(num)
        
        for i in range(len(actStr)):
        
            action[i] = float(actStr[len(actStr)-1-i])
        return action 

    def updateModel(self):
        self.learner.updateModel(self.states,self.actions)

    def newModel(self):
        self.learner.newModel(self.states,self.actions)

    def saveModel(self):
        self.learner.saveModel()

    def getNumData(self): 
        return self.learner.getNumData()
        
    def getName(self):
        return "Dagger"
        
    def reset(self):
        self.actions = numpy.array([0])
        self.states  = numpy.zeros(489)

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
