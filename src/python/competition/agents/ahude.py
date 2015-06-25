import numpy
import IPython
__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$May 1, 2009 2:46:34 AM$"

from marioagent import MarioAgent
from utils.learner import Learner
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import time 

class Ahude(MarioAgent):
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
    STATE_DIM = 27136 


    def getTraining(self):
        return self.initialTraining
    def reset(self):
        self.isEpisodeOver = False
        self.trueJumpCounter = 0;
        self.trueSpeedCounter = 0;
        
    def __init__(self,initialTraining,fl,gamma=1e-3,labelState= True):
        """Constructor"""
        self.file = fl
        self.trueJumpCounter = 0
        self.trueSpeedCounter = 0
        self.weight = numpy.zeros(1)
        self.initialTraining = initialTraining
        self.labelState = labelState
        self.actionTaken = 0
        self.action = numpy.zeros(6, int)
        self.actions = numpy.array([0])
        self.states  = csr_matrix(numpy.zeros([1,self.STATE_DIM]))
        self.actionStr = ""
        self.learner = Learner()
        self.count = 0; 
        self.mismatch = 0.0; 
        self.countLean = 0.0
        self.notComplete = True; 
        self.askedHelp = False; 
        self.off = False 
        self.prevMario = 0.0
        self.gamma = gamma 
        self.iters = 0 

        
    def loadModel(self):
        self.learner.Load(self.gamma)

    def getAction(self):
        """ Possible analysis of current observation and sending an action back
        """
#        print "M: mayJump: %s, onGround: %s, level[11,12]: %d, level[11,13]: %d, jc: %d" \
#            % (self.mayMarioJump, self.isMarioOnGround, self.levelScene[11,12], \
#            self.levelScene[11,13], self.trueJumpCounter)
#        if (self.isEpisodeOver):
#            return numpy.ones(5, int)
        #time.sleep(1)
       
        if((self.initialTraining or self.learner.askForHelp(self.obsArray.T) == -1) and not self.off):
            self.action = numpy.zeros(6,int)
            self.action[5] = 1
            #print "ASK FOR HELP",self.count

            self.record_action = True; 
            self.askedHelp = True
            #print "ASKING FOR HELP"
            if(not self.initialTraining):
                self.actionTaken = self.learner.getAction(self.obsArray.T)
        else: 
            if(not self.off):
                self.count += 1 
            actInt = self.learner.getAction(self.obsArray.T)
            self.askedHelp = False
            self.action = self.int2bin(actInt)
            self.actionTaken = actInt
            #print "ACTION TAKEN", actInt," ",self.action
            self.record_action = self.labelState

        return self.action
    def getNumHelp(self): 
        numHelps = self.count 
        self.count = 0 
        return numHelps 

    def integrateObservation(self, obs):
        """This method stores the observation inside the agent"""
        self.obs = obs
        if (len(obs) != 8):
            self.isEpisodeOver = True

        else:
            self.mayMarioJump, self.isMarioOnGround, self.marioFloats, self.enemiesFloats, self.levelScene, dummy,action,self.obsArray = obs
            if(self.off or self.iters < 6):
                self.iters += 1
                return
            elif(self.initialTraining):
                self.actions = numpy.vstack((self.actions,numpy.array([action])))
                self.obsArray = csr_matrix(self.obsArray)
                self.states = vstack((self.states,self.obsArray.T))
            elif(self.record_action and self.prevMario != self.marioFloats[0]): 
                if((self.actionTaken != action)):
                    self.prevMario = self.marioFloats[0]
                    self.actions = numpy.vstack((self.actions,numpy.array([action])))
                    self.states = numpy.vstack((self.states,self.obsArray.T))
                    if(action == 26 or action == 10):
                        weight = 2
                    else: 
                        weight = 1 
                    self.weight = numpy.vstack((self.weight,weight))


            #self.printLevelScene()
    def int2bin(self,num):
        action = numpy.zeros(6)
        actStr = numpy.binary_repr(num)
        
        for i in range(len(actStr)):
            action[i] = float(actStr[len(actStr)-1-i])
        return action 

    def updateModel(self):
        self.learner.updateModel(self.states,self.actions,self.weight)
        self.dataAdded = self.actions.shape[0]
    

    def getNumData(self): 
        return self.learner.getNumData()
        
    def newModel(self):
        self.learner.newModel(self.states,self.actions)

    def saveModel(self):
        self.learner.saveModel()

    def getMismatch(self):
        m = 0
        return m 

    def getDataAdded(self):
        return self.dataAdded
        
    def reset(self):
        self.actions = numpy.array([0])
        self.states  = numpy.zeros([1,self.STATE_DIM])
        self.weight = numpy.zeros(1)
        self.iters = 0
        
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
