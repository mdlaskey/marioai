__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$May 13, 2009 1:29:41 AM$"

from tcpenvironment import TCPEnvironment
from utils.dataadaptor import extractObservation

class MarioEnvironment(TCPEnvironment):
    """ An Environment class, wrapping access to the MarioServer, 
    and allowing interactions to a level. """

    # Level settings
    levelDifficulty = 0
    levelType = 0
    creaturesEnabled = True
    initMarioMode = 2
    levelSeed = 1
    levelLength = 4200
    timeLimit = 40
    fastTCP = False
  
    # Other settings
    visualization = True
    otherServerArgs = ""
    numberOfFitnessValues = 5

    def getSensors(self):
        data = TCPEnvironment.getSensors(self)
#        print "data: ", data
        return extractObservation(data)

    def reset(self):
        argstring = "-ld %d -lt %d -mm %d -ls %d -tl %d -ll %d " % (self.levelDifficulty,
                                                            self.levelType,
                                                            self.initMarioMode,
                                                            self.levelSeed,
                                                            self.timeLimit,
                                                            self.levelLength
                                                            )
        if self.creaturesEnabled:
            argstring += "-pw off "
        else:
            argstring += "-pw on "
        if self.visualization:
            argstring += "-vis on "
        else:
            argstring += "-vis off "
        if self.fastTCP:
            argstring += "-fastTCP on"

        self.client.sendData("reset -maxFPS on " + argstring + self.otherServerArgs + "\r\n")


    def changeLevel(self):
        self.levelSeed += 1

        if(self.levelSeed == 9 or self.levelSeed == 18):
            self.levelSeed +=1 
   
    def setLevelBack(self):
        self.levelSeed = 1
