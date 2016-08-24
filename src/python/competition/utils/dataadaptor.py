__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$Apr 30, 2009 1:53:54 PM$"

import numpy as np
import IPython
    
from bitsTest import powsof2

labels = [] 
pastLevels = []
pastActions = [] 

def show(el):
#    powsof2 = (1, 2, 4, 8, 16, 32, 64, 128)
    print "block (", el, ") :",
    for  i in range(16):
        print ((int(el) & powsof2[i])),
    print

def binary_labels(x):

    labels = dict( {
        25:0,
        -10:1,
        6:2,
        -11:3,
        21:4,
        20:5,
        12:6,
        2:7,
        13:8,
        4:9,
        3:10,
        15:11,
        7:12,
        16:13
    })

    if (x in labels):
        return True 
    else: 
        return 1


def asmbleObsv(pLevels,pActions,mario_state):
    obs = np.array([])

    for level in pLevels:
        if(obs.shape[0] == 0):
            obs = level 
        else: 
            obs = np.vstack([obs,level])

    for action in pActions:
        obs = np.vstack([obs,action])

    obs = np.vstack([obs,mario_state])

    return obs



def decode(estate):
    """
    decodes the encoded state estate, which is a string of 61 chars
    """
#    powsof2 = (1, 2, 4, 8, 16, 32, 64, 128)
    dstate = np.empty(shape = (22, 22), dtype = np.int)
    for i in range(22):
        for j in range(22):
            dstate[i, j] = 2
    row = 0
    col = 0
    totalBitsDecoded = 0
    reqSize = 31
    assert len(estate) == reqSize, "Error in data size given %d! Required: %d \n data: %s " % (len(estate), reqSize, estate)
    check_sum = 0
    for i in range(len(estate)):
        cur_char = estate[i]
        if (ord(cur_char) != 0):
#            show(ord(cur_char))
            check_sum += ord(cur_char)
        for j in range(16):
            totalBitsDecoded += 1
            if (col > 21):
                row += 1
                col = 0
            if ((int(powsof2[j]) & int(ord(cur_char))) != 0):
#                show((int(ord(cur_char))))
                dstate[row, col] = 1
            else:
                dstate[row, col] = 0
            col += 1
            if (totalBitsDecoded == 484):
                break
    print "totalBitsDecoded = ", totalBitsDecoded
    return dstate, check_sum;


def extractObservation(data):
    """
     parse the array of strings and return array 22 by 22 of doubles
    """

    obsLength = 489
    obs_array = np.zeros(obsLength)
   
    levelScene = np.empty(shape = (22, 22), dtype = np.int)
    enemiesFloats = []

    dummy = 0
    if(data[0] == 'E'): #Encoded observation, fastTCP mode, have to be decoded
#        assert len(data) == eobsLength
        mayMarioJump = (data[1] == '1')
        isMarioOnGround = (data[2] == '1')
        levelScene, check_sum_got = decode(data[3:34])
        check_sum_recv = int(data[34:])
#        assert check_sum_got == check_sum_recv, "Error check_sum! got %d != etalon %d" % (check_sum_got, check_sum_recv)
        if check_sum_got != check_sum_recv:
            print "Error check_sum! got %d != recv %d" % (check_sum_got, check_sum_recv)
#        for i in range(22):
#            for j in range(22):
#               if levelScene[i, j] != 0:
#                   print '1',
#               else:
#                   print ' ',
#            print 
        
#        enemies = decode(data[0][64:])
        return (mayMarioJump, isMarioOnGround, levelScene)
    data = data.split(' ')
    if (data[0] == 'FIT'):
        status = int(data[1])
        distance = float(data[2])
        timeLeft = int(data[3])
        marioMode = int(data[4])
        coins = int(data[5])
        
#        print "S: %s, F: %s " % (data[1], data[2])
        #print "status %s, dist %s, timeleft %s, mmode %s, coins %s" % (status, distance, timeLeft, marioMode, coins) 
        return status, distance, timeLeft, marioMode, coins
    elif(data[0] == 'O'):
        mario_state = np.array([])
        for i in range(1,obsLength): 
            
            if(data[i] == "true"):
                obs_array[i-1] = 1.0

                if(mario_state.shape[0] == 0):
                    mario_state = np.array([1.0])
                else:
                    mario_state = np.vstack((mario_state,np.array([1.0])))

            elif(data[i] == "false"):
                obs_array[i-1] == 0.0

                if(mario_state.shape[0] == 0):
                    mario_state = np.array([0.0])
                else:
                    mario_state = np.vstack((mario_state,np.array([0.0])))
            else:
                obs_array[i-1] = float(data[i])

        
        mayMarioJump = (data[1] == 'true')
        isMarioOnGround = (data[2] == 'true')
#        assert len(data) == obsLength, "Error in data size given %d! Required: %d \n data: %s " % (len(data), obsLength, data)
        k = 0
        level_scene = np.array([])
        for i in range(22):
            for j in range(22):
                levelScene[i, j] = int(data[k + 3])
                binFeat = np.zeros([14,1])

                if(levelScene[i,j] != 0):
                    binFeat[binary_labels(levelScene[i,j])] = 1

                if(level_scene.shape[0] == 0):
                    level_scene = binFeat
                else:
                    level_scene = np.vstack([level_scene,binFeat])

           

                k += 1
        k += 3
        marioFloats = (float(data[k]), float(data[k + 1]))
        #obs_array = np.zeros(2)
        #obs_array[0] = marioFloats[0]
        #obs_array[1] = marioFloats[1]
        k += 2  
        k_a = k
        action = 0; 
        i = 0
        action_vec = np.zeros([5,1])
        while k < k_a+5:
       
            if(data[k] == "true"):
                action_vec[i] = 1
                action += 2**i
            i+=1
            k+=1


        while k < len(data):
            enemiesFloats.append(float(data[k]))
            k += 1

        if(len(pastLevels) < 4):
            pastLevels.append(level_scene)
        else:
            pastLevels.pop(0)
            pastLevels.append(level_scene)

        if(len(pastActions) < 6):
            pastActions.append(action_vec)
        else:
            pastActions.pop(0)
            pastActions.append(action_vec)

        # IPython.embed()
        obs_array = asmbleObsv(pastLevels,pastActions,mario_state)
        return (mayMarioJump, isMarioOnGround, marioFloats, enemiesFloats, levelScene, dummy,action,obs_array)
    else:
        raise "Wrong format or corrupted observation..."
