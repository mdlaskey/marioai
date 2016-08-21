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

#from pybrain.... episodic import EpisodicExperiment
#TODO: reset sends: vis, diff=, lt=, ll=, rs=, mariomode, time limit, pw,
# with creatures, without creatures HIGH.
# send creatures.
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

    
    # # #test dagger
    #iterations = 40
    #rounds = 1
    iterations = 50
    rounds = 15
    #agent = Dagger(IT,useKMM = False)
    #exp = EpisodicExperiment(task, agent) 
    #T = Tester(agent,exp)
    #dagger_results = T.test(rounds = rounds,iterations = iterations)
    #dagger_data = dagger_results[-1]
    #dagger_results = dagger_results[:-1]
    #results.append(dagger_results)
    #names.append('dagger')
    #pickle.dump(results,open('results.p','wb'))

    #agent = Dagger(IT, useKMM=False)
    #exp = EpisodicExperiment(task, agent)
    #T = Tester(agent, exp)
    #dagger_data, _, acc = T.test(rounds = rounds, iterations = iterations)
     
    agent = Supervise(IT,useKMM = False)
    exp = EpisodicExperiment(task, agent) 
    T = Tester(agent,exp)
    sl_data, sup_data, acc = T.test(rounds = rounds, iterations = iterations)

    np.save('./data/sup_data.npy', sup_data)
    np.save('./data/sl_data.npy', sl_data)
    np.save('./data/acc.npy', acc)    
    
    IPython.embed()

    analysis = Analysis()
    analysis.get_perf(sup_data, range(iterations))
    analysis.get_perf(sl_data, range(iterations))
    analysis.plot(names=['Supervisor', 'Supervised Learning'], label='Reward', filename='./results/return_plots.eps')#, ylims=[0, 1600])

    acc_a = Analysis()
    acc_a.get_perf(acc, range(iterations))
    acc_a.plot(names=['Supervised Learning Acc.'], label='Accuracy', filename='./results/acc_plots.eps')

    """


    agent = Dagger(IT,useKMM = False)
    exp = EpisodicExperiment(task, agent) 
    T = Tester(agent,exp)
    dagger_data, _, acc = T.test(rounds = rounds, iterations = iterations)

    np.save('./data/dagger_data.npy', dagger_data)
    np.save('./data/acc.npy', acc)    
    
    IPython.embed()

    analysis = Analysis()
    analysis.get_perf(dagger_data, range(iterations))
    analysis.plot(names=['DAgger'], label='Reward', filename='./results/return_plots.eps')

    acc_a = Analysis()
    acc_a.get_perf(acc, range(iterations))
    acc_a.plot(names=['DAgger Acc.'], label='Accuracy', filename='./results/acc_plots.eps')

    """
    
    #agent = Supervise(IT,useKMM = False)
    #exp = EpisodicExperiment(task, agent) 
    #T = Tester(agent,exp)
    #supervise_results = T.test(rounds = rounds, iterations = iterations)
    #supervise_data = supervise_results[-1]
    #supervise_results = supervise_results[:-1]
    #results.append(supervise_results)
    #names.append('supervise')
    #pickle.dump(results,open('results.p','wb'))

    #IPython.embed()

    #analysis = Analysis()
    #analysis.get_perf(supervise_data, results[1][5])
    #analysis.get_perf(dagger_data, results[0][5])
    #analysis.plot(names=['Supervise', 'DAgger'], label='Reward', filename='./return_plot.eps')#, ylims=[-1, 0])




    # agent = Sheath(IT,useKMM = False,sigma = 1.0)
    # exp = EpisodicExperiment(task, agent) 
    # T = Tester(agent,exp)
    # dagger_results = T.test(rounds = 10,iterations = 35)
    # results.append(dagger_results)
    # names.append('sheath_1')
    # pickle.dump(results,open('results.p','wb'))

    # agent = Sheath(IT,useKMM = False,sigma = 1e-1)
    # exp = EpisodicExperiment(task, agent) 
    # T = Tester(agent,exp)
    # dagger_results = T.test(rounds = 10,iterations = 35)
    # results.append(dagger_results)
    # names.append('sheath_1')
    # pickle.dump(results,open('results.p','wb'))


    
    # agent = Sheath(IT,useKMM = False,sigma = 0.5)
    # exp = EpisodicExperiment(task, agent) 
    # T = Tester(agent,exp)
    # dagger_results = T.test(rounds = 10,iterations = 35)
    # results.append(dagger_results)
    # names.append('sheath_1')

    # pickle.dump(results,open('results.p','wb'))
    # agent = Sheath(IT,useKMM = False,sigma = 1e-1)
    # exp = EpisodicExperiment(task, agent) 
    # T = Tester(agent,exp)
    # dagger_results = T.test(rounds = 4,iterations = 35)
    # results.append(dagger_results)
    # names.append('sheath_1')
    

    # agent = Sheath(IT,useKMM = False,sigma = 1e-2)
    # exp = EpisodicExperiment(task, agent) 
    # T = Tester(agent,exp)
    # dagger_results = T.test(rounds = 4,iterations = 35)
    # results.append(dagger_results)
    # names.append('sheath_1')
    # # # # # #test big ahude
    # agent = Ahude(IT,f,gamma = 1e-2,labelState = True, useKMM = True)
    # exp = EpisodicExperiment(task, agent) 
    # T = Tester(agent,exp)
    # ahude_big_results = T.test(rounds = 3)
    # results.append(ahude_big_results)
    # names.append('ahude_1e-1')

    # pickle.dump(results,open('results.p','wb'))


    # # # # # #test med ahude
    # agent = Ahude(IT,f,gamma = 1e-2,labelState = False,useKMM = True)
    # exp = EpisodicExperiment(task, agent) 
    # T = Tester(agent,exp)
    # ahude_med_results = T.test(rounds = 3)
    # results.append(ahude_med_results)
    # names.append('ahude_1e-2')
    
    # # #

    # # # # # # #test small ahude 
    # agent = Ahude(IT,f,gamma = 1e-3)
    # exp = EpisodicExperiment(task, agent) 
    # T = Tester(agent,exp)
    # ahude_small_results = T.test() 
    # results.append(ahude_small_results)
    # names.append('ahude_1e-3')
    
 
    # pickle.dump(results,open('results.p','wb'))

    #plt.figure(1)
    #for i in range(len(results)):
    #    plt.plot(results[i][5],results[i][1])
    
    
    #plt.legend(names,loc='upper left')

    # plt.figure(2)
    # for i in range(len(results)):
    #     plt.plot(results[i][0])

    # plt.legend(names,loc='upper left')

    # plt.figure(3)
    # for i in range(0,len(results)):
    #     plt.plot(results[i][3])

    # plt.legend(names,loc='upper left')


    plt.show()
    
    IPython.embed()
    f.close()           
       

    #agent.saveModel()
    print "finished"



if __name__ == "__main__":
    main()
else:
    print "This is module to be run rather than imported."
