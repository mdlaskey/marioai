
__author__ = "Michael Laskey"

from experiments.episodicexperiment import EpisodicExperiment
from tasks.mariotask import MarioTask
from agents.forwardagent import ForwardAgent
from agents.forwardrandomagent import ForwardRandomAgent
from agents.ahude import Ahude
from agents.dagger import Dagger 
import matplotlib.pyplot as plt
import numpy as np
from analysis import Analysis

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
        


 
    def test(self,rounds=20,iterations=35, learning_samples=1, eval_samples=1,  prefix = ''):
        num_ask_help = np.zeros(iterations) 
        num_mismatch = np.zeros(iterations)
        avg_data = np.zeros(iterations) 
        avg_distance = np.zeros(iterations)
        avg_precision = np.zeros(iterations)
        avg_human_input = np.zeros(iterations)
        
        num_ask_help_r = np.zeros((rounds, iterations))
        num_mismatch_r = np.zeros((rounds, iterations))
        avg_data_r = np.zeros((rounds, iterations))
        avg_distance_r = np.zeros((rounds, iterations))
        avg_precision_r = np.zeros((rounds, iterations))
        avg_human_input_r = np.zeros((rounds, iterations))
        sup_data_r = np.zeros((rounds, iterations))
        acc_data = np.zeros((rounds, iterations))
        test_acc_data = np.zeros((rounds, iterations))
        avg_losses_r = np.zeros((rounds, iterations))
        avg_js_r = np.zeros((rounds, iterations))
        # self.exp.task.env.changeLevel()

        for r in range(rounds):
            print "Trial: " + str(r)
            self.agent.initialTraining = True
            self.exp.doEpisodes(1)
            self.agent.newModel()
            self.agent.saveModel()
            self.agent.initialTraining = False
            self.agent.loadModel()
            self.agent.reset()


            distances = np.zeros(iterations)
            mis_match = np.zeros(iterations)
            data = np.zeros(iterations)
            losses = np.zeros(iterations)
            js = np.zeros(iterations)
            sup_data = np.zeros(iterations)
            num_help = np.zeros(iterations)
            precision = np.zeros(iterations)
            human_input = np.zeros(iterations)
            acc = np.zeros(iterations)
            test_acc = np.zeros(iterations)
            for t in range(iterations):
                
                rewards, loss, j, sup_rewards,  = self.exp.doEpisodes(1, learning_samples, eval_samples)
                rewards = np.mean(rewards, axis=0)
                loss = np.mean(loss, axis=0)
                j = np.mean(j, axis=0)

                data[t] = rewards[-1]
                losses[t] = loss[-1]
                js[t] = j[-1]

                # data[t] = rewards[0][-1]             # taking from the first sample
                if self.agent._name == 'supervise':
                    sup_rewards = np.mean(sup_rewards, axis=0)
                    sup_data[t] = sup_rewards[-1]
                    #sup_data[t] = sup_rewards[0][-1]    # taking from the first sample
                #self.agent.updateModel()
                acc[t] = self.agent.learner.accs
                test_acc[t] = self.agent.learner.test_accs

                # if(self.agent._getName() == 'Ahude'):
                #     num_help[t] = self.agent.getNumHelp()
                #     mis_match[t] = self.agent.getMismatch()
                #     self.agent.off = True
                #     rewards = self.exp.doEpisodes(1)
                #     self.agent.off = False

                # size = len(rewards[0])
                size = len(rewards)
                    
                #distances[t] = rewards[0][size-1]
                distances[t] = rewards[size-1]

                precision[t] = self.agent.learner.getPrecision()
                human_input[t] = self.agent.getNumHumanInput()
                self.agent.reset()
             
            # if(self.agent._getName() == 'Ahude'):
            #     num_ask_help = num_ask_help + num_help
            #     num_mismatch = num_mismatch + mis_match


            avg_data_r[r, :] = data
            avg_losses_r[r, :] = losses
            avg_js_r[r, :] = js

            if self.agent._name == 'supervise':
                sup_data_r[r, :] = sup_data
            acc_data[r, :] = acc
            test_acc_data[r, :] = test_acc


            # plot single trial
            if self.agent._name == 'supervise':

                np.save('./data/' + prefix + 'loss_round' + str(r) + '.npy', losses)
                np.save('./data/' + prefix + 'sl_reward_round' + str(r) + '.npy', data)
                np.save('./data/' + prefix + 'sup_reward_round' + str(r) + '.npy', sup_data)
                np.save('./data/' + prefix + 'acc_round' + str(r) + '.npy', acc)
                np.save('./data/' + prefix + 'test_acc_round' + str(r) + '.npy', test_acc)
                np.save('./data/' + prefix + 'js_round' + str(r) + '.npy', js)

                a = Analysis()
                a.get_perf(np.array([sup_data]), range(iterations))
                a.get_perf(np.array([data]), range(iterations))
                a.plot(names=['Supervisor', 'Supervised Learning'], label='Rewards', filename='./results/' + prefix + 'return_plot' + str(r) + '.eps')
    
                a = Analysis()
                a.get_perf(np.array([losses]), range(iterations))
                a.plot(names=['Supervised Learning'], label='Loss', filename='./results/' + prefix + 'loss_plot' + str(r) + '.eps', ylims=[0, 1])

                a = Analysis()
                a.get_perf(np.array([js]), range(iterations))
                a.plot(names=['Supervised Learning'], label='J()', filename='./results/' + prefix + 'js_plot' + str(r) + '.eps')


            elif self.agent._name == 'dagger':

                np.save('./data/' + prefix + 'loss_round' + str(r) + '.npy', losses)
                np.save('./data/' + prefix + 'dagger_reward_round' + str(r) + '.npy', data)
                np.save('./data/' + prefix + 'acc_round' + str(r) + '.npy', acc)
                np.save('./data/' + prefix + 'test_acc_round' + str(r) + '.npy', test_acc)
                np.save('./data/' + prefix + 'js_round' + str(r) + '.npy', js)

                a = Analysis()
                a.get_perf(np.array([data]), range(iterations))
                a.plot(names=['DAgger'], label='Reward', filename='./results/' + prefix + 'return_plot' + str(r) + '.eps')
                
                a = Analysis()
                a.get_perf(np.array([losses]), range(iterations))
                a.plot(names=['DAgger'], label='Loss', filename='./results/' + prefix + 'loss_plot' + str(r) + '.eps', ylims=[0,1])

                a = Analysis()
                a.get_perf(np.array([js]), range(iterations))
                a.plot(names=['DAgger'], label='J()', filename='./results/' + prefix + 'js_plot' + str(r) + '.eps')


            avg_data = data+avg_data
            avg_distance = distances+avg_distance 
            avg_precision = precision+avg_precision
            avg_human_input = avg_human_input + human_input
          
            # self.exp.task.env.changeLevel()
        
        num_ask_help = num_ask_help/rounds
        num_mismatch = num_mismatch/rounds
        avg_data = avg_data/rounds
        avg_distance = avg_distance/rounds 
        avg_precision = avg_precision/rounds 
        avg_human_input = avg_human_input/rounds
        self.exp.task.env.setLevelBack()
        
        #print avg_distance
        if self.agent._name == 'supervise':
            return avg_data_r, sup_data_r, acc_data, avg_losses_r, avg_js_r, test_acc_data
        else:
            return avg_data_r, None, acc_data, avg_losses_r, avg_js_r, test_acc_data
        #return avg_data,avg_distance,num_mismatch,num_ask_help,avg_precision,avg_human_input, avg_data_r

             
             
        
  
