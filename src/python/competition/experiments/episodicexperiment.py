__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$May 12, 2009 11:18:19 PM$"

from experiment import Experiment
import IPython 


#class EpisodicExperiment(Experiment):
#    """ The extension of Experiment to handle episodic tasks. """
#
#    def doEpisodes(self, number = 1):
#        """ returns the rewards of each step as a list """
#        all_rewards = []
#        for dummy in range(number):
#            rewards = []
#            self.stepid = 0
#            # the agent is informed of the start of the episode
#            self.agent.newEpisode()
#            self.task.reset()
#            while not self.task.isFinished():
#                r = self._oneInteraction()
#                rewards.append(r)
#            all_rewards.append(rewards)
#        return all_rewards


from experiment import Experiment

class EpisodicExperiment(Experiment):
    """ The extension of Experiment to handle episodic tasks. """
    
    def _runTrajectory(self, number):
        all_rewards = []
        for dummy in range(number):
            rewards = []
            self.stepid = 0
            # the agent is informed of the start of the episode
            self.agent.newEpisode()
            self.task.reset()
            while not self.task.isFinished():
                r = self._oneInteraction()
                rewards.append(r)
            all_rewards.append(rewards)
        return all_rewards

    def doEpisodes(self, number = 1):
        """ returns the rewards of each step as a list """
        all_rewards = []
        #self.task.env.changeLevel()

        if self.agent.initialTraining or self.agent._name == 'dagger':
            print "Running initial training"
            self.agent.isLearning = True
            all_rewards = self._runTrajectory(number)
            self.agent.isLearning = False
            return all_rewards, None
        else:
            # Must be supervise here
            # run the sample trial
            all_rewards = self._runTrajectory(number)

            # run the learning trial
            self.agent.isLearning = True
            sup_rewards = self._runTrajectory(number)
            self.agent.isLearning = False
            self.agent.updateModel()
            print sup_rewards
            # return reward from sampled trial
            return all_rewards, sup_rewards


        # # DO THE SUPERVISED TRAJECTORY FIRST
        # if self.agent._name == 'supervise':
        #     print "Running initial trail"
        #     self.agent.isLearning = True
        #     for dummy in range(number):
        #         rewards = []
        #         self.stepid = 0
        #         # the agent is informed of the start of the episode
        #         self.agent.newEpisode()
        #         self.task.reset()
        #         while not self.task.isFinished():
        #             r = self._oneInteraction()
        #             rewards.append(r)
        #         all_rewards.append(rewards)
        #     if self.agent.initialTraining:
        #         self.agent.newModel()
        #         self.agent.saveModel()
        #         self.agent.loadModel()
        #     self.agent.isLearning=False
        #     self.agent.updateModel()




        # all_rewards = []
        # for dummy in range(number):
        #     rewards = []
        #     self.stepid = 0
        #     # the agent is informed of the start of the episode
        #     self.agent.newEpisode()
        #     self.task.reset()
        #     while not self.task.isFinished():
        #         r = self._oneInteraction()
        #         rewards.append(r)
        #     all_rewards.append(rewards)
        # print "REWARDS: " + str(all_rewards)
        # return all_rewards
        

#class EpisodicExperiment(Experiment):
#    """
#    Documentation
#    """
#
#    statusStr = ("Loss...", "Win!")
#    agent = None
#    task = None
#
#    def __init__(self, agent, task):
#        """Documentation"""
#        self.agent = agent
#        self.task = task
#
#    def doEpisodes(self, amount):
#        for i in range(amount):
#            self.agent.newEpisode()
#            self.task.startNew()
#            while not self.task.isFinished():
#                obs = self.task.getObservation()
#                if len(obs) == 3:
#                    self.agent.integrateObservation(obs)
#                    self.task.performAction(self.agent.produceAction())
#                
#            r = self.task.getReward()
#            s = self.task.getStatus()
#            print "Episode #%d finished with status %s, fitness %f..." % (i, self.statusStr[s], r)
#            self.agent.giveReward(r)
