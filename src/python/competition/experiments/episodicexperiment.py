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
            self.agent.reset_task()
            while not self.task.isFinished():
                r = self._oneInteraction()
                rewards.append(r)
            all_rewards.append(rewards)
        return all_rewards


    def _run_dagger_traj(self, learning_samples, eval_samples):
        if self.agent.initialTraining:
            self.agent.isLearning = True
            self._runTrajectory(1)
            self.agent.isLearning = False
        else:
            self.agent.isLearning = True
            
            all_rewards = []
            for _ in range(max(learning_samples, eval_samples)):
                if _ >= learning_samples:
                    self.agent.isLearning  = False
                else:
                    print "Learning"
                rewards = []
                self.stepid = 0
                self.agent.newEpisode()
                self.task.reset()
                self.agent.reset_task()
                while not self.task.isFinished():
                    r = self._oneInteraction()
                    rewards.append(r)
                if _ < eval_samples:
                    print "evaluating"
                    all_rewards.append(rewards)
                print "\n\n\n"
            self.agent.updateModel()

            return all_rewards, None

    def _run_supervise_traj(self, learning_samples, eval_samples):
        print "Supervised learning starts here"
        if self.agent.initialTraining:
            print "INITIAL TRAINING"
            self.agent.isLearning = True
            self._runTrajectory(1)
            self.agent.isLearning = False
        else:
            all_rewards = self._runTrajectory(eval_samples)

            self.agent.isLearning = True
            sup_rewards = self._runTrajectory(learning_samples)
            self.agent.isLearning = False

            self.agent.updateModel()
            print "REWARD LENGTH: " + str(len(all_rewards))
            print "SUP LENGTH: " + str(len(sup_rewards))
            return all_rewards, sup_rewards

    def doEpisodes(self, number = 1, learning_samples=1, eval_samples=1):
        """ returns the rewards of each step as a list """
        all_rewards = []
        self.task.env.changeLevel()

        self.learning_samples = learning_samples
        self.eval_samples = eval_samples

        #if self.agent.initialTraining or self.agent._name == 'dagger':
        if self.agent._name == 'dagger':
            # print "Running initial training"
            # self.agent.isLearning = True
            # all_rewards = self._runTrajectory(number)
            # if not self.agent.initialTraining:
            #     self.agent.updateModel()
            # self.agent.isLearning = False
            # print "REWARDS: " + str(all_rewards)
            # return all_rewards, None
            return self._run_dagger_traj(self.learning_samples, self.eval_samples)
        else:
            # Must be supervise here
            # run the sample trial
            # all_rewards = self._runTrajectory(eval_samples)

            # # run the learning trial
            # self.agent.isLearning = True
            # sup_rewards = self._runTrajectory(learning_samples)
            # self.agent.isLearning = False
            # self.agent.updateModel()
            # print sup_rewards
            # # return reward from sampled trial
            # return all_rewards, sup_rewards
            return self._run_supervise_traj(self.learning_samples, self.eval_samples)


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
