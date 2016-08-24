__author__ = "Sergey Karakovskiy, sergey at idsia fullstop ch"
__date__ = "$May 12, 2009 11:18:19 PM$"

from experiment import Experiment


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
import cPickle as pickle

class EpisodicExperiment(Experiment):
    """ The extension of Experiment to handle episodic tasks. """
    
    def _runTrajectory(self, number):
        all_rewards = []
        all_losses = []
        all_js = []
        for dummy in range(number):
            print "     Episode: " + str(dummy)
            self.task.env.changeLevel()
            rewards = []
            self.stepid = 0
            # the agent is informed of the start of the episode
            self.agent.newEpisode()
            self.task.reset()
            self.agent.reset_task()
            # self.agent.reset()
            #print "resetting"
            while not self.task.isFinished():
                r = self._oneInteraction()
                rewards.append(r)
            loss = self.agent.get_loss()
            j_error = self.agent.get_j()
            all_losses.append([loss])
            all_rewards.append([rewards[-1]])
            all_js.append([j_error])
        return all_rewards, all_losses, all_js


    def _run_dagger_traj(self, learning_samples, eval_samples):
        if self.agent.initialTraining:
            self.agent.isLearning = True
            self._runTrajectory(1)
            self.agent.isLearning = False
        else:
            self.agent.isLearning = True
            
            all_rewards = []
            all_losses = []
            all_js = []
            for _ in range(max(learning_samples, eval_samples)):
                self.task.env.changeLevel()
                print "Episode: " + str(_)
                if _ >= learning_samples:
                    self.agent.isLearning  = False
                rewards = []
                self.stepid = 0
                self.agent.newEpisode()
                self.task.reset()
                self.agent.reset_task()
                # self.agent.reset()
                while not self.task.isFinished():
                    r = self._oneInteraction()
                    rewards.append(r)
                if _ < eval_samples:
                    loss = self.agent.get_loss()
                    j_error = self.agent.get_j()
                    all_losses.append([loss])
                    all_js.append([j_error])
                    all_rewards.append([rewards[-1]])
                
            self.agent.updateModel()

            return all_rewards, all_losses, all_js, None


    def _run_supervise_traj(self, learning_samples, eval_samples):
        print "Supervised learning starts here"
        if self.agent.initialTraining:
            print "INITIAL TRAINING"
            self.agent.isLearning = True
            self._runTrajectory(1)
            self.agent.isLearning = False
        else:
            all_rewards, all_losses, all_js = self._runTrajectory(eval_samples)

            self.agent.isLearning = True
            sup_rewards, _, __ = self._runTrajectory(learning_samples)
            self.agent.isLearning = False

            self.agent.updateModel()
            print "REWARD LENGTH: " + str(len(all_rewards))
            print "SUP LENGTH: " + str(len(sup_rewards))
            return all_rewards, all_losses, all_js, sup_rewards


    def doEpisodes(self, number = 1, learning_samples=1, eval_samples=1):
        """ returns the rewards of each step as a list """
        all_rewards = []

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

    def collectEpisodes(self, number=1, learning_samples=1):
        self.learning_samples = learning_samples

        assert self.agent._name == 'supervise'

        self.agent.isLearning = True
        self._runTrajectory(learning_samples)
        self.agent.isLearning = False

    def evalEpisodes(self, round, iteration, directory, eval_samples = 1,):
        assert self.agent._name == 'supervise'

        self.eval_samples = eval_samples

        self.agent.states = pickle.load(open(directory + 'states_' + 'round' + str(round) + '_iter' + str(iteration) + '.p', 'r'))
        self.agent.actions = pickle.load(open(directory + 'actions_' + 'round' + str(round) + '_iter' + str(iteration) + '.p', 'r'))

        self.agent.updateModel()

        self.agent.isLearning = False
        all_rewards, all_losses, all_js = self._runTrajectory(eval_samples)

        return all_rewards, all_losses, all_js, None
        


   

