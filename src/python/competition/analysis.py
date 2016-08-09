#from svm import LinearSVM
#from net import Net
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import IPython
import cPickle

class Analysis():
    
    def __init__(self):
        #self.density = np.zeros([H,W])
        self.x = None
        self.mean = None
        self.err = None


    def compute_std_er_m(self,data):
        n = data.shape[0]
        std = np.std(data)

        return std/np.sqrt(n)

    def compute_m(self,data):
        n = data.shape[0]
        return np.sum(data)/n

    def get_perf(self,data, x, color=None):
        #SAve each mean and err at the end
        #iters = data.shape[1]
        mean = np.zeros(len(x))
        err = np.zeros(len(x))
        #x = np.zeros(iters)

        for i in range(len(x)):
            mean[i] = self.compute_m(data[:,i]) # for iteration i, compute mean reward across trials
            #x[i] = i
            err[i] = self.compute_std_er_m(data[:,i])
        
        if color is None:
            plt.errorbar(x,mean,yerr=err,linewidth=5.0)
        else:
            plt.errorbar(x,mean,yerr=err,linewidth=5.0, color=color)


        self.mean = mean
        self.err = err
        self.x = x

        return [mean,err]
    
    def set_errorbar(self):
        plt.errorbar(self.x,self.mean,yerr=self.err,linewidth=5.0)
        

    def save(self, filename='analysis.p'):
        #[self.mean, self.err, self.density, self.train_loss, self.test_loss]
        return cPickle.dump(self, open(filename, 'wb'))

    @staticmethod
    def load(filename):
        a = cPickle.load(open(filename, 'rb'))
        if a.x is not None and a.mean is not None:    
            a.set_errorbar()
        return a

    def plot(self, names = None, label = None, filename=None, ylims=None):
        if label is None:
            label = 'Reward'
        plt.ylabel(label)
        plt.xlabel('Iterations')

        if names is None:
            names = ['Sup']        
            #names = ['NN_Supervise','LOG_Supervisor']
        plt.legend(names,loc='upper center',prop={'size':10}, bbox_to_anchor=(.5, 1.12), fancybox=False, ncol=len(names))

        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 22},

        axes = plt.gca()
        #axes.set_xlim([0,self.iters])
        if not ylims is None:
            axes.set_ylim(ylims)
            #axes.set_ylim([-60, 100])
        
        if filename is not None:
            plt.savefig(filename, format='eps', dpi=1000)
        #plt.show(block=True)
        plt.show(block=False)
        plt.close()

    def reset_density(self):
        self.density = np.zeros([self.h,self.w])        

    def count_states(self,all_states):
        N = all_states.shape[0]
        current_density = np.zeros([self.h,self.w])
        for i in range(N):
            x = all_states[i,0]
            y = all_states[i,1]
            current_density[x,y] = current_density[x,y] +1.0
        
        norm = np.sum(current_density)

        current_density = current_density/norm
        self.density = self.density+ current_density/self.iters

        

    def compile_density(self):
        density_r = np.zeros([self.h*self.w,3])

        self.m_val = 0.0
        for w in range(self.w):
            for h in range(self.h):
                val = self.density[w,h]
                if(val > self.m_val):
                    self.m_val = val
                val_r = np.array([h,w,val])
                density_r[w*self.w+h,:] = val_r
        print "M VAL ", self.m_val
        return density_r


    def plot_setup(self,weights=None,color='density'):
        plt.xlabel('X')
        plt.ylabel('Y')
        cm = plt.cm.get_cmap('gray_r')

        axes = plt.gca()
        axes.set_xlim([0,15])
        axes.set_ylim([0,15])
        density_r = self.compile_density()

        #print density_r
        #print np.sum(density_r)
        if(color == 'density'):
            a = np.copy(density_r[:,2])
            a[112] = 0.0
            plt.scatter(density_r[:,1],density_r[:,0], c= a, cmap = cm,s=300,edgecolors='none') 
        else: 
            a = weights
            plt.scatter(weights[:,0],weights[:,1], c= weights[:,2], cmap = cm,s=300) 
            
        #plt.scatter(density_r[:,1],density_r[:,0], c= density_r[:,2],cmap = cm,s=300,edgecolors='none', color='blue')

        #save each density if called 
       
        
        # #PLOT GOAL STATE
        if not self.rewards == None and not self.sinks == None:
            r_xs = []
            r_ys = []
            for r in self.rewards:
                r_xs.append(r.x)
                r_ys.append(r.y)

            s_xs = []
            s_ys = []
            for s in self.sinks:
                s_xs.append(s.x)
                s_ys.append(s.y)
            plt.scatter([r_xs], [r_ys], c='green', s=300)
            plt.scatter([s_xs], [s_ys], c='red', s=300)
            
        else:
            # PLOT GOAL STATE
            plt.scatter([7],[7], c= 'green',s=300)
            # PLOT SINK STATE
            plt.scatter([4],[2], c= 'red',s=300)
        


    def plot_scatter(self,weights=None,color='density'):
        self.plot_setup(weights, color)
        #plt.show()
        plt.show(block=False)
        plt.close()

    def plot_save(self, name, weights=None, color='density'):
        self.plot_setup(weights, color)
        plt.savefig(name, format='eps', dpi=1000)

    def save_states(self, name):
        self.plot_save(name)

    def show_states(self):
        self.plot_scatter()


    def show_weights(self,weights):
        self.plot_scatter(weights=weights,color = 'weights')

