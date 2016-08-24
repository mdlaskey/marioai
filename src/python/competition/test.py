from analysis import Analysis
import numpy as np
data = np.load("data/dt-sup-eval-eval_sl_reward_round0.npy")
if len(data.shape) == 1:
#    print "hi"
    data = np.array([data])
#IPython.embed()
a = Analysis()
a.get_perf(data, range(5))
a.plot(filename='test.eps')
