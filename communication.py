from __future__ import division
import numpy as np

%matplotlib inline
from matplotlib import pyplot as plot
import seaborn as sns
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf')

def common_value(*args):
    if len(args) == 0:
        raise ValueError("argument is empty")
    x = args[0]
    assert all(x == y for y in args[1:]), "no common value"
    return x
    
def rand_p(size):
    a = np.random.random(size - 1)
    a.sort()
    return np.diff(np.concatenate(((0,), a, (1,))))

def cummean(a):
    return np.true_divide(np.cumsum(a), range(1, len(a) + 1))

def uniform_pdist(size):
    return (1/size,) * size

class System:

    @classmethod
    def from_rand(self, n_inputs, n_outputs):
        return System(np.array(tuple(rand_p(n_outputs) for input_ in range(n_inputs))))
    
    def __init__(self, matrix):
        assert isinstance(matrix, np.ndarray)
        # this assertion fails when the matrices are too large because of rounding error
        #assert all(np.dot(matrix, (1,) * matrix.shape[1]) == (1,) * matrix.shape[0])
        self.matrix = matrix
        
    def n_inputs(self):
        return self.matrix.shape[0]
        
    def n_outputs(self):
        return self.matrix.shape[1]
        
    def success_p(self, input_p = None):
        if input_p is None:
            input_p = uniform_p(self.n_inputs())
        return np.trace(np.dot(np.diag(input_p), self.matrix))
        
    def randoutput(self, input_p = None):
        if isinstance(input_p, int):
            input_ = input_p
        else:
            input_ = np.random.choice(range(self.n_inputs()), p=input_p)
        return np.random.choice(range(self.n_outputs()), p=self.matrix[input_, :])
        
class ProductSystem:

    @classmethod
    def from_rand(self, n_input_meanings, n_signals, n_output_meanings):
        return ProductSystem(System.from_rand(n_input_meanings, n_signals), System.from_rand(n_signals, n_output_meanings))
        
    @classmethod
    def rand_dyad(self, n_meanings, n_signals):
        return ProductSystem.from_rand(n_meanings, n_signals, n_meanings)
    
    @classmethod
    def rand_agent(self, n_meanings, n_signals):
        sys = System.from_rand(n_meanings, n_signals)
        return ProductSystem(sys, System(np.transpose(sys.matrix)))

    def __init__(self, psys, rsys):
        assert isinstance(psys, System)
        assert isinstance(rsys, System)
        assert psys.n_outputs() == rsys.n_inputs()
        self.psys = psys
        self.rsys = rsys
    
    def n_input_meanings(self):
        return self.psys.n_inputs()
        
    def n_signals(self):
        return common_value(self.psys.n_outputs(), self.rsys.n_inputs())
        
    def n_output_meanings(self):
        return self.rsys.n_outputs()
        
    def sys(self):
        return System(np.dot(self.psys.matrix, self.rsys.matrix))
        
    def success_p(self, input_meaning_p = None):
        return self.sys().success_p(input_meaning_p)
        
    def randoutput(self, input_meaning_p = None):
        signal = self.psys.randoutput(input_meaning_p)
        return self.rsys.randoutput(signal)
        
    def randsuccess(self, input_meaning_p = None):
        assert(self.n_input_meanings() == self.n_output_meanings())
        if isinstance(input_meaning_p, int):
            input_meaning = input_meaning_p
        else:
            input_meaning = np.random.choice(range(self.n_input_meanings()), p=input_meaning_p)
        return self.randoutput(input_meaning) == input_meaning
        
def test_success_formula(n_meanings, n_signals, n_trials, n_runs):
    input_meaning_p = rand_p(n_meanings)
    print input_meaning_p
    print
    dyad = ProductSystem.rand_agent(n_meanings, n_signals)
    print dyad.psys.matrix
    print
    print dyad.rsys.matrix
    print
    data = tuple(cummean(tuple(dyad.randsuccess(input_meaning_p) for j in range(n_trials))) for i in range(n_runs))
    graph = sns.tsplot(data, err_style='unit_traces')
    graph.set(xlabel='Number of trials', ylabel='Success proportion', ylim=[0,1])
    success_p = dyad.success_p(input_meaning_p)
    print success_p
    graph2 = sns.tsplot((success_p,) * n_trials, color=sns.color_palette()[1])