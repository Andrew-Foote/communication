from __future__ import division
import numpy as np

#%matplotlib inline # remove this if running outside Jupyter
from matplotlib import pyplot as plot
import seaborn as sns
#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('svg', 'pdf')

def common_value(*args):
    """Returns the common value of the arguments, if they are all equal.
Otherwise, raises ValueError."""
    if len(args) == 0:
        raise ValueError("argument is empty")
    x = args[0]
    assert all(x == y for y in args[1:]), "no common value"
    return x
    
def rand_p(size):
    """Randomly generates a 1-dimensional NumPy array containing non-negative
real numbers which add up to exactly 1, of the given size. The marginal
probability distribution of each item in the array is beta(1, size - 1)."""
    a = np.random.random(size - 1)
    a.sort()
    return np.diff(np.concatenate(((0,), a, (1,))))

def cummean(a):
    """Returns a 1-dimensional NumPy array consisting of the first item in a,
the mean of the first two items in a, the mean of the first three items in a,
and so on."""
    return np.true_divide(np.cumsum(a), range(1, len(a) + 1))

def uniform_p(size):
    return (1/size,) * size

def conc_p(size, i):
    a = np.zeros(size)
    a[i] = 1
    return a

class System:
    """A system with a number of inputs and a number of outputs, represented by
a two-dimensional NumPy array with rows corresponding to inputs and columns
corresponding to outputs. The entries are non-negative real numbers which sum
to 1 along each row, so a probability distribution over the columns (outputs)
can be read off each row (input); these probability distributions represent the
behaviour of the system as it responds to inputs."""

    @classmethod
    def from_rand(self, n_inputs, n_outputs):
        """Generates a random System."""
        return System(np.array(tuple(rand_p(n_outputs) for input_ in range(n_inputs))))
    
    def __init__(self, matrix):
        assert isinstance(matrix, np.ndarray)
        # this assertion fails when the matrices are too large because of
        # rounding error
        #assert all(np.dot(matrix, (1,) * matrix.shape[1]) == (1,) * matrix.shape[0])
        self.matrix = matrix
        
    def n_inputs(self):
        return self.matrix.shape[0]
        
    def n_outputs(self):
        return self.matrix.shape[1]
        
    def success_p(self, input_p = None):
        """A 'success' is when the output is the same as the input (i.e. the
indices of the corresponding rows and columns are the same). This method
returns the probability of success upon receiving a random input (the
probability distribution over the inputs can be given as an argument, but by
default it is uniform)."""
        if input_p is None:
            input_p = uniform_p(self.n_inputs())
        return np.trace(np.dot(np.diag(input_p), self.matrix))
        
    def randoutput(self, input_p = None):
        """Returns a randomly-chosen output (as a column index) the system
chooses when responding to a specific randomly-chosen input (the probability
distribution over the inputs can be given as an argument, but by default it is
uniform)."""
        if input_p is None:
            input_p = uniform_p(self.n_inputs())
        input_ = np.random.choice(range(self.n_inputs()), p=input_p)
        return np.random.choice(range(self.n_outputs()), p=self.matrix[input_, :])
        
class ProductSystem:
    """A pair of input-output Systems. One (the producer) takes meanings as
input and outputs signals. These signals are received by the other (the
receiver) as input, and the receiver then outputs a meaning. This can be
thought of as representing a communication between two entities: the input
meaning is the meaning that the communicating entity intends to communicate,
the signal is how they communicate it, the output meaning is what the other
entity understands from the communication."""
    
    @classmethod
    def from_rand(self, n_input_meanings, n_signals, n_output_meanings):
        """Generates a random ProductSystem."""
        return ProductSystem(System.from_rand(n_input_meanings, n_signals),
                             System.from_rand(n_signals, n_output_meanings))
        
    @classmethod
    def rand_dyad(self, n_meanings, n_signals):
        """Like from_rand, but number of input meanings and number of output
meanings are assumed to be identical."""
        return ProductSystem.from_rand(n_meanings, n_signals, n_meanings)

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
        """The pair of entities can be thought of as a single input-output
system parsing meanings; this method returns the System object representing
that system."""
        return System(np.dot(self.psys.matrix, self.rsys.matrix))
        
    def success_p(self, input_meaning_p = None):
        """Like the success_p function of System; 'success' is when the
receiver outputs the same signal that was input to the producer, i.e.
communication is successful."""
        return self.sys().success_p(input_meaning_p)
        
    def randoutput(self, input_meaning_p = None):
        """Like the randoutput function of System."""
        signal = self.psys.randoutput(input_meaning_p)
        return self.rsys.randoutput(conc_p(self.n_signals(), signal))
        
    def randsuccess(self, input_meaning_p = None):
        """Runs the randoutput method and returns True if the output meaning
was the same as the input meaning (otherwise False)."""
        assert(self.n_input_meanings() == self.n_output_meanings())
        if input_meaning_p is None:
            input_meaning_p = uniform_p(self.n_input_meanings())
        input_meaning = np.random.choice(range(self.n_input_meanings()), p=input_meaning_p)
        return self.randoutput(conc_p(self.n_input_meanings(), input_meaning)) == input_meaning
        
def test_success_formula(n_meanings, n_signals, n_trials, n_runs):
    """Generates a random probability distribution over input meanings, and a
random ProductSystem using the rand_dyad classmethod. Then tests whether the
success_p function gives the right probability of successful communication
for this ProductSystem, by running the randsuccess method repeatedly and
recording the cumulative proportion of successes in a graph.

A number of 'runs' can be carried out. Each one plots a separate line on the
graph. The final graph will show the mean over all the lines, in addition to
the individual lines which are drawn in a lighter colour. A horizontal green
line will also show the success probability as calculated by the success_p
method. The user should judge by the closeness of the other lines to this
green one whether the test was successful."""
    input_meaning_p = rand_p(n_meanings)
    print(input_meaning_p)
    print()
    dyad = ProductSystem.rand_dyad(n_meanings, n_signals)
    print(dyad.psys.matrix)
    print()
    print(dyad.rsys.matrix)
    print()
    data = tuple(cummean(tuple(dyad.randsuccess(input_meaning_p) \
                               for j in range(n_trials))) \
                 for i in range(n_runs))
    graph = sns.tsplot(data, err_style='unit_traces')
    graph.set(xlabel='Number of trials', ylabel='Success proportion', ylim=[0,1])
    success_p = dyad.success_p(input_meaning_p)
    print(success_p)
    graph2 = sns.tsplot((success_p,) * n_trials, color=sns.color_palette()[1])