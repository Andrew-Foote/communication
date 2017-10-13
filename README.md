This short script implements a simple model of communication where entities attempt to communicate meanings to each other by signalling. Each entity has the capability to map meanings to signals, when producing a signal, or signals to meanings, when receiving a signal. Communications are not always successful as the signal might be interpreted as having a different meaning from the one that the producer intended.

The script contains classes that can be used to represent these communicating entities (System for a single entity, ProductSystem for a pair of communicating entities). Methods of these classes allow for simulating communication; see the example below.

>>> a = np.ndarray((1, 0, 0),
                   (0, 1, 0),
                   (0, 0, 1))
>>> sys = ProductSystem(a, a)
>>> sys.randsuccess()
True

Here, the first line creates a NumPy array which can be thought of as representing an entity and their signal-processing system; the rows correspond to meanings and the columns correspond to three signals. So there are three meanings that the entity may try to communicate and three signals it may use. If it tries to communicate meaning 1, the top row shows that it has a 100% chance of using signal 1 to do that, and zero chance of using signals 2 or 3. If the top row was 1/3, 1/3, 1/3 instead, it would have an equal chance of using any of the three signals. The other rows can be read in the same way.

The second line creates a pair of communicating entities, which both have the same system, represented by the variable 'a'. The third line simulates a communication between these two entities in which the meaning that the producer attempts to convey is selected at random. It always returns True, because the layout of the array 'a' is such that meaning 1 always maps to signal 1 and signal 1 always maps to meaning 1, meaning 2 always maps to signal 2 and signal 2 always maps to meaning 2, and so on.

The ProductSystem class also contains a method that can be used to calculate the probability of success for a given communication, using a simple formula. There is a function ('test_success_formula') in the file which tests the accuracy of this calculation by simulating communications and calculating the actual success rates, and plotting the resulting data as a graph so that the user can assess by eye whether the calculated probability of success is reliably close to the success rate over many trials.