""" Bayesian networks """

from probability import BayesNet, enumeration_ask, elimination_ask, rejection_sampling, likelihood_weighting, gibbs_ask
from timeit import timeit, repeat
import pickle
import numpy as np

T, F = True, False

class DataPoint:
    """
    Represents a single datapoint gathered from one lap.
    Attributes are exactly the same as described in the project spec.
    """
    def __init__(self, muchfaster, early, overtake, crash, win):
        self.muchfaster = muchfaster
        self.early = early
        self.overtake = overtake
        self.crash = crash
        self.win = win

def generate_bayesnet():
    """
    Generates a BayesNet object representing the Bayesian network in Part 2
    returns the BayesNet object
    """
    bayes_net = BayesNet()
    # load the dataset, a list of DataPoint objects
    data = pickle.load(open("data/bn_data.p","rb"))
    # BEGIN_YOUR_CODE ######################################################
    counts = {
        'MuchFaster': {True: 0, False: 0},
        'Early': {True: 0, False: 0},
        'Overtake': {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0},
        'NotOvertake': {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0},
        'Crash': {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0},
        'NotCrash': {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0},
        'Win': {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0},
        'NotWin': {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0},
    }

    total_data = len(data)

    for point in data:
        counts['MuchFaster'][point.muchfaster] += 1
        counts['Early'][point.early] += 1

        if (point.overtake):
            counts['Overtake'][(point.muchfaster, point.early)] += 1
        else:
            counts['NotOvertake'][(point.muchfaster, point.early)] += 1

        if (point.crash):
            counts['Crash'][(point.muchfaster, point.early)] += 1
        else:
            counts['NotCrash'][(point.muchfaster, point.early)] += 1

        if (point.win):
            counts['Win'][(point.overtake, point.crash)] += 1
        else:
            counts['NotWin'][(point.overtake, point.crash)] += 1
    
    P_MuchFaster = counts['MuchFaster'][True] / total_data
    P_Early = counts['Early'][True] / total_data

    P_Overtake = {
        key: counts['Overtake'][key] / (counts['Overtake'][key] + counts['NotOvertake'][key])
        for key in counts['Overtake']
    }

    P_Crash = {
        key: counts['Crash'][key] / (counts['Crash'][key] + counts['NotCrash'][key])
        for key in counts['Crash']
    }

    P_Win = {
        key: counts['Win'][key] / (counts['Win'][key] + counts['NotWin'][key])
        for key in counts['Win']
    }


    T, F = True, False
    bayes_net = BayesNet([
        ('MuchFaster', '', P_MuchFaster),
        ('Early', '', P_Early),
        ('Overtake', 'MuchFaster Early', P_Overtake),
        ('Crash', 'MuchFaster Early', P_Crash),
        ('Win', 'Overtake Crash', P_Win)
    ])
    # END_YOUR_CODE ########################################################
    return bayes_net

def find_best_overtake_condition(bayes_net):
    """
    Finds the optimal condition for overtaking the car, as described in Part 3
    Returns the optimal values for (MuchFaster,Early)
    """
    
    # BEGIN_YOUR_CODE ######################################################

    best_condition = None
    max_probability = -1
    
    # Conditions to check for MuchFaster and Early
    conditions = [(True, True), (True, False), (False, True), (False, False)]
    
    # Iterate over the conditions for MuchFaster and Early
    for muchfaster, early in conditions:
        # Define the evidence for the query
        evidenceWin = {'MuchFaster': muchfaster, 'Early': early, 'Crash': False}
        evidence = {'MuchFaster': muchfaster, 'Early': early}
        
        # Use elimination_ask to get the probability distribution for 'Overtake'
        WinNotCrash = elimination_ask('Win', evidenceWin, bayes_net).prob[True]

        NotCrash = elimination_ask('Crash', evidence, bayes_net).prob[False]

        final_prob = WinNotCrash * NotCrash
        
        print(f"P(Overtake=True | MuchFaster={muchfaster}, Early={early}) = {final_prob}")
        
        # Track the best condition (highest probability for 'Overtake' being True)
        if final_prob > max_probability:
            max_probability = final_prob
            best_condition = (muchfaster, early)

    return best_condition
    
    # END_YOUR_CODE ########################################################

def main():
    bayes_net = generate_bayesnet()
    cond = find_best_overtake_condition(bayes_net)
    print("Best overtaking condition: MuchFaster={}, Early={}".format(cond[0],cond[1]))

if __name__ == "__main__":
    main()

