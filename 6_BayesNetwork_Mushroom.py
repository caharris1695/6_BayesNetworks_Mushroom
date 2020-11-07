# Dataset Name: Mushroom Dataset
# Dataset Location: https://archive.ics.uci.edu/ml/datasets/Mushroom
# Paper Read: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.103.2349&rep=rep1&type=pdf
# Date Started: 10_27_2020
# Algorithm: Bayesian Networks

# Attribute Information:
# 1. cap_shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# 2. cap_surface: fibrous=f,grooves=g,scaly=y,smooth=s
# 3. cap_color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
# 4. bruises?: bruises=t,no=f
# 5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
# 6. gill_attachment: attached=a,descending=d,free=f,notched=n
# 7. gill_spacing: close=c,crowded=w,distant=d
# 8. gill_size: broad=b,narrow=n
# 9. gill_color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
# 10. stalk_shape: enlarging=e,tapering=t
# 11. stalk_root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
# 12. stalk_surface_above_ring: fibrous=f,scaly=y,silky=k,smooth=s
# 13. stalk_surface_below_ring: fibrous=f,scaly=y,silky=k,smooth=s
# 14. stalk_color_above_ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
# 15. stalk_color_below_ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
# 16. veil_type: partial=p,universal=u
# 17. veil_color: brown=n,orange=o,white=w,yellow=y
# 18. ring_number: none=n,one=o,two=t
# 19. ring_type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
# 20. spore_print_color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
# 21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
# 22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d

# REFERENCE: https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/estimators/HillClimbSearch.py

# Imports for Probablistic graphical models and Bayesian Networks
import networkx as nx
from itertools import permutations
from networkx.drawing.nx_agraph import graphviz_layout
import pylab
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator, ParameterEstimator
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import PC, HillClimbSearch
from pgmpy.inference import VariableElimination
#!/usr/bin/env python
from itertools import permutations
from collections import deque

# How the progress bar is done
from tqdm import trange

# Imports from the referenced script
from pgmpy.estimators import (
    StructureScore,
    StructureEstimator,
    K2Score,
    ScoreCache,
    BDeuScore,
    BicScore,
)
from pgmpy.base import DAG
from pgmpy.global_vars import SHOW_PROGRESS

# Standard Machine Learning Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

# Dataset Columns
# Target Column is Poisenous(p) vs Edible (e) which is what we want to predict
columns = ['target','cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', \
           'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', \
           'stalk_surface_below_ring', 'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', \
           'ring_type', 'spore_print_color', 'population', 'habitat']

# Import data
df = pd.read_csv('agaricus-lepiota.data', header=0, names=columns)

# Split the Data into train and test sets
from sklearn.model_selection import train_test_split
targets = df['target']
features = df.drop(['target'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)

# For Bayesian Networks the training data has both features and targets in the Data
# Used by the HillClimbing Algoritm
train_data = pd.concat([X_train,y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
# Hill Climbing is used to get the structure of the bayesian network ...
# For the most part, this is a learning exercise to learn how to implement an algorithm
# Based off a template to learn more about Probablistic Graphical Models
# CODE: https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/estimators/HillClimbSearch.py
class HillClimb(StructureEstimator):

    def __init__(self, data):
        '''
        Class for heuristic hill climb searches for DAGs, to learn
        network structure from data. `estimate` attempts to find a model with optimal score.
        Parameters
        ----------
        data: pandas DataFrame object
        datafame object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.NaN`.
        Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)
        '''

        super(HillClimb, self).__init__(data)


    def _legal_operations(self, model, score, tabu_list, max_indegree, black_list):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Friedman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered. A list of
        edges can optionally be passed as `black_list` or `white_list` to exclude those
        edges or to limit the search.
        """
        tabu_list = set(tabu_list)

        # Step 1: Get all legal operations for adding edges.
        # Legal moves are: Non Cyclical (nx.has_path checks), not reversed edges,
        # and not the current edges
        potential_new_edges = (
            set(permutations(self.variables, 2))
            - set(model.edges())
            - set([(Y, X) for (X, Y) in model.edges()])
        )
        for (X, Y) in potential_new_edges:
            # Check if adding (X, Y) will create a cycle.
            if not nx.has_path(model, Y, X):
                operation = ("+", (X, Y))
                if ((operation not in tabu_list)
                    and ((X, Y) not in black_list)
                ):
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if len(new_parents) <= max_indegree:
                        score_delta = score(variable=Y, parents=new_parents) - score(variable=Y, parents=old_parents)
                        yield (operation, score_delta)

        # Step 2: Get all legal operations for removing edges
        # Removing an edge has only to do with the current model edges 
        for (X, Y) in model.edges():
            operation = ("-", (X, Y))
            if (operation not in tabu_list):
                old_parents = model.get_parents(Y)
                new_parents = old_parents[:]
                new_parents.remove(X)
                score_delta = score(Y, new_parents) - score(Y, old_parents)
                yield (operation, score_delta)

        # Step 3: Get all legal operations for flipping edges
        for (X, Y) in model.edges():
            # Check if flipping creates any cycles
            if not any(
                map(lambda path: len(path) > 2, nx.all_simple_paths(model, X, Y))
            ):
                operation = ("flip", (X, Y))
                if (
                    ((operation not in tabu_list) and ("flip", (Y, X)) not in tabu_list)
                    and ((Y, X) not in black_list)
                ):
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = old_Y_parents[:]
                    new_Y_parents.remove(X)
                    if len(new_X_parents) <= max_indegree:
                        score_delta = (
                            score(X, new_X_parents)
                            + score(Y, new_Y_parents)
                            - score(X, old_X_parents)
                            - score(Y, old_Y_parents)
                        )
                        yield (operation, score_delta)

    def estimate(self, tabu_length=100, max_indegree=2,black_list=None,
                 epsilon=1e-4, max_iter=1e6, show_progress=True):

        # We will be using K2Score for this model
        score = K2Score(data=self.data)
        # Model gets the score for a node and its parents
        # This is used on every iteration for all possible changes
        # This is greddy and picks the best available option
        score_fn = score.local_score
        # Initialize a Starting DAG
        # PGMPY made a DAG class that adds some functionality to nx.DiGrpah
        start_dag = DAG()
        start_dag.add_nodes_from(self.variables)
        # Set the edges we do not want to have in the graph
        if black_list is None:
            black_list=set()
        else:
            black_list=set(black_list)

        # Just change Maxindegree to a certain number when doing the model

        # I think this is to keep track of the changes we already made to the model
        tabu_list = deque(maxlen=tabu_length)
        # Initialize a current model
        current_model = start_dag
        if show_progress:
            iteration = trange(int(max_iter))
        else:
            iteration = range(int(max_iter))
        for _ in iteration:
            # Get the best operations based on K2 score with self._legal_operations
            best_operation, best_score_change = max(
                self._legal_operations(
                    model = current_model,
                    score = score_fn,
                    tabu_list = tabu_list,
                    max_indegree = max_indegree,
                    black_list = black_list,
                ),
                key=lambda t: t[1]
            )

            if best_score_change < epsilon:
                break
            elif best_operation[0] == '+':
                current_model.add_edge(*best_operation[1])
                tabu_list.append(("-", best_operation[1]))
            elif best_operation[0] == '-':
                current_model.remove_edge(*best_operation[1])
                tabu_list.append(("+", best_operation[1]))
            elif best_operation[0] == 'flip':
                X, Y = best_operation[1]
                current_model.remove_edge(X,Y)
                current_model.add_edge(Y,X)
                tabu_list.append(best_operation)

        return current_model




# Implement Bayesian Structure learning (Hill Climb Search) on training data
model = HillClimb(train_data)
blacklisted = [("target",i) for i in df.columns if i != 'target']
estimated_model = model.estimate(show_progress=True)

# Initialize a Bayesian Model with the edges from structure learning
bayes_model = BayesianModel(ebunch = estimated_model.edges())
# Graph the Bayesian Model what was learned
nx.draw(bayes_model, with_labels=True)
pylab.show()
# The Node for Veil Type with dropped from the model during learning
bayes_model.add_node('veil_type')
# Parameter Estimation
bayes_model.fit(train_data, estimator=BayesianEstimator,prior_type='Bdeu')

# Make an inference class to make predictions
infer = VariableElimination(bayes_model)

# Organize the test data to make predictions
evidence = []

for row in test_data.iterrows():
    blank_row = {i:None for i in df.columns if i != 'target'}
    for cls in blank_row:
        blank_row[cls] = row[1].to_dict()[cls]
    evidence.append(blank_row)

# Initialize a list for the predictions we will be making
predictions = []

# Progress counter
c = 0

# Loop through the test example evidance to make predictions
for ev, y_true in zip(evidence, y_test):
    c+=1
    if c%100 == 0:
        print("Iteration: ", c)
    # Get condtional probabiliy of each class given the evidence
    c_distribution = infer.query(['target'], evidence=ev, show_progress=False)
    # Make a prediciton
    pred = c_distribution.state_names['target'][np.argmax(c_distribution.values)]
    # Add the predictions to all the predictions
    predictions.append(pred)

# Get the accuracy of the model
accuracy = np.sum(np.array(predictions) == y_test.values) / len(predictions)
print("Accuracy: ",  accuracy)

# 99% Accuracy
