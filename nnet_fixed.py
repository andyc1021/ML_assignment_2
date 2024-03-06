import mlrose_hiive as mlrose
import numpy as np
import random
import pandas as pd

class CustomNeuralNetwork(mlrose.NeuralNetwork):
    def __init__(self, hidden_nodes=None,
                 activation='relu',
                 algorithm='random_hill_climb',
                 max_iters=100,
                 bias=True,
                 is_classifier=True,
                 learning_rate=0.1,
                 early_stopping=False,
                 clip_max=1e+10,
                 restarts=0,
                 schedule=mlrose.GeomDecay(),
                 pop_size=200,
                 mutation_prob=0.1,
                 max_attempts=10,
                 random_state=None,
                 curve=False):
        self.classes_ = [0, 1]
        super().__init__(
            hidden_nodes=hidden_nodes,
            activation=activation,
            algorithm=algorithm,
            max_iters=max_iters,
            bias=bias,
            is_classifier=is_classifier,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            clip_max=clip_max,
            restarts=restarts,
            schedule=schedule,
            pop_size=pop_size,
            mutation_prob=mutation_prob,
            max_attempts=max_attempts,
            random_state=random_state,
            curve=curve)