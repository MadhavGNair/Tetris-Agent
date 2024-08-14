### Starting Simulator

## Generic imports
import random
import numpy

## From Simulator Specific Code
from cogworks.tetris.game import State
from cogworks.tetris.game import Board
from cogworks.tetris.game import zoids

from cogworks.tetris import features
from cogworks.tetris import simulator

from cogworks import feature

# change these values to match the output of LearnTest.py for different hyperparameters
print('running...')

testfeatures = {
    features.landing_height: -3.002,
    features.eroded_cells: -13.611,
    features.row_trans: -6.217,
    features.col_trans: -11.646,
    features.pits: -22.990,
    features.cuml_wells: -0.222}


## Create a piece generator (can use a list here, or a generator function)

# True Random Generator Function
def zoid_gen(zoids, rng):
    while True:
        yield rng.choice(zoids)


# # Create a move generator (this is where overhang detection or time pressure filtering are implemented. move_drop
# is basic version)

move_gen = simulator.move_drop

## Define Features

feats = testfeatures

## Pick a seed (if using)
# seeds = 101

## Create simulator "object"
scores = []
one_lines = []
two_lines = []
three_lines = []
four_lines = []

# Loop through seeds to average out at the end
for seed in range(100, 10001, 100):
    state = State(None, Board(20, 10))
    sim = simulator.simulate(state, zoid_gen(zoids.classic, random.Random(seed)), move_gen,
                             simulator.policy_best(lambda state: sum(feature.evaluate(state, feats).values()),
                                                   random.Random(-seed).choice))

    ## Run the simulator
    # Inside this loop, print episode level data to file (take a good look at state and delta classes in game.py)

    for (episode, state) in enumerate(sim, 1):
        # print(episode, state.delta.zoid)
        # print(state.board)
        # print('======================================')
        # print('======================================')

        # set some kind of limit
        if episode >= 525:
            break

    if state.lines_cleared() != 0:
        scores.append(state.score())
        one_lines.append((state.lines_cleared(1)/state.lines_cleared())*100)
        two_lines.append((state.lines_cleared(2)/state.lines_cleared())*100)
        three_lines.append((state.lines_cleared(3)/state.lines_cleared())*100)
        four_lines.append((state.lines_cleared(4)/state.lines_cleared())*100)
    else:
        scores.append(state.score())
        one_lines.append(0)
        two_lines.append(0)
        three_lines.append(0)
        four_lines.append(0)

print('STATS:')
print('Total scores:')
# print(scores)
print('Mean = ', numpy.mean(scores))
print('Std = ', numpy.std(scores))
print('One line clears:')
# print(one_lines)
print('Mean = ', numpy.mean(one_lines))
print('Std = ', numpy.std(one_lines))
print('Two line clears:')
# print(two_lines)
print('Mean = ', numpy.mean(two_lines))
print('Std = ', numpy.std(two_lines))
print('Three line clears:')
# print(three_lines)
print('Mean = ', numpy.mean(three_lines))
print('Std = ', numpy.std(three_lines))
print('Four line clears (Tetris):')
# print(four_lines)
print('Mean = ', numpy.mean(four_lines))
print('Std = ', numpy.std(four_lines))
