## Generic imports
import random

## From Simulator Specific Code
from cogworks.tetris.game import State
from cogworks.tetris.game import Board
from cogworks.tetris.game import zoids

from cogworks.tetris import features
from cogworks.tetris import simulator

from cogworks import feature
from cogworks import learning

import HumanTest as human_tester

seed = 0

print("running...")


def zoid_gen(zoids, rng):
    while True:
        yield rng.choice(zoids)


move_gen = simulator.move_drop


def test(feats):
    # if printing game information to a file, do it here
    state = State(None, Board(20, 10))
    sim = simulator.simulate(state, zoid_gen(zoids.classic, random.Random(seed)), move_gen,
                             simulator.policy_best(lambda state: sum(feature.evaluate(state, feats).values()),
                                                   random.Random(-seed).choice), lookahead=1)

    for (episode, state) in enumerate(sim, 1):
        # some end condition
        if episode >= 525:
            break
    # if no end condition, [pass] inside for loop
    # return whatever value determines goodness, scores, lines cleared, etc
    # print(state.score)
    return state.score()


generation_num = 0


def human_test(feats, idx):
    pred_score = human_tester.predict_move(feats, 498, 0, True)
    print("score at generation ", idx, " is ", pred_score)
    return pred_score


# if starting at all zeroes, put features in a list, if features have actual values, define as a dictionary

feats = [features.landing_height,
         features.eroded_cells,
         features.row_trans,
         features.col_trans,
         features.pits,
         features.cuml_wells]

# # cross entropy takes [features][standard deviation][#models per generation][#kept per generation][rng that takes
# seed][test function defined above]
trainer = learning.cross_entropy(feats, 10, 100, 10, random.Random(seed), human_test)

iter_num = 0

for (depth, (weights, stdev)) in enumerate(trainer, 1):
    iter_num += 1
    print('============= ITERATION ', iter_num, ' =============')
    seed += 1

    stable = True
    print('Iteration {:6d}: {:>9} {:>9}'.format(depth, 'mean', 'variation'))

    for feat in feats:
        var = abs(stdev[feat] / weights[feat])
        print('{:16}: {:> 9.3f} {:> 9.3}'.format(feat.__name__, weights[feat], var))
        if var > 0.01:
            stable = False

    if stable:
        break
