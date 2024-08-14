from __future__ import print_function, division
import csv
import json
import random

from cogworks.tetris.game import State, Board, zoids
from cogworks.tetris import simulator
from cogworks import feature


# Function to parse a Board from its JSON representation
def parse_board(rep):
    rows = len(rep)
    cols = None
    for row in rep:
        if cols is None:
            cols = len(row)
        else:
            assert cols == len(row)
    board = Board(rows, cols, zero=False)
    board.heights = [0] * cols
    for r in range(0, rows):
        for c in range(0, cols):
            board.data[r, c] = bool(rep[r][c])
            if not board.heights[c] and board.data[r, c]:
                board.heights[c] = r
    return board


# Function to get num_reps board representations from the given human data file
def get_reps(num_reps, filename, start_idx, ordered):
    reps = {}
    with open(filename, 'r') as file:
        reader = csv.DictReader(file, dialect='excel-tab')
        req_keys = ['end', 'curr_zoid', 'next_zoid', 'zoid_rot', 'zoid_row', 'zoid_col', 'board_rep']
        # get ordered or random set of representations from the file
        if ordered:
            for idx, line in enumerate(line for line in reader):
                # store lines from given index to needed number of representations
                if start_idx <= idx < num_reps + start_idx:
                    reps[idx] = {key: line[key] for key in req_keys}
        else:
            index_history = []
            for idx, line in enumerate(line for line in reader):
                chosen_idx = idx + random.randint(0, 499)
                if idx <= num_reps:
                    while (chosen_idx in index_history) and chosen_idx < 500:
                        chosen_idx = idx + random.randint(0, 499)
                    index_history.append(chosen_idx)
                    reps[chosen_idx] = {key: line[key] for key in req_keys}
    return reps


# Function to generate a list of possible immediate futures from a state, given a zoid and move_gen
def gen_futures(state, zoid, move_gen):
    return [state.future(zoid, *move) for move in move_gen(state, zoid)]


# Function to calculate the models' predictive power
def predict_move(feats, num_reps, idx, ordered):
    human_data = get_reps(num_reps, 'Collected Data/extreme_novice.tsv', idx, ordered)
    state = State(None, Board(20, 10))
    all_zoids = {
        'I': zoids.classic.I,
        'T': zoids.classic.T,
        'L': zoids.classic.L,
        'J': zoids.classic.J,
        'O': zoids.classic.O,
        'Z': zoids.classic.Z,
        'S': zoids.classic.S
    }
    predictive_score = 0
    line_num = 0
    for rep in human_data.values():
        line_num += 1
        zoid = all_zoids[rep['curr_zoid']]
        rot = int(rep['zoid_rot'])
        row = int(rep['zoid_row'])
        col = int(rep['zoid_col'])
        end = rep['end']

        # get the human board representation (with history)
        human_board = state.future(zoid, rot, 20 - row, col)
        # create a new board (without history) for model prediction
        model_board = parse_board(json.loads(rep['board_rep'][1:-1]))
        model_board.clear(model_board.full())

        # print('HUMAN BOARD:')
        # print(human_board.board)
        # print('MODEL BOARD:')
        # print(model_board)
        # print('=================================')

        # check if the boards are identical element-wise
        assert human_board.board == model_board

        # generate all possible states for the current zoid based on model features
        possible = gen_futures(state, zoid, simulator.move_drop)
        possible = [(sum(feature.evaluate(s, feats).values()), s) for s in possible]

        # order the moves in a given order
        possible.sort(key=lambda x: -x[0])
        scores, possible = zip(*possible)

        # if the predicted move is identical to the move made by the human, increment score
        if possible[0].board == human_board.board:
            predictive_score += 1

        # update state to maintain history if game has not ended
        if end == 'TRUE':
            state = State(None, Board(20, 10))
        else:
            state = human_board
    return (predictive_score / num_reps) * 100
