import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import copy
import time
import traceback
import multiprocessing
from tqdm import tqdm
from functools import partial
try:
    import nn_009
    AI_AVAILABLE = True
except ImportError:
    print("Warning: nn_008 module not found. AI modes will be disabled.")
    AI_AVAILABLE = False    

def evaluation_func(board : list[list[int]]):
    num_used = 0
    for i in range(0, board_size):
        for j in range(0, board_size):
            if board[i][j] != 0:
                num_used += 1
    for i in range(0, board_size):
        for j in range(0, board_size):
            if board[i][j] != 0:
                for (x, y) in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    cnt = 0
                    for d in range(0, 5):
                        ni = i + d * x
                        nj = j + d * y
                        if 0 <= ni and ni < board_size and 0 <= nj and nj < board_size and board[i][j] == board[ni][nj]:
                            cnt += 1
                        else:
                            break
                    if cnt == 5:
                        if board[i][j] == 1:
                            return (1 - num_used * 3e-4)
                        else:
                            return -(1 - num_used * 3e-4)
    return 0

board_size = nn_009.board_size
class Config:
    num_workers=5
    num_games=5
    num_simulations=100
    name1 = "run_1000.pth"
    name2 = "run_1000.pth"
    Model1 = nn_009.Model(name1)
    Model2 = nn_009.Model(name2)

def play_single_game(_, Model1, Model2):
    board = [[0] * board_size for _ in range(board_size)]
    for i in range(0, board_size * board_size):
        if i % 2 == 0:
            x, y = Model1.call(board,simulations=Config.num_simulations)
        else:
            x, y = Model2.call(board,simulations=Config.num_simulations)
        board[x][y] = 1
        if evaluation_func(board) != 0:
            print("Length = ", i + 1, board)
            return 1 if i % 2 == 0 else -1
        for x in range(0, board_size):
            for y in range(0, board_size):
                board[x][y] *= -1
    return 0

def print_res(results, name1, name2):
    win, loss, draw = 0, 0, 0
    for v in results:
        if v == 1:
            win += 1
        if v == -1:
            loss += 1
        if v == 0:
            draw += 1
    print(f"{name1} Black vs {name2} White")
    print("Win percentage: ", win / len(results))
    print("Draw percentage: ", draw / len(results))
    print("Loss percentage: ", loss / len(results))
    print("=" * 66)

def run(Model1, Model2, name1, name2):
    with multiprocessing.get_context('spawn').Pool(
        processes=Config.num_workers
    ) as pool:
        func = partial(
            play_single_game,
            Model1=Model1,
            Model2=Model2
        )
        results = list(tqdm(pool.imap(func, range(Config.num_games)),total=Config.num_games, desc="Generating games"))
        print_res(results, name1, name2)

if __name__ == "__main__":
    run(Config.Model1, Config.Model2, Config.name1, Config.name2)
    run(Config.Model2, Config.Model1, Config.name2, Config.name1)