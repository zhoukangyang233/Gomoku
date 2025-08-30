### zky's Gomoku AI

#### Chinese description

这个项目提供了一个五子棋 AI 的训练脚本，算法为 AlphaZero 所用的 MCTS+CNN。

run_10_2000.pth 中的文件是使用 nn_012.py 在 4090 上训练了一到两天的 AI。

gmk_run_0629.py 提供了一个图形化界面，可以很方便地研究这个 AI 的棋谱。

gmk_pk_0619 提供了对两个 AI 从初始盘面开始对弈的胜率评估程序。

#### 英文描述

This project provides a training script for a Gomoku AI, using the MCTS+CNN algorithm employed in AlphaZero.

The file run_10_2000.pth contains an AI trained with nn_012.py on an RTX 4090 GPU for one to two days.

The script gmk_run_0629.py offers a graphical interface, making it convenient to study the AI’s game records.

The program gmk_pk_0619 provides a tool to evaluate the win rate when two AIs play against each other starting from the initial board.
