import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
import math
from functools import partial
import torch.nn.functional as F
import copy
import os
import time

board_size = 15
class Config:
    batch_size = 256
    num_epochs = 3
    learning_rate = 1e-4
    train_ratio = 0.9
    num_samples = 100
    channel = 32
    num_workers = 10
    train_simulation = 100
    base_path = None
    model_path = 'gomoku_cnn_test' # path of training checkpoints
    debug_on_play = False

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ValueCNN(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, num_blocks=5, value_dim = 128):
        super(ValueCNN, self).__init__()
        
        self.conv_init = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(hidden_channels)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])
        
        self.policy_conv1 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)
        self.policy_bn1 = nn.BatchNorm2d(hidden_channels // 2)
        self.policy_conv2 = nn.Conv2d(hidden_channels // 2, 1, kernel_size=3, padding=1)
        
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, value_dim)
        self.value_fc2 = nn.Linear(value_dim, 1)

    def forward(self, x):
        x = F.relu(self.bn_init(self.conv_init(x)))
        
        for block in self.res_blocks:
            x = block(x)
        
        policy = F.relu(self.policy_bn1(self.policy_conv1(x)))
        policy = self.policy_conv2(policy)
        policy = policy.squeeze(1)  
        policy_logits = policy.view(x.size(0), -1)
        
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(x.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)) 
        
        return value, policy_logits
    
    def calc(self, x):
        self.eval()
        with torch.no_grad():
            value, logits = self.forward(x)
            probs = F.softmax(logits, dim=1).view(-1, board_size, board_size)
            return value, probs

class GomokuDataset(Dataset):
    def __init__(self, boards, policies, values):
        self.boards = boards
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return self.boards[idx],self.policies[idx], self.values[idx]

def board_to_tensor(board : list[list[int]]):
    """
    将board_sizexboard_size的棋盘转换为3通道的tensor
    board: List[List[int]], 1代表当前方, -1代表对方, 0代表空
    返回: (3, board_size, board_size) tensor
    """
    board = np.array(list(board))
    
    # 创建3个通道
    current_player = (board == 1).astype(np.float32)  # 当前玩家的棋子
    opponent = (board == -1).astype(np.float32)       # 对手的棋子
    empty = (board == 0).astype(np.float32)           # 空位
    
    # 堆叠成3通道
    tensor = np.stack([current_player, opponent, empty], axis=0)
    return torch.FloatTensor(tensor)

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

def generate_random_board(model):
    """生成随机的棋盘状态"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    perm = []
    for i in range(0, board_size):
        for j in range(0, board_size):
            perm.append((i, j))
    random.shuffle(perm)

    best_val = 1e9
    best_board = [[0 for _ in range(board_size)] for _ in range(board_size)]
    if random.randint(0, 5) == 0:
        num_run = 0
    else:
        num_run = random.randint(50, 500)

    for t in range(0, num_run):
        num_moves = random.randint(0, 10)
        board = [[0 for _ in range(board_size)] for _ in range(board_size)]
        current_player = -1
        for _ in range(num_moves):
            # 随机选择一个空位下棋
            i, j = perm[_]
            board[i][j] = current_player
            current_player = -current_player
        if evaluation_func(board) != 0:
            continue
        #print(board)
        board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
        with torch.no_grad():
            value, policy = model.calc(board_tensor)
        #print("value = ", value)
        val = max(float(value), -float(value))
        if val < best_val:
            best_val = val
            best_board = board
    #   _val)
    return best_board

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.compute_value = None
        self.value = None

accumulate_sum = 0
class MCTS:
    def __init__(self, model, c_puct=0.8, parallel=0, use_rand=0.008):
        self.c_puct = c_puct
        self.use_rand = use_rand
        if Config.debug_on_play:
            self.use_rand = 0
        #self.device = 'cpu'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

    def board_to_key(self, board):
        return str(board)  # 或使用更高效的哈希方法

    def no_child(self, board):
        for i in range(0, board_size):
            for j in range(0, board_size):
                if board[i][j] == 0:
                    return False
        return True

    def is_terminal(self, board):
        if self.no_child(board):
            return True
        return evaluation_func(board) != 0

    def run(self, root_board, num_simulations):
        global accumulate_sum
        accumulate_sum = 0
        all_beg = time.time_ns()
        root = MCTSNode(root_board)
        
        for _ in range(num_simulations):
            node = root
            search_path = [node]
            
            while node.children:
                node = self.select_child(node)
                search_path.append(node)
            
            if not self.is_terminal(node.board):
                self.expand_node(node)
            
            value = self.evaluate_node(node)
            for node in reversed(search_path):
                node.visit_count += 1
                node.value_sum += value
                value = -value
            #print(len(search_path), end=" ")
        
        #print("sum = ", accumulate_sum, ", total = ", time.time_ns() - all_beg, ", ratio = ", accumulate_sum / (time.time_ns() - all_beg))
        return self.get_results(root)
    
    def select_child(self, node : MCTSNode):
        global accumulate_sum
        beg = time.time_ns()
        total_visits = sum((child.visit_count if child != None else 0) for child, prior in node.children.values())
        sqrt_total_visits = math.sqrt(total_visits + 1)
        
        best_score = -1e9
        best_move = None
        
        exp1 = 0
        exp2 = 0
        for child, prior in node.children.values():
            if child != None:
                exp1 += child.value_sum
                exp2 += child.visit_count
        ave = exp1 / (exp2 + 1e-5)

        #print(len(node.children.items()))

        tmp = 0
        for move, (child, prior) in node.children.items():
            explore = self.c_puct * prior * sqrt_total_visits
            exploit = ave
            if child != None and child.visit_count != 0:
                exploit = child.value_sum / child.visit_count
                explore /= (child.visit_count + 1)
            score = explore - exploit
            if score > best_score:
                best_score = score
                best_move = move
        accumulate_sum += time.time_ns() - beg
        #print(type(best_score))
        
        
        
        chd, pri = node.children[best_move]
        if chd == None:
            i, j = best_move
            new_board = copy.deepcopy(node.board)
            new_board[i][j] = 1 
            for x in range(board_size):
                for y in range(board_size):
                    new_board[x][y] *= -1
            chd = MCTSNode(new_board, parent=node, move=best_move)
            node.children[best_move] = chd, pri

        return chd
    
    def expand_node(self, node : MCTSNode):
        global accumulate_sum
        tm_beg = time.time_ns()
        board_tensor = board_to_tensor(node.board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value, policy = self.model.calc(board_tensor)
        
        node.value = float(value)
        policy = policy.squeeze(0).cpu().numpy().tolist()

        sum_1 = 0
        for i in range(board_size):
            for j in range(board_size):
                if node.board[i][j] == 0:
                    sum_1 += policy[i][j]
        if sum_1 == 0:
            sum_1 += 1e-10
        for i in range(board_size):
            for j in range(board_size):
                if node.board[i][j] == 0:  
                    node.children[(i, j)] = None, policy[i][j] / sum_1 + random.normalvariate(mu=0, sigma=self.use_rand)
        accumulate_sum += time.time_ns() - tm_beg
    
    def evaluate_board(self, board):
        global accumulate_sum
        if self.no_child(board):
            return 0
        eval = evaluation_func(board)
        if eval != 0:
            return eval
        board_tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
        tm_beg = time.time_ns()
        with torch.no_grad():
            value, _ = self.model.calc(board_tensor)
        accumulate_sum += time.time_ns() - tm_beg
        return value.item()

    def evaluate_node(self, node : MCTSNode):
        if node.compute_value != None:
            return node.compute_value
        if self.no_child(node.board):
            return 0
        eval = evaluation_func(node.board)
        if eval != 0:
            return eval
        if node.value == None:
            assert False
        return node.value
    def get_results(self, root):
        probs = np.zeros((board_size, board_size))
        total_visits = sum((child.visit_count if child != None else 0) for child, prior in root.children.values())
        
        for move, (child, prior) in root.children.items():
            if child != None:
                i, j = move
                probs[i, j] = child.visit_count / total_visits if total_visits > 0 else 0        
        return root.value_sum / root.visit_count, probs

def find_board(model):
    return 0

def augment_data(boards, policies, values):
    augmented_boards = []
    augmented_policies = []
    augmented_values = []
    
    for board, policy, value in zip(boards, policies, values):
        value = torch.tensor(value).clone().detach().float()
        
        # D4 对称变换
        for k in range(4):
            # 旋转 k * 90°
            rotated_board = torch.rot90(board, k, [1, 2])
            rotated_policy = torch.rot90(policy, k, [0, 1])
            augmented_boards.append(rotated_board)
            augmented_policies.append(rotated_policy)
            augmented_values.append(value)
            
            # 水平翻转后再旋转 k * 90°
            flipped_board = torch.flip(board, [2])
            flipped_policy = torch.flip(policy, [1])
            rotated_flipped_board = torch.rot90(flipped_board, k, [1, 2])
            rotated_flipped_policy = torch.rot90(flipped_policy, k, [0, 1])
            augmented_boards.append(rotated_flipped_board)
            augmented_policies.append(rotated_flipped_policy)
            augmented_values.append(value)
    
    return (
        torch.stack(augmented_boards),
        torch.stack(augmented_policies),
        torch.stack(augmented_values)
    )

def generate_selfplay_data(model, num_games, num_simulations=Config.train_simulation):
    # 使用多进程并行生成游戏
    
    model_state_dict = model.state_dict()
    with multiprocessing.get_context('spawn').Pool(
        processes=Config.num_workers
    ) as pool:
        func = partial(
            generate_single_game,
            model_state_dict=model_state_dict,
            num_simulations=num_simulations
        )
        results = list(tqdm(
            pool.imap(func, range(num_games)),
            total=num_games,
            desc="Generating games"
        ))

    
    # 整合结果
    boards, policies, values = [], [], []
    for game_boards, game_policies, game_values in results:
        boards.extend(game_boards)
        policies.extend(game_policies)
        values.extend(game_values)
    
    boards = torch.stack(boards)
    policies = torch.stack(policies)
    values = torch.FloatTensor(values)
    
    # 数据增强：旋转和翻转
    print("len_boards_raw = ", len(boards))
    boards, policies, values = augment_data(boards, policies, values)
    
    return boards, policies, values

def calc_next_move(board, probs, temperature=0):
    valid_moves = []
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == 0:
                valid_moves.append((i, j, probs[i][j]))
    if temperature == 0:
        valid_moves.sort(key=lambda x: x[2], reverse=True)
        return valid_moves[0][:2]
    else:
        moves = [(i, j) for i, j, _ in valid_moves]
        probs = np.array([p for _, _, p in valid_moves], dtype=np.float64)
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        probs = probs / probs.sum()
        chosen_index = np.random.choice(len(moves), p=probs)
        return moves[chosen_index]  


def generate_single_game(_, model_state_dict, num_simulations):
    """生成单局游戏数据，使用传入的模型状态字典在CPU上构建模型"""
    # 在子进程中创建新模型（避免GPU冲突）
    """生成单局游戏数据"""
    game_boards = []
    game_policies = []
    game_values = []
    #print("Start!")
    
    model = ValueCNN()
    model.load_state_dict(model_state_dict)
    model.eval()  # 设置为评估模式
    board = generate_random_board(model)
    temperature=0.1*random.randint(0,10)
    with torch.no_grad():
        mcts = MCTS(model)
        
        game_values = []
        for move_num in range(board_size*board_size):
            # 保存当前状态
            game_boards.append(board_to_tensor(copy.deepcopy(board)))
            
            # MCTS获取策略
            value, action_probs = mcts.run(board, num_simulations)
            game_policies.append(torch.FloatTensor(action_probs))
            #print(action_probs)
            game_values.append(value)
            
            # 选择动作
            
            action = calc_next_move(board, action_probs, temperature)
            
            # 执行动作
            board[action[0]][action[1]] = 1

            # 检查游戏结束
            if mcts.is_terminal(board):
                break
            
            # 翻转视角
            for i in range(board_size):
                for j in range(board_size):
                    board[i][j] *= -1
            #print(action[0], action[1])
        #print(len(game_values), game_values[len(game_values) - 1])
    
    print(len(game_boards))
    return game_boards, game_policies, game_values

def debug_play(model_path, num_simulations=1000):
    
    model = ValueCNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    with torch.no_grad():
        board = [[0]*board_size for _ in range(board_size)]
        mcts = MCTS(model)
        
        for move_num in range(board_size*board_size):
            value, action_probs = mcts.run(board, num_simulations)
            mcts.print_values(board)
            
            # 选择动作
            action = np.unravel_index(
                np.random.choice(np.arange(board_size*board_size), p=action_probs.flatten()),
                (board_size, board_size)
            )
            
            # 执行动作
            board[action[0]][action[1]] = 1

            if Config.debug_on_play:
                for i in range(0, board_size):
                    st = "|"
                    for j in range(0, board_size):
                        h = board[i][j]
                        if i == action[0] and j == action[1]:
                            if (move_num & 1 == 1):
                                st += 'X'
                            else:
                                st += 'O'
                        else:
                            if h == 1:
                                st += 'o' if (move_num & 1 == 0) else 'x'
                            elif h == 0:
                                st += ' '
                            else:
                                st += 'x' if (move_num & 1 == 0) else 'o'
                    st += '|'
                    print(st)
                print('\n')
            
            # 检查游戏结束
            if mcts.is_terminal(board):
                final_value = mcts.evaluate_board(board)
                break
            
            # 翻转视角
            for i in range(board_size):
                for j in range(board_size):
                    board[i][j] *= -1
    
    return

#def selfplay(model):
#    board = [[0 for _ in range(0, board_size)] for _ in range(0, board_size)]
#    moves = []
#    values = []
#    boards = []
#    policies = []
#    #print("ORG_BOARD = ", board)
#    num = 0
#    while len(moves) < board_size * board_size:
#        num += 1
#        boards.append(board)
#        val, policy, mv = predict_next(model, copy.deepcopy(board))
#        (x, y) = mv
#        values.append(val)
#        policies.append(policy)
#        moves.append(mv)
#        #print(board[x][y], x, y)
#        board[x][y] = 1
#        for i in range(0, board_size):
#            for j in range(0, board_size):
#                board[i][j] *= -1 
#        #print("BOARD: ", board)
#        #print(board)
#        for i in range(0, board_size):
#            st = "|"
#            for j in range(0, board_size):
#                h = board[i][j]
#                if i == x and j == y:
#                    if (num & 1 == 0):
#                        st += 'X'
#                    else:
#                        st += 'O'
#                else:
#                    if h == 1:
#                        st += 'o' if (num & 1 == 0) else 'x'
#                    elif h == 0:
#                        st += ' '
#                    else:
#                        st += 'x' if (num & 1 == 0) else 'o'
#            st += '|'
#            print(st)
#        print('\n')
#        if evaluation_func(board) != 0:
#            break
#    #print("moves = ", moves)
#    return boards, policies, values, moves

class Model:
    def __init__(self, location, use_rand=0.01,simulations=200, c_puct=0.8):
        self.simulations=simulations
        self.model = ValueCNN()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(location,map_location=torch.device(self.device),weights_only=True))
        self.mcts = MCTS(self.model,use_rand=use_rand,c_puct=c_puct)
    def call(self, board, temperature=0, simulations=-1,debug=0):
        if simulations == -1:
            simulations = self.simulations
        _, action_probs = self.mcts.run(copy.deepcopy(board), simulations)
        if debug != 0:
            print("expected_value = ", _)
            print("board_prob = ")
            for i in range(0, board_size):
                for j in range(0, board_size):
                    if board[i][j] == 1:
                        print("  o ", end=" ")
                    elif board[i][j] == -1:
                        print("  x ", end=" ")
                    else:
                        print(f"{action_probs[i][j]:.2f}", end=" ")
                print('\n')

        return calc_next_move(board, action_probs, temperature)



def train_model(model, train_loader, val_loader, config):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    value_criterion = nn.MSELoss()
    policy_criterion = nn.KLDivLoss(reduction='batchmean')
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5)
    
    best_val_loss = float('inf')

    
    train_losses = []
    val_losses = []
    
    print(f"开始训练，使用设备: {device}")
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        train_value_loss, train_policy_loss = 0, 0
        
        for batch_boards, batch_policies, batch_values in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}'):
            batch_boards = batch_boards.to(device)
            batch_policies = batch_policies.to(device).view(batch_policies.size(0), -1)
            batch_values = batch_values.to(device).unsqueeze(1)  # 添加维度匹配
            
            optimizer.zero_grad()
            
            pred_values, pred_policies = model(batch_boards)
            
            # 计算损失
            value_loss = value_criterion(pred_values, batch_values)
            policy_loss = policy_criterion(
                F.log_softmax(pred_policies, dim=1),
                batch_policies.view(-1, batch_policies.size(-1))
            )

            loss = 3 * value_loss + policy_loss
            
            loss.backward()
            optimizer.step()
            
            train_value_loss += value_loss.item()
            train_policy_loss += policy_loss.item()

        
        model.eval()
        val_value_loss, val_policy_loss = 0, 0
        with torch.no_grad():
            for boards, policies, values in val_loader:
                boards = boards.to(device)
                policies = policies.to(device).view(policies.size(0), -1)
                values = values.to(device).unsqueeze(1)
                
                pred_values, pred_policies = model(boards)
                val_value_loss += value_criterion(pred_values, values).item()
                val_policy_loss += policy_criterion(
                    F.log_softmax(pred_policies, dim=1),
                    policies.view(-1, policies.size(-1))  # 修改：使用正确的变量名和动态尺寸
                ).item()
        
        # 计算平均损失
        avg_train_value = train_value_loss / len(train_loader)
        avg_train_policy = train_policy_loss / len(train_loader)
        avg_val_value = val_value_loss / len(val_loader)
        avg_val_policy = val_policy_loss / len(val_loader)

        print("Train: ", train_value_loss, train_policy_loss)
        print("Val: ", val_value_loss, val_policy_loss)
        
        # 学习率调度
        val_total_loss = 4 * avg_val_value + avg_val_policy
        scheduler.step(val_total_loss)

        train_losses.append(avg_train_value + avg_train_policy)
        val_losses.append(avg_val_value + avg_val_policy)
        
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            #torch.save(model.state_dict(), config.model_path)
    
    return train_losses, val_losses


def plot_training_history(train_losses, val_losses):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

# 主训练函数
def work():
    """
    主训练函数
    evaluation_func: 你的评估函数 f: List[List[int]] -> float
    """
    config = Config()
    model = ValueCNN()
    if config.base_path != None:
        model.load_state_dict(torch.load(config.base_path, weights_only=True))
    
    for t in range(0, 2000):
        model.eval()
        print("Working on training step ", t)
        # 生成训练数据
        boards, policies, values = generate_selfplay_data(model, config.num_samples)
        
        # 划分训练集和验证集
        num_train = int(len(boards) * config.train_ratio)
        train_boards = boards[:num_train]
        train_policies = policies[:num_train]
        train_values = values[:num_train]
        val_boards = boards[num_train:]
        val_policies = policies[num_train:]
        val_values = values[num_train:]
        
        # 创建数据集和数据加载器
        train_dataset = GomokuDataset(train_boards, train_policies, train_values)
        val_dataset = GomokuDataset(val_boards, val_policies, val_values)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        # 训练模型
        train_losses, val_losses = train_model(model, train_loader, val_loader, config)
        
        # 绘制训练历史
        #plot_training_history(train_losses, val_losses)
        print(train_losses)
        print(val_losses)
        
        
        # 保存模型
        os.makedirs(config.model_path, exist_ok=True) 
        checkpoint = os.path.join(config.model_path, f"{t + 1}.pth")
        print(checkpoint)
        torch.save(model.state_dict(), checkpoint)
        print(f"模型已保存为 {checkpoint}")
    
    return model

# 使用示例
if __name__ == "__main__":
    if Config.debug_on_play:
        debug_play(Config.base_path)
    else:
        work()
