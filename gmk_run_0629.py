import asyncio
import platform
import pygame
import numpy as np
import copy
import math

import nn_012 as gomoku_cnn
from nn_012 import MCTS

class Config:
    ai_model = 'run4_1258.pth'
    ai_simulation = 200
    simulation_update = 10
    show_shape = 'circle' # square or circle
    show_type = 'colorful' # colorful / red / green
    show_nn = True

BACKGROUND = (220, 179, 92)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class GomokuGUI:
    def __init__(self, board_size=15):
        pygame.init()
        self.board_size = board_size
        self.cell_size = 40
        self.margin = 20
        self.width = self.height = self.margin * 2 + self.cell_size * (self.board_size - 1)
        self.screen = pygame.display.set_mode((self.width, self.width + 100))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1
        self.move_history = []
        self.history_index = -1

        self.mcts = MCTS(model=Config.ai_model,use_rand=0)
        self.root = None

        self.enable_research = False
        self.show_nn = False
        self.show_nn_prob = None
        self.show_nn_val = None
        self.autoplay = False

        self.buttons = {
            'research': pygame.Rect(10, self.width + 10, 150, 30),
            'autoplay': pygame.Rect(170, self.width + 10, 150, 30),
            'play': pygame.Rect(330, self.width + 10, 150, 30),
            'back': pygame.Rect(10, self.width + 50, 150, 30),
            'forward': pygame.Rect(170, self.width + 50, 150, 30)
        }
        print(type(self.buttons))
        print(self.buttons)
        if Config.show_nn:
            self.buttons['show nn'] = pygame.Rect(330, self.width + 50, 150, 30)


    def draw_board(self):
        self.screen.fill(BACKGROUND)

        if self.board_size == 11:
            star_points = [2, 5, 8]  # 适合11x11棋盘的星位
        elif self.board_size == 15:
            star_points = [3, 7, 11]  # 适合15x15棋盘的星位
        else:
            star_points = []
        for i in star_points:
            for j in star_points:
                if i < self.board_size and j < self.board_size:
                    pygame.draw.circle(self.screen, BLACK, (self.margin + j * self.cell_size, self.margin + i * self.cell_size), 3)
        
        for i in range(self.board_size):
            start = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, BLACK, (self.margin, start), (self.width - self.margin, start), 2)
            pygame.draw.line(self.screen, BLACK, (start, self.margin), (start, self.width - self.margin), 2)

        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 1:
                    pygame.draw.circle(self.screen, BLACK, (self.margin + j * self.cell_size, self.margin + i * self.cell_size), self.cell_size // 2 - 2)
                elif self.board[i][j] == -1:
                    pygame.draw.circle(self.screen, WHITE, (self.margin + j * self.cell_size, self.margin + i * self.cell_size), self.cell_size // 2 - 2)
        if self.history_index != -1:
            (i, j), player = self.move_history[self.history_index]
            pygame.draw.circle(self.screen, (255, 0, 0), 
                              (self.margin + j * self.cell_size, self.margin + i * self.cell_size), 5, 2)


    def draw_heatmap(self, visit_matrix, value_matrix):
        max_visit = np.max(visit_matrix)
        if max_visit == 0:
            return
        sum_value, sum_cnt = 0, 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if visit_matrix[i][j] > 0:
                    sum_value += visit_matrix[i][j] * visit_matrix[i][j] * value_matrix[i][j]
                    sum_cnt += visit_matrix[i][j] * visit_matrix[i][j]
        mean_value = sum_value / sum_cnt
        for i in range(self.board_size):
            for j in range(self.board_size):
                if visit_matrix[i][j] > 0:
                    alpha = int(255 * math.pow(visit_matrix[i][j] / max_visit, 0.75))
                    s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)

                    diff = max(min(int((mean_value - value_matrix[i][j]) * 3 * 255), 255), 0)
                    if Config.show_type == 'colorful':
                        color = (diff, 255 - diff, 0, alpha)
                    elif Config.show_type == 'red':
                        color = (255, 0, 0, alpha)
                    else:
                        color = (0, 255, 0, alpha)

                    if Config.show_shape == 'circle':
                        radius = int(self.cell_size // 2) - 2
                        pygame.draw.circle(s, color, (s.get_width() // 2, s.get_height() // 2), radius)
                        self.screen.blit(s, (self.margin + j * self.cell_size - self.cell_size // 2, self.margin + i * self.cell_size - self.cell_size // 2))
                    else:
                        s.fill(color)
                        self.screen.blit(s, (self.margin + j * self.cell_size - self.cell_size // 2, self.margin + i * self.cell_size - self.cell_size // 2))

    def get_grid_pos(self, pos):
        x, y = pos
        i = round((y - self.margin) / self.cell_size)
        j = round((x - self.margin) / self.cell_size)
        if 0 <= i < self.board_size and 0 <= j < self.board_size:
            return (i, j)
        return None

    def draw_hover_info(self, mouse_pos, visit_matrix, value_matrix, str1 = "Visit", str2 = "Value"):
        grid_pos = self.get_grid_pos(mouse_pos)
        if grid_pos:
            i, j = grid_pos
            if visit_matrix[i][j] > 0:
                text = []
                #[f"{str1}: {visit_matrix[i][j]}", f"{str2}: {value_matrix[i][j]:.2f}"]
                for (str, val) in [(str1, visit_matrix[i][j]), (str2, value_matrix[i][j])]:
                    if float(val).is_integer():
                        text.append(f"{str}: {int(val)}")
                    else:
                        text.append(f"{str}: {val:.3f}")
                for _ in range(2):
                    txt_surface = self.font.render(text[_], True, (255, 15, 100))
                    self.screen.blit(txt_surface, (mouse_pos[0] + 10, mouse_pos[1] + 10 + _ * 15))


    def draw_buttons(self):
        for name, rect in self.buttons.items():
            color = (255, 255, 0) if (name == 'research' and self.enable_research) or (name == 'autoplay' and self.autoplay) or (name == 'show nn' and self.show_nn) else WHITE
            pygame.draw.rect(self.screen, color, rect)
            text = self.font.render(name.capitalize(), True, BLACK)
            self.screen.blit(text, (rect.x + 10, rect.y + 5))

    def draw_info(self, simulations, val):
        info_text = f"Sim: {simulations}, Val: {val:.2f}"
        txt_surface = self.font.render(info_text, True, BLACK)
        self.screen.blit(txt_surface, (10, self.width + 80))

    def get_move_from_pos(self, pos):
        x, y = pos
        i = round((y - self.margin) / self.cell_size)
        j = round((x - self.margin) / self.cell_size)
        if 0 <= i < self.board_size and 0 <= j < self.board_size and self.board[i][j] == 0:
            return (i, j)
        return None

    def make_move(self, move):
        if move is not None:
            i, j = move
            self.board[i][j] = self.current_player
            self.move_history = self.move_history[:self.history_index + 1]
            self.move_history.append((move, self.current_player))
            self.history_index += 1
            self.current_player = -self.current_player
            self.root = None  # Reset root after move

    def undo_move(self):
        if self.history_index >= 0:
            move, player = self.move_history[self.history_index]
            i, j = move
            self.board[i][j] = 0
            self.current_player = player
            self.history_index -= 1
            self.root = None

    def redo_move(self):
        if self.history_index < len(self.move_history) - 1:
            self.history_index += 1
            move, player = self.move_history[self.history_index]
            i, j = move
            self.board[i][j] = player
            self.current_player = -player
            self.root = None

    def run_mcts(self, num_simulations, cur_root=None):
        new_board = copy.deepcopy(self.board)
        if self.current_player == -1:
            for i in range(0, self.board_size):
                for j in range(0, self.board_size):
                    new_board[i][j] = -new_board[i][j]
        result = self.mcts.run(new_board, num_simulations, train=0, cur_root=cur_root, return_root=1)
        _, self.root = result

    def get_ai_move(self):
        if gomoku_cnn.evaluation_func(self.board):
            return None
        visit_matrix = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        for move, (child, _) in self.root.children.items():
            if child is not None:
                i, j = move
                visit_matrix[i][j] = child.visit_count
        move = np.unravel_index(np.argmax(visit_matrix), visit_matrix.shape)
        return move

    def press_research(self):
        self.enable_research = not self.enable_research
        if self.enable_research:
            self.show_nn = False
    def press_autoplay(self):
        self.autoplay = not self.autoplay
    def press_play(self):
        self.show_nn_val, self.show_nn_prob = None, None
        already_done = self.root.visit_count if self.root else 0
        if already_done < Config.ai_simulation:
            self.run_mcts(Config.ai_simulation - already_done)
        move = self.get_ai_move()
        self.make_move(move)
    def press_back(self):
        self.show_nn_val, self.show_nn_prob = None, None
        self.undo_move()
    def press_forward(self):
        self.show_nn_val, self.show_nn_prob = None, None
        self.redo_move()
    def press_show_nn(self):
        if self.show_nn == False:
            if self.enable_research == False and gomoku_cnn.evaluation_func(self.board) == 0:
                self.show_nn = True
                self.root = None
        else:
            self.show_nn = False

    async def update_loop(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                #print("O_o")
                pos = pygame.mouse.get_pos()
                for name, rect in self.buttons.items():
                    if rect.collidepoint(pos):
                        if name == 'research':
                            self.press_research()
                        elif name == 'autoplay':
                            self.press_autoplay()
                        elif name == 'play':
                            self.press_play()
                        elif name == 'back':
                            self.press_back()
                        elif name == 'forward':
                            self.press_forward()
                        elif name == 'show nn':
                            self.press_show_nn()
                        break
                else:
                    self.show_nn_val, self.show_nn_prob = None, None
                    move = self.get_move_from_pos(pos)
                    if move:
                        self.make_move(move)
            elif event.type == pygame.KEYDOWN:
                print(event.key)
                if event.key == pygame.K_r:  # 'R' for research
                    self.press_research()
                elif event.key == pygame.K_a:  # 'A' for autoplay
                    self.press_autoplay()
                elif event.key == pygame.K_p:  # 'P' for play
                    self.press_play()
                elif event.key == pygame.K_b:  # 'B' for back
                    self.press_back()
                elif event.key == pygame.K_f:  # 'F' for forward
                    self.press_forward()
                elif event.key == pygame.K_s and 'show nn' in self.buttons:  # 'N' for show nn
                    self.press_show_nn()

        if self.enable_research or self.autoplay:
            self.run_mcts(Config.simulation_update if self.root else Config.simulation_update * 3, cur_root=self.root)

        if self.autoplay and self.root and self.root.visit_count >= Config.ai_simulation:
            move = self.get_ai_move()
            self.make_move(move)

        self.draw_board()
        if self.root and self.enable_research:
            val, visit_matrix, value_matrix = self.mcts.show_data(self.root)
            self.draw_heatmap(visit_matrix, value_matrix)
            self.draw_hover_info(pygame.mouse.get_pos(), visit_matrix, value_matrix)
            self.draw_info(self.root.visit_count, val)
        elif self.show_nn:
            if self.show_nn_prob == None:
                board = copy.deepcopy(self.board)
                if self.current_player == -1:
                    for i in range(self.board_size):
                        for j in range(self.board_size):
                            board[i][j] = -board[i][j]
                self.show_nn_prob, self.show_nn_val = gomoku_cnn.show_nn(self.mcts.model, board)
            #print(self.show_nn_prob)
            #print(self.show_nn_val)
            self.draw_heatmap(self.show_nn_prob, self.show_nn_val)
            self.draw_hover_info(pygame.mouse.get_pos(), self.show_nn_prob, self.show_nn_val, "Prob", "Value")
        self.draw_buttons()
        pygame.display.flip()
        return True

    def setup(self):
        pygame.display.set_caption("Gomoku AI GUI")

    async def main(self):
        self.setup()
        running = True
        while running:
            running = await self.update_loop()
            await asyncio.sleep(1.0 / 60)  # 60 FPS

        pygame.quit()

if platform.system() == "Emscripten":
    game = GomokuGUI()
    asyncio.ensure_future(game.main())
else:
    if __name__ == "__main__":
        game = GomokuGUI()
        asyncio.run(game.main())