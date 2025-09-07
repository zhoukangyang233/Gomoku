import asyncio
import platform
import pygame
import numpy as np
import copy
import math

import nn_012 as gomoku_cnn
from nn_012 import MCTS

class Config:
    ai_model = 'run10_2000.pth'
    ai_simulation = 200            # 目标模拟次数（由拉条在 50~5000 间非线性调节，且为10的倍数）
    simulation_update = 10         # 每帧/步累加的模拟量
    show_shape = 'circle'          # square or circle
    show_type = 'colorful'         # colorful / red / green
    show_nn = True

# 颜色
BACKGROUND = (220, 179, 92)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BTN_FILL = (245, 245, 245)
BTN_ACTIVE = (255, 255, 0)   # 恢复为原来的亮黄色
BTN_BORDER = (60, 60, 60)
BLUE = (30, 144, 255)

class GomokuGUI:
    def __init__(self, board_size=15):
        pygame.init()
        self.board_size = board_size
        self.cell_size = 40
        self.margin = 20
        self.width = self.height = self.margin * 2 + self.cell_size * (self.board_size - 1)

        # ===== 三行 UI =====
        self.row_height = 40
        self.vgap = 8
        self.row1_y = self.width + 5
        self.row2_y = self.row1_y + self.row_height + self.vgap
        self.row3_y = self.row2_y + self.row_height + self.vgap  # 拉条行顶
        self.slider_zone_h = 50  # 第三行高度
        self.ui_height = (self.row3_y + self.slider_zone_h + 10) - self.width

        self.screen = pygame.display.set_mode((self.width, self.width + self.ui_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1
        self.move_history = []
        self.history_index = -1

        self.mcts = MCTS(model=Config.ai_model, use_rand=0)
        self.root = None

        self.enable_research = False
        self.show_nn = False
        self.show_nn_prob = None
        self.show_nn_val = None

        # === 双方是否由 AI 控制 ===
        self.black_is_ai = False
        self.white_is_ai = False

        # === 按钮容器 ===
        self.buttons = {}

        # ===== 行内居中布局助手 =====
        def layout_row_centered(names, widths, y, height, pad=12):
            total = sum(widths) + pad * (len(widths) - 1)
            x = (self.width - total) // 2
            rects = {}
            for name, w in zip(names, widths):
                rects[name] = pygame.Rect(x, y, w, height)
                x += w + pad
            return rects

        # === 第1行：Research、Black、White（整体居中） ===
        row1_names = ['research', 'black', 'white']
        row1_widths = [150, 150, 150]
        self.buttons.update(layout_row_centered(row1_names, row1_widths, self.row1_y, self.row_height))

        # === 第2行：Play、Back(方块)、Forward(方块)、Show nn（整体居中） ===
        square = self.cell_size  # 40x40 方块
        row2_names = ['play', 'back', 'forward', 'show nn']
        row2_widths = [150, square, square, 150]
        self.buttons.update(layout_row_centered(row2_names, row2_widths, self.row2_y, self.row_height))

        # === 第3行：整行拉条 ===
        self.slider_left_margin = 14
        self.slider_right_margin = 14
        self.slider_left = self.slider_left_margin
        self.slider_right = self.width - self.slider_right_margin
        self.slider_y_info = self.row3_y + 6            # 信息文字行（上）
        self.slider_y = self.row3_y + 28                # 轨道中心
        self.slider_dragging = False

        # 对数刻度：50~5000，步进10
        self._log_min = math.log10(50)
        self._log_max = math.log10(5000)
        self.slider_t = self._value_to_t(Config.ai_simulation)  # [0,1]

        # 图标颜色
        self.icon_color = (40, 40, 40)

    # --- 拉条映射：log10 轴 + 步进10 ---
    def _t_to_value(self, t):
        t = max(0.0, min(1.0, t))
        v = 10 ** (self._log_min + t * (self._log_max - self._log_min))
        v = int(round(v / 10.0) * 10)
        v = max(50, min(5000, v))
        return v

    def _value_to_t(self, v):
        v = int(round(v / 10.0) * 10)
        v = max(50, min(5000, v))
        return (math.log10(v) - self._log_min) / (self._log_max - self._log_min)

    def _update_slider_from_x(self, x):
        self.slider_t = (x - self.slider_left) / (self.slider_right - self.slider_left)
        self.slider_t = max(0.0, min(1.0, self.slider_t))
        Config.ai_simulation = self._t_to_value(self.slider_t)

    def _clamped_text_pos(self, x_center, label_surface):
        w = label_surface.get_width()
        x = int(x_center - w // 2)
        x = max(2, min(self.width - 2 - w, x))
        return x

    # --- 绘制简洁箭头图标（矢量）---
    def _draw_arrow_left(self, surface, rect, color):
        cx = rect.x + rect.width // 2
        cy = rect.y + rect.height // 2
        w = rect.width * 0.55
        h = rect.height * 0.35
        shaft = rect.width * 0.18
        tip = (cx - w/2, cy)
        tail_top = (cx + w/2, cy - h/2)
        tail_bot = (cx + w/2, cy + h/2)
        pygame.draw.polygon(surface, color, [(tip[0], tip[1]), (tail_top[0], tail_top[1]), (tail_bot[0], tail_bot[1])])
        pygame.draw.rect(surface, color, pygame.Rect(cx + w/2 - shaft, cy - h/6, shaft, h/3))

    def _draw_arrow_right(self, surface, rect, color):
        cx = rect.x + rect.width // 2
        cy = rect.y + rect.height // 2
        w = rect.width * 0.55
        h = rect.height * 0.35
        shaft = rect.width * 0.18
        tip = (cx + w/2, cy)
        tail_top = (cx - w/2, cy - h/2)
        tail_bot = (cx - w/2, cy + h/2)
        pygame.draw.polygon(surface, color, [(tip[0], tip[1]), (tail_top[0], tail_top[1]), (tail_bot[0], tail_bot[1])])
        pygame.draw.rect(surface, color, pygame.Rect(cx - w/2, cy - h/6, shaft, h/3))

    # --- 绘制拉条 ---
    def draw_slider(self, val_for_info=None):
        # 信息行：Visited/Target 与 Value（保证不遮挡）
        visited = self.root.visit_count if self.root else 0
        if val_for_info is None:
            val_text = "--"
        else:
            try:
                val_text = f"{val_for_info:.2f}"
            except:
                val_text = str(val_for_info)

        info1 = self.font.render(f"Visited/Target: {visited} / {Config.ai_simulation}    Value: {val_text}", True, BLACK)
        self.screen.blit(info1, (self.slider_left, self.slider_y_info))

        # 轨道
        pygame.draw.line(self.screen, BLACK,
                         (self.slider_left, self.slider_y),
                         (self.slider_right, self.slider_y), 2)

        # 刻度（新增 100）：50, 100, 200, 500, 1000, 2000, 5000
        ticks = [50, 100, 200, 500, 1000, 2000, 5000]
        for tv in ticks:
            t = self._value_to_t(tv)
            x = int(self.slider_left + t * (self.slider_right - self.slider_left))
            pygame.draw.line(self.screen, BLACK, (x, self.slider_y - 6), (x, self.slider_y + 6), 2)
            label = self.font.render(str(tv), True, BLACK)
            lx = self._clamped_text_pos(x, label)
            self.screen.blit(label, (lx, self.slider_y + 8))

        # 手柄
        handle_x = int(self.slider_left + self.slider_t * (self.slider_right - self.slider_left))
        pygame.draw.circle(self.screen, BLUE, (handle_x, self.slider_y), 8)

    def draw_board(self):
        self.screen.fill(BACKGROUND)

        if self.board_size == 11:
            star_points = [2, 5, 8]
        elif self.board_size == 15:
            star_points = [3, 7, 11]
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

    def draw_hover_info(self, mouse_pos, visit_matrix, value_matrix, str1="Visit", str2="Value"):
        grid_pos = self.get_grid_pos(mouse_pos)
        if grid_pos:
            i, j = grid_pos
            if visit_matrix[i][j] > 0:
                text = []
                for (s, val) in [(str1, visit_matrix[i][j]), (str2, value_matrix[i][j])]:
                    if float(val).is_integer():
                        text.append(f"{s}: {int(val)}")
                    else:
                        text.append(f"{s}: {val:.3f}")
                for _ in range(2):
                    txt_surface = self.font.render(text[_], True, (255, 15, 100))
                    self.screen.blit(txt_surface, (mouse_pos[0] + 10, mouse_pos[1] + 10 + _ * 15))

    def draw_buttons(self):
        for name, rect in self.buttons.items():
            # 背景填充（激活=亮黄）
            if name == 'research':
                fill = BTN_ACTIVE if self.enable_research else BTN_FILL
            elif name == 'black':
                fill = BTN_ACTIVE if self.black_is_ai else BTN_FILL
            elif name == 'white':
                fill = BTN_ACTIVE if self.white_is_ai else BTN_FILL
            elif name == 'show nn':
                fill = BTN_ACTIVE if self.show_nn else BTN_FILL
            else:
                fill = BTN_FILL

            pygame.draw.rect(self.screen, fill, rect, border_radius=8)
            pygame.draw.rect(self.screen, BTN_BORDER, rect, width=1, border_radius=8)

            # 标签文本或图标
            if name == 'black':
                label = f"Black: {'AI' if self.black_is_ai else 'Human'}"
                text = self.font.render(label, True, BLACK)
                tx = rect.x + (rect.width - text.get_width()) // 2
                ty = rect.y + (rect.height - text.get_height()) // 2
                self.screen.blit(text, (tx, ty))
            elif name == 'white':
                label = f"White: {'AI' if self.white_is_ai else 'Human'}"
                text = self.font.render(label, True, BLACK)
                tx = rect.x + (rect.width - text.get_width()) // 2
                ty = rect.y + (rect.height - text.get_height()) // 2
                self.screen.blit(text, (tx, ty))
            elif name == 'back':
                self._draw_arrow_left(self.screen, rect, self.icon_color)
            elif name == 'forward':
                self._draw_arrow_right(self.screen, rect, self.icon_color)
            else:
                label = 'Show nn' if name == 'show nn' else name.capitalize()
                text = self.font.render(label, True, BLACK)
                tx = rect.x + (rect.width - text.get_width()) // 2
                ty = rect.y + (rect.height - text.get_height()) // 2
                self.screen.blit(text, (tx, ty))

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

    # === 按钮动作 ===
    def press_research(self):
        self.enable_research = not self.enable_research
        if self.enable_research:
            self.show_nn = False

    def press_black(self):
        self.black_is_ai = not self.black_is_ai

    def press_white(self):
        self.white_is_ai = not self.white_is_ai

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
        if not self.show_nn:
            if (not self.enable_research) and gomoku_cnn.evaluation_func(self.board) == 0:
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
                pos = pygame.mouse.get_pos()
                # 拉条命中区：整行
                if (self.slider_left - 10 <= pos[0] <= self.slider_right + 10 and
                    self.row3_y <= pos[1] <= self.row3_y + self.slider_zone_h):
                    self.slider_dragging = True
                    self._update_slider_from_x(max(self.slider_left, min(self.slider_right, pos[0])))
                else:
                    # 按钮点击
                    for name, rect in self.buttons.items():
                        if rect.collidepoint(pos):
                            if name == 'research':
                                self.press_research()
                            elif name == 'black':
                                self.press_black()
                            elif name == 'white':
                                self.press_white()
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
                        # 棋盘落子（人类回合）
                        self.show_nn_val, self.show_nn_prob = None, None
                        move = self.get_move_from_pos(pos)
                        if move:
                            is_ai_turn = (self.current_player == 1 and self.black_is_ai) or \
                                         (self.current_player == -1 and self.white_is_ai)
                            if not is_ai_turn:
                                self.make_move(move)

            elif event.type == pygame.MOUSEBUTTONUP:
                self.slider_dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.slider_dragging:
                    pos = pygame.mouse.get_pos()
                    self._update_slider_from_x(max(self.slider_left, min(self.slider_right, pos[0])))

            elif event.type == pygame.KEYDOWN:
                # 方向键 ← / → 控制后退 / 前进
                if event.key == pygame.K_LEFT:
                    self.press_back()
                elif event.key == pygame.K_RIGHT:
                    self.press_forward()
                elif event.key == pygame.K_r:       # Research
                    self.press_research()
                elif event.key == pygame.K_p:       # Play（单步 AI）
                    self.press_play()
                elif event.key == pygame.K_s:       # Show nn
                    self.press_show_nn()

        # === 自动思考与落子：仅当轮到 AI 方或开启研究时 ===
        is_ai_turn = (self.current_player == 1 and self.black_is_ai) or \
                     (self.current_player == -1 and self.white_is_ai)

        if self.enable_research or is_ai_turn:
            self.run_mcts(Config.simulation_update if self.root else Config.simulation_update * 2, cur_root=self.root)

        if is_ai_turn and self.root and self.root.visit_count >= Config.ai_simulation:
            move = self.get_ai_move()
            self.make_move(move)

        # === 绘制 ===
        self.draw_board()

        # 研究显示 / NN 显示
        val_for_info = None
        if self.root and self.enable_research:
            val, visit_matrix, value_matrix = self.mcts.show_data(self.root)
            val_for_info = val
            self.draw_heatmap(visit_matrix, value_matrix)
            self.draw_hover_info(pygame.mouse.get_pos(), visit_matrix, value_matrix)
        elif self.show_nn:
            if self.show_nn_prob is None:
                board = copy.deepcopy(self.board)
                if self.current_player == -1:
                    for i in range(self.board_size):
                        for j in range(self.board_size):
                            board[i][j] = -board[i][j]
                self.show_nn_prob, self.show_nn_val = gomoku_cnn.show_nn(self.mcts.model, board)
            self.draw_heatmap(self.show_nn_prob, self.show_nn_val)
            self.draw_hover_info(pygame.mouse.get_pos(), self.show_nn_prob, self.show_nn_val, "Prob", "Value")
            if self.show_nn_val is not None:
                try:
                    val_for_info = float(np.mean(self.show_nn_val))
                except:
                    val_for_info = None

        self.draw_buttons()
        self.draw_slider(val_for_info=val_for_info)

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

def main():
    game = GomokuGUI()
    asyncio.run(game.main())
