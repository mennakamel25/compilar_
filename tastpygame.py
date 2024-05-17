import pygame
import sys
import random

# Constants
SCREEN_SIZE = WIDTH, HEIGHT = 300, 300
LINE_WIDTH = 15
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4

# Colors
COLORS = {
    'BG_COLOR': (28, 170, 156),
    'LINE_COLOR': (23, 145, 135),
    'CIRCLE_COLOR': (239, 231, 200),
    'CROSS_COLOR': (66, 66, 66)
}

# Functions
def draw_board(surface):
    for row in range(1, BOARD_ROWS):
        pygame.draw.line(surface, COLORS['LINE_COLOR'], (0, row * SQUARE_SIZE), (WIDTH, row * SQUARE_SIZE), LINE_WIDTH)
    for col in range(1, BOARD_COLS):
        pygame.draw.line(surface, COLORS['LINE_COLOR'], (col * SQUARE_SIZE, 0), (col * SQUARE_SIZE, HEIGHT), LINE_WIDTH)

def initialize_board():
    return [['' for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]

def draw_figures(surface, board):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == 'O':
                pygame.draw.circle(surface, COLORS['CIRCLE_COLOR'], (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), CIRCLE_RADIUS, CIRCLE_WIDTH)
            elif board[row][col] == 'X':
                pygame.draw.line(surface, COLORS['CROSS_COLOR'], (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE), CROSS_WIDTH)
                pygame.draw.line(surface, COLORS['CROSS_COLOR'], (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE), (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), CROSS_WIDTH)

def is_winner(board, player):
    for i in range(BOARD_ROWS):
        if all(board[i][j] == player for j in range(BOARD_COLS)):  # Check rows
            return True
        if all(board[j][i] == player for j in range(BOARD_COLS)):  # Check columns
            return True
    if all(board[i][i] == player for i in range(BOARD_ROWS)):  # Check diagonal (top-left to bottom-right)
        return True
    if all(board[i][BOARD_COLS - i - 1] == player for i in range(BOARD_ROWS)):  # Check diagonal (top-right to bottom-left)
        return True
    return False

def is_board_full(board):
    return all(board[i][j] != '' for i in range(BOARD_ROWS) for j in range(BOARD_COLS))

def ai_move(board):
    empty_cells = [(i, j) for i in range(BOARD_ROWS) for j in range(BOARD_COLS) if board[i][j] == '']
    return random.choice(empty_cells)

# Main function
def main():
    pygame.init()
    WINDOW = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("Tic Tac Toe")
    clock = pygame.time.Clock()

    board = initialize_board()
    turn = random.choice(['X', 'O'])
    game_over = False

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                mouseX = event.pos[0] // SQUARE_SIZE
                mouseY = event.pos[1] // SQUARE_SIZE

                if board[mouseY][mouseX] == '':
                    board[mouseY][mouseX] = turn
                    if is_winner(board, turn) or is_board_full(board):
                        game_over = True
                    else:
                        turn = 'O' if turn == 'X' else 'X'

        WINDOW.fill(COLORS['BG_COLOR'])
        draw_board(WINDOW)
        draw_figures(WINDOW, board)
        pygame.display.update()

        if turn == 'O' and not game_over:
            row, col = ai_move(board)
            board[row][col] = 'O'
            if is_winner(board, 'O') or is_board_full(board):
                game_over = True
            else:
                turn = 'X'

        clock.tick(30)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()

if __name__ == "__main__":
    main()
