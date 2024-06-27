import pygame
import pygame.freetype
import numpy as np
from tensorflow import keras
import os

# Neural Network Model

def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 13)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='tanh')
    ])
    model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
    return model

def board_to_input(board):
    piece_to_int = {'p': 0, 'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5}
    input_array = np.zeros((8, 8, 13))
    for i in range(8):
        for j in range(8):
            piece = board[i][j]
            if piece == '--':
                input_array[i, j, 12] = 1  # Empty square channel
            else:
                color = 0 if piece[0] == 'w' else 6
                input_array[i, j, piece_to_int[piece[1]] + color] = 1
    return input_array

# Training loop
def train_model(model, X, y):
    model.fit(np.array([X]), np.array([y]), epochs=1, verbose=0)

# Collect training data
def collect_training_data(model, board, move, game_result=None):
    X = board_to_input(board)
    if game_result is not None:
        # Use the game result directly if provided
        y = game_result
    else:
        # Use the model's current evaluation if the game is ongoing
        y = model.predict(np.array([X]))[0][0]
    return X, y


# Initialize Pygame
pygame.init()

# Set up the display
BOARD_SIZE = 512
BORDER_SIZE = 40
INPUT_HEIGHT = 100
WIDTH = BOARD_SIZE + 2 * BORDER_SIZE
HEIGHT = BOARD_SIZE + 2 * BORDER_SIZE + INPUT_HEIGHT
DIMENSION = 8
SQ_SIZE = BOARD_SIZE // DIMENSION
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
INPUT_BG = (200, 200, 200)
RED = (255, 0, 0)

# Font
FONT = pygame.freetype.Font(None, 24)
COORD_FONT = pygame.freetype.Font(None, 18)

# Add new constants for the victory screen (corrected)
VICTORY_BG = (0, 0, 0, 180)  # Semi-transparent black
VICTORY_TEXT_COLOR = (255, 255, 255)  # White
BUTTON_COLOR = (100, 100, 255)  # Light blue
BUTTON_TEXT_COLOR = (0, 0, 0)  # Black

# Initialize the board
def init_board():
    return [
        ['bR', 'bN', 'bB', 'bK', 'bQ', 'bB', 'bN', 'bR'],
        ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp'],
        ['--', '--', '--', '--', '--', '--', '--', '--'],
        ['--', '--', '--', '--', '--', '--', '--', '--'],
        ['--', '--', '--', '--', '--', '--', '--', '--'],
        ['--', '--', '--', '--', '--', '--', '--', '--'],
        ['wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp'],
        ['wR', 'wN', 'wB', 'wK', 'wQ', 'wB', 'wN', 'wR']
    ]

# Add a new function to get the current player
def get_current_player(turn):
    return 'w' if turn % 2 == 0 else 'b'

# New function to generate all valid moves for a given color
def generate_valid_moves(board, color):
    valid_moves = []
    for start_row in range(8):
        for start_col in range(8):
            if board[start_row][start_col][0] == color:
                for end_row in range(8):
                    for end_col in range(8):
                        move = f"{chr(97+start_col)}{8-start_row}{chr(97+end_col)}{8-end_row}"
                        if is_valid_move(board, move, color)[0]:
                            valid_moves.append(move)
    return valid_moves


def save_model(model, filename='chess_model.keras'):
    model.save(filename, save_format='keras')
    print(f"Model saved to {filename}")

def load_model(filename='chess_model.keras'):
    if os.path.exists(filename):
        try:
            model = keras.models.load_model(filename)
            print(f"Model loaded from {filename}")
        except Exception as e:
            print(f"Failed to load model from {filename}: {e}")
            print("Creating a new model.")
            model = create_model()
    else:
        print(f"No saved model found at {filename}. Creating a new model.")
        model = create_model()
    return model

# Modified AI choose move function
def ai_choose_move(model, board, color):
    valid_moves = generate_valid_moves(board, color)
    if not valid_moves:
        return None  # No valid moves available
    
    # Evaluate all valid moves
    move_evaluations = []
    for move in valid_moves:
        temp_board = [row[:] for row in board]  # Create a copy of the board
        make_move(temp_board, move)
        X = board_to_input(temp_board)
        evaluation = model.predict(np.array([X]))[0][0]
        move_evaluations.append((move, evaluation))
    
    # Choose the best move (highest evaluation for white, lowest for black)
    best_move = max(move_evaluations, key=lambda x: x[1] if color == 'w' else -x[1])[0]
    return best_move

# Add a function to check for a win condition
def check_win_condition(board):
    white_king = any('wK' in row for row in board)
    black_king = any('bK' in row for row in board)
    
    if not white_king:
        return 'b'  # Black wins
    elif not black_king:
        return 'w'  # White wins
    else:
        return None  # No winner yet
# Draw the board
def draw_board(screen):
    colors = [LIGHT_SQUARE, DARK_SQUARE]
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(
                BORDER_SIZE + col*SQ_SIZE, 
                BORDER_SIZE + row*SQ_SIZE, 
                SQ_SIZE, SQ_SIZE))

    # Draw rank numbers
    for i in range(DIMENSION):
        COORD_FONT.render_to(screen, 
                             (BORDER_SIZE // 2, BORDER_SIZE + i*SQ_SIZE + SQ_SIZE//2), 
                             str(8-i), BLACK)
        COORD_FONT.render_to(screen, 
                             (WIDTH - BORDER_SIZE // 2, BORDER_SIZE + i*SQ_SIZE + SQ_SIZE//2), 
                             str(8-i), BLACK)

    # Draw file letters
    for i in range(DIMENSION):
        COORD_FONT.render_to(screen, 
                             (BORDER_SIZE + i*SQ_SIZE + SQ_SIZE//2, BORDER_SIZE // 2), 
                             chr(97 + i), BLACK)
        COORD_FONT.render_to(screen, 
                             (BORDER_SIZE + i*SQ_SIZE + SQ_SIZE//2, HEIGHT - INPUT_HEIGHT - BORDER_SIZE // 2), 
                             chr(97 + i), BLACK)

# Draw a single piece
def draw_piece(screen, piece, row, col):
    color = WHITE if piece[0] == 'w' else BLACK
    x = BORDER_SIZE + col * SQ_SIZE
    y = BORDER_SIZE + row * SQ_SIZE
    center = (x + SQ_SIZE // 2, y + SQ_SIZE // 2)
    
    if piece[1] == 'p':  # Pawn
        pygame.draw.circle(screen, color, center, SQ_SIZE // 4)
    elif piece[1] == 'R':  # Rook
        pygame.draw.rect(screen, color, (x + SQ_SIZE // 4, y + SQ_SIZE // 4, SQ_SIZE // 2, SQ_SIZE // 2))
    elif piece[1] == 'N':  # Knight
        points = [
            (x + SQ_SIZE // 4, y + SQ_SIZE * 3 // 4),
            (x + SQ_SIZE // 2, y + SQ_SIZE // 4),
            (x + SQ_SIZE * 3 // 4, y + SQ_SIZE * 3 // 4)
        ]
        pygame.draw.polygon(screen, color, points)
    elif piece[1] == 'B':  # Bishop
        pygame.draw.polygon(screen, color, [
            (x + SQ_SIZE // 2, y + SQ_SIZE // 4),
            (x + SQ_SIZE // 4, y + SQ_SIZE * 3 // 4),
            (x + SQ_SIZE * 3 // 4, y + SQ_SIZE * 3 // 4)
        ])
    elif piece[1] == 'Q':  # Queen
        pygame.draw.circle(screen, color, center, SQ_SIZE // 3)
        pygame.draw.rect(screen, color, (x + SQ_SIZE // 3, y + SQ_SIZE // 3, SQ_SIZE // 3, SQ_SIZE // 3))
    elif piece[1] == 'K':  # King
        pygame.draw.rect(screen, color, (x + SQ_SIZE // 3, y + SQ_SIZE // 4, SQ_SIZE // 3, SQ_SIZE // 2))
        pygame.draw.rect(screen, color, (x + SQ_SIZE // 4, y + SQ_SIZE // 3, SQ_SIZE // 2, SQ_SIZE // 3))

    # Draw a small label for the piece type
    font = pygame.font.SysFont('arial', int(SQ_SIZE * 0.3))
    text_surface = font.render(piece[1], True, WHITE if piece[0] == 'b' else BLACK)
    text_rect = text_surface.get_rect(center=(x + SQ_SIZE // 2, y + SQ_SIZE // 2))
    screen.blit(text_surface, text_rect)

   

# Draw all pieces
def draw_pieces(screen, board):
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            piece = board[row][col]
            if piece != '--':
                draw_piece(screen, piece, row, col)

# Draw input area
def draw_input_area(screen, input_text, error_message=""):
    pygame.draw.rect(screen, INPUT_BG, (0, HEIGHT - INPUT_HEIGHT, WIDTH, INPUT_HEIGHT))
    FONT.render_to(screen, (10, HEIGHT - INPUT_HEIGHT + 10), "Enter move (e.g. e2e4):", BLACK)
    pygame.draw.rect(screen, WHITE, (10, HEIGHT - INPUT_HEIGHT + 40, WIDTH - 20, 40))
    FONT.render_to(screen, (15, HEIGHT - INPUT_HEIGHT + 45), input_text, BLACK)
    if error_message:
        FONT.render_to(screen, (10, HEIGHT - INPUT_HEIGHT + 85), error_message, RED)

# Convert algebraic notation to board indices
def algebraic_to_index(algebraic):
    col = ord(algebraic[0].lower()) - ord('a')
    row = 8 - int(algebraic[1])
    return row, col

# Make a move on the board
def make_move(board, move):
    start, end = move[:2], move[2:]
    start_row, start_col = algebraic_to_index(start)
    end_row, end_col = algebraic_to_index(end)
    
    piece = board[start_row][start_col]
    board[start_row][start_col] = '--'
    board[end_row][end_col] = piece

# Helper function to get the color of a piece
def get_piece_color(piece):
    return piece[0] if piece != '--' else None

# Helper function to check if a move is diagonal
def is_diagonal_move(start_row, start_col, end_row, end_col):
    return abs(start_row - end_row) == abs(start_col - end_col)

# Helper function to check if the path is clear for diagonal movement
def is_diagonal_path_clear(board, start_row, start_col, end_row, end_col):
    row_step = 1 if end_row > start_row else -1
    col_step = 1 if end_col > start_col else -1
    current_row, current_col = start_row + row_step, start_col + col_step

    while current_row != end_row and current_col != end_col:
        if board[current_row][current_col] != '--':
            return False
        current_row += row_step
        current_col += col_step

    return True

# Helper function to check if the path is clear for horizontal movement
def is_horizontal_path_clear(board, row, start_col, end_col):
    step = 1 if end_col > start_col else -1
    for col in range(start_col + step, end_col, step):
        if board[row][col] != '--':
            return False
    return True

# Helper function to check if the path is clear for vertical movement
def is_vertical_path_clear(board, col, start_row, end_row):
    step = 1 if end_row > start_row else -1
    for row in range(start_row + step, end_row, step):
        if board[row][col] != '--':
            return False
    return True

def is_horizontal_move(start_row, end_row):
    return start_row == end_row

def is_valid_king_move(start_row, start_col, end_row, end_col):
    row_diff = abs(end_row - start_row)
    col_diff = abs(end_col - start_col)
    return row_diff <= 1 and col_diff <= 1 and (row_diff != 0 or col_diff != 0)

# Validate bishop move
def is_valid_bishop_move(board, start_row, start_col, end_row, end_col):
    # Check if the move is diagonal
    if not is_diagonal_move(start_row, start_col, end_row, end_col):
        return False

    # Check if the path is clear
    return is_diagonal_path_clear(board, start_row, start_col, end_row, end_col)

# Validate knight move
def is_valid_knight_move(start_row, start_col, end_row, end_col):
    row_diff = abs(end_row - start_row)
    col_diff = abs(end_col - start_col)
    return (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2)

# Helper function to check if a move is vertical
def is_vertical_move(start_col, end_col):
    return start_col == end_col

# Validate pawn move
def is_valid_pawn_move(board, start_row, start_col, end_row, end_col):
    piece = board[start_row][start_col]
    color = piece[0]
    direction = -1 if color == 'w' else 1  # White moves up, black moves down

    # Normal move (1 square forward)
    if is_vertical_move(start_col, end_col) and end_row == start_row + direction:
        return board[end_row][end_col] == '--'
    
    # First move (option to move 2 squares)
    if is_vertical_move(start_col, end_col) and end_row == start_row + 2*direction:
        return (start_row == 1 and color == 'b') or (start_row == 6 and color == 'w') and \
               board[start_row + direction][start_col] == '--' and board[end_row][end_col] == '--'
    
    # Capture move (diagonal)
    if is_diagonal_move(start_row, start_col, end_row, end_col) and end_row == start_row + direction:
        return board[end_row][end_col] != '--' and board[end_row][end_col][0] != color

    return False

# Validate rook move
def is_valid_rook_move(board, start_row, start_col, end_row, end_col):
    # Check if the move is either horizontal or vertical
    if not (is_horizontal_move(start_row, end_row) or is_vertical_move(start_col, end_col)):
        return False

    # Check if the path is clear
    if is_horizontal_move(start_row, end_row):
        return is_horizontal_path_clear(board, start_row, start_col, end_col)
    else:  # Vertical move
        return is_vertical_path_clear(board, start_col, start_row, end_row)
# Validate move

def is_valid_queen_move(board, start_row, start_col, end_row, end_col):
    # Queen can move like a rook or a bishop
    return is_valid_rook_move(board, start_row, start_col, end_row, end_col) or \
           is_valid_bishop_move(board, start_row, start_col, end_row, end_col)

# Add a new function to handle pawn promotion
def promote_pawn(board, end_row, end_col):
    piece = board[end_row][end_col]
    color = piece[0]
    if (color == 'w' and end_row == 0) or (color == 'b' and end_row == 7):
        board[end_row][end_col] = color + 'Q'  # Promote to Queen

def is_valid_move(board, move, current_player):
    if len(move) != 4:
        return False, "Invalid move format"
    
    start, end = move[:2], move[2:]
    try:
        start_row, start_col = algebraic_to_index(start)
        end_row, end_col = algebraic_to_index(end)
    except ValueError:
        return False, "Invalid square notation"
    
    if start_row < 0 or start_row > 7 or start_col < 0 or start_col > 7 or \
       end_row < 0 or end_row > 7 or end_col < 0 or end_col > 7:
        return False, "Move out of board"
    
    piece = board[start_row][start_col]
    if piece == '--':
        return False, "No piece at start position"
    
      # Check if the piece belongs to the current player
    if piece[0] != current_player:
        return False, f"It's {current_player}'s turn"

    # Check if the destination square has a piece of the same color
    if board[end_row][end_col] != '--' and board[end_row][end_col][0] == piece[0]:
        return False, "Cannot capture your own piece"

    # Pawn-specific move validation
    if piece[1] == 'p':
        if not is_valid_pawn_move(board, start_row, start_col, end_row, end_col):
            return False, "Invalid pawn move"
    
    # Bishop-specific move validation
    elif piece[1] == 'B':
        if not is_valid_bishop_move(board, start_row, start_col, end_row, end_col):
            return False, "Invalid bishop move"
        
    # Knight-specific move validation
    elif piece[1] == 'N':
        if not is_valid_knight_move(start_row, start_col, end_row, end_col):
            return False, "Invalid knight move"
    
    # Rook-specific move validation
    elif piece[1] == 'R':
        if not is_valid_rook_move(board, start_row, start_col, end_row, end_col):
            return False, "Invalid rook move"
    elif piece[1] == 'Q':
        if not is_valid_queen_move(board, start_row, start_col, end_row, end_col):
            return False, "Invalid queen move"
     # King-specific move validation
    elif piece[1] == 'K':
        if not is_valid_king_move(start_row, start_col, end_row, end_col):
            return False, "Invalid king move"
    

    # Add more specific piece movement rules here for other pieces
    
    return True, ""

# Make a move on the board
def make_move(board, move):
    start, end = move[:2], move[2:]
    start_row, start_col = algebraic_to_index(start)
    end_row, end_col = algebraic_to_index(end)
    
    piece = board[start_row][start_col]
    board[start_row][start_col] = '--'
    board[end_row][end_col] = piece

    if piece[1] == 'p':
        promote_pawn(board, end_row, end_col)


def draw_victory_screen(screen, winner):
    # Create a semi-transparent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill(VICTORY_BG)
    screen.blit(overlay, (0, 0))
    
    # Draw the victory message
    victory_text = f"{'White' if winner == 'w' else 'Black'} wins!"
    text_rect = FONT.get_rect(victory_text, size=48)
    text_rect.center = (WIDTH // 2, HEIGHT // 2 - 50)
    FONT.render_to(screen, text_rect, victory_text, VICTORY_TEXT_COLOR, size=48)
    
    # Draw the reset button
    button_rect = pygame.Rect(WIDTH // 2 - 75, HEIGHT // 2 + 50, 150, 50)
    pygame.draw.rect(screen, BUTTON_COLOR, button_rect)
    button_text = "Reset Game"
    text_rect = FONT.get_rect(button_text, size=24)
    text_rect.center = button_rect.center
    FONT.render_to(screen, text_rect, button_text, BUTTON_TEXT_COLOR, size=24)
    
    return button_rect  # Return the button rect for click detection

def ai_game_loop(screen, clock, model):
    board = init_board()
    running = True
    turn = 0
    winner = None
    move_history = []
    reset_delay = 180  # 3 seconds at 60 FPS

    while running:
        current_player = get_current_player(turn)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not winner:
            ai_move = ai_choose_move(model, board, current_player)
            if ai_move:
                X, y = collect_training_data(model, board, ai_move)
                make_move(board, ai_move)
                move_history.append((current_player, ai_move))
                winner = check_win_condition(board)
                if winner:
                    y = 1 if winner == 'w' else -1
                train_model(model, X, y)
                turn += 1
            else:
                # If AI has no valid moves, it loses
                winner = 'w' if current_player == 'b' else 'b'

        screen.fill(GRAY)
        draw_board(screen)
        draw_pieces(screen, board)
        
        if not winner:
            turn_text = f"{current_player.upper()}'s turn"
            FONT.render_to(screen, (10, HEIGHT - 30), turn_text, BLACK)
        else:
            victory_text = f"{'White' if winner == 'w' else 'Black'} wins! Resetting in {reset_delay // 60} seconds..."
            text_rect = FONT.get_rect(victory_text, size=36)
            text_rect.center = (WIDTH // 2, HEIGHT // 2)
            FONT.render_to(screen, text_rect, victory_text, WHITE, size=36)
            
            reset_delay -= 1
            if reset_delay <= 0:
                board = init_board()
                turn = 0
                winner = None
                move_history = []
                reset_delay = 180  # Reset the delay for the next game
        
        display_move_history(screen, move_history)
        
        pygame.display.flip()
        clock.tick(60)

    return model  # Return the model so it can be saved in the main function



def display_move_history(screen, move_history):
    history_text = "Move History:"
    FONT.render_to(screen, (WIDTH - 200, 10), history_text, BLACK)
    for i, (player, move) in enumerate(move_history[-10:]):  # Display last 10 moves
        move_text = f"{player}: {move}"
        FONT.render_to(screen, (WIDTH - 200, 40 + i * 20), move_text, BLACK)

# Modify the main function to use model saving/loading
def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    model = load_model()  # Load the model or create a new one if no saved model exists
    model = ai_game_loop(screen, clock, model)  # Get the updated model from the game loop
    save_model(model)  # Save the model after the game loop ends
    pygame.quit()

if __name__ == "__main__":
    main()