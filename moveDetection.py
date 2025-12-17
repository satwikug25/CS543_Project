import chess
import numpy as np
import copy

# Initial board position in FEN
INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

class BoardState:
    def __init__(self):
        self.board = chess.Board(INITIAL_FEN)
        self.piece_types = self._initialize_piece_types()
        self.history = []  # List of (FEN, move) tuples
        self.turn = 'w'  # Track whose turn it is ('w' or 'b')

    def _initialize_piece_types(self):
        piece_types = [['empty'] * 8 for _ in range(8)]
        piece_types[0] = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        piece_types[1] = ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P']
        piece_types[6] = ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p']
        piece_types[7] = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']
        return piece_types

    def set_board_state(self, detected_state):
        initial_pieces = self._initialize_piece_types()
        for row in range(8):
            for col in range(8):
                if detected_state[row][col] == 'piece-white':
                    if initial_pieces[row][col].isupper():
                        self.piece_types[row][col] = initial_pieces[row][col]
                    else:
                        self.piece_types[row][col] = 'P'
                elif detected_state[row][col] == 'piece-black':
                    if initial_pieces[row][col].islower():
                        self.piece_types[row][col] = initial_pieces[row][col]
                    else:
                        self.piece_types[row][col] = 'p'
                else:
                    self.piece_types[row][col] = 'empty'

    def update_from_detection(self, detected_state):
        from_square, to_square = self._detect_move(detected_state)
        if from_square and to_square:
            fen_before = self.get_fen()
            self.history.append((fen_before, f"{from_square}->{to_square}"))

            from_row, from_col = self._algebraic_to_coords(from_square)
            to_row, to_col = self._algebraic_to_coords(to_square)

            # Store the piece that's moving
            moving_piece = self.piece_types[from_row][from_col]
            
            # Clear both squares
            self.piece_types[from_row][from_col] = 'empty'
            self.piece_types[to_row][to_col] = 'empty'
            
            # Place the piece in its new position
            self.piece_types[to_row][to_col] = moving_piece

            # Update the chess board
            move = chess.Move.from_uci(f"{from_square}{to_square}")
            self.board.push(move)

            if (moving_piece.lower() == 'p' and 
                ((to_row == 0 and moving_piece.isupper()) or 
                 (to_row == 7 and moving_piece.islower()))):
                self.piece_types[to_row][to_col] = 'Q' if moving_piece.isupper() else 'q'

            # Flip turn after move
            self.turn = 'b' if self.turn == 'w' else 'w'

    def undo_last_move(self):
        if not self.history:
            print("No moves to undo.")
            return
        last_fen, last_move = self.history.pop()
        self._load_fen_into_pieces(last_fen)
        print(f"Undid move: {last_move}")
        # Flip turn back after undo
        self.turn = 'b' if self.turn == 'w' else 'w'

    def _load_fen_into_pieces(self, fen):
        rows = fen.split(' ')[0].split('/')
        for row_idx, row_str in enumerate(rows):
            col_idx = 0
            for char in row_str:
                if char.isdigit():
                    for _ in range(int(char)):
                        self.piece_types[7 - row_idx][col_idx] = 'empty'
                        col_idx += 1
                else:
                    self.piece_types[7 - row_idx][col_idx] = char
                    col_idx += 1

    def _detect_move(self, detected_state):
        differences = []
        for row in range(8):
            for col in range(8):
                current = 'empty' if self.piece_types[row][col] == 'empty' else (
                    'piece-white' if self.piece_types[row][col].isupper() else 'piece-black'
                )
                if current != detected_state[row][col]:
                    differences.append((row, col))

        if len(differences) != 2:
            raise ValueError(f"Invalid move detected - expected exactly 2 squares to change, got {len(differences)}")

        from_square, to_square = None, None
        for row, col in differences:
            current = 'empty' if self.piece_types[row][col] == 'empty' else (
                'piece-white' if self.piece_types[row][col].isupper() else 'piece-black'
            )
            if current != 'empty' and detected_state[row][col] == 'empty':
                from_square = self._coords_to_algebraic(row, col)
            elif current == 'empty' and detected_state[row][col] != 'empty':
                to_square = self._coords_to_algebraic(row, col)

        return from_square, to_square

    def _coords_to_algebraic(self, row, col):
        # Convert from (row,col) to algebraic notation
        # row: 0-7 (bottom to top)
        # col: 0-7 (right to left)
        file = chr(ord('h') - col)  # h->a maps to 0->7
        rank = str(row + 1)  # 0->7 maps to 1->8
        return file + rank

    def _algebraic_to_coords(self, square):
        # Convert from algebraic notation to (row,col)
        # square format: 'e4', 'h1', etc.
        file = ord('h') - (ord(square[0]) - ord('a'))  # h->a maps to 0->7
        rank = int(square[1]) - 1  # 1->8 maps to 0->7
        return rank, file

    def get_fen(self):
        fen = ""
        for row in range(7, -1, -1):
            empty_count = 0
            for col in range(8):
                piece = self.piece_types[row][col]
                if piece == 'empty':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen += str(empty_count)
                        empty_count = 0
                    fen += piece
            
            if empty_count > 0:
                fen += str(empty_count)
            if row > 0:
                fen += '/'
        
        # Add remaining FEN components with current turn
        fen += f" {self.turn} KQkq - 0 1"
        return fen

    def print_history(self):
        for idx, (fen, move) in enumerate(self.history):
            print(f"Move {idx + 1}: {move} → {fen}")
