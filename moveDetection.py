import chess
import numpy as np
import copy

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

class BoardState:
    def __init__(self):
        self.board = chess.Board(STARTING_FEN)
        self.piece_grid = self._setup_initial_pieces()
        self.move_log = []
        self.active_color = 'w'

    def _setup_initial_pieces(self):
        grid = [['empty'] * 8 for _ in range(8)]
        grid[0] = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        grid[1] = ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P']
        grid[6] = ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p']
        grid[7] = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']
        return grid

    def set_board_state(self, detection_grid):
        initial_setup = self._setup_initial_pieces()
        for r in range(8):
            for c in range(8):
                if detection_grid[r][c] == 'piece-white':
                    if initial_setup[r][c].isupper():
                        self.piece_grid[r][c] = initial_setup[r][c]
                    else:
                        self.piece_grid[r][c] = 'P'
                elif detection_grid[r][c] == 'piece-black':
                    if initial_setup[r][c].islower():
                        self.piece_grid[r][c] = initial_setup[r][c]
                    else:
                        self.piece_grid[r][c] = 'p'
                else:
                    self.piece_grid[r][c] = 'empty'

    def update_from_detection(self, detection_grid):
        src_sq, dst_sq = self._find_move_squares(detection_grid)
        if src_sq and dst_sq:
            prev_fen = self.get_fen()
            self.move_log.append((prev_fen, f"{src_sq}->{dst_sq}"))

            src_r, src_c = self._notation_to_indices(src_sq)
            dst_r, dst_c = self._notation_to_indices(dst_sq)

            moved_piece = self.piece_grid[src_r][src_c]
            
            self.piece_grid[src_r][src_c] = 'empty'
            self.piece_grid[dst_r][dst_c] = 'empty'
            
            self.piece_grid[dst_r][dst_c] = moved_piece

            chess_move = chess.Move.from_uci(f"{src_sq}{dst_sq}")
            self.board.push(chess_move)

            if (moved_piece.lower() == 'p' and 
                ((dst_r == 0 and moved_piece.isupper()) or 
                 (dst_r == 7 and moved_piece.islower()))):
                self.piece_grid[dst_r][dst_c] = 'Q' if moved_piece.isupper() else 'q'

            self.active_color = 'b' if self.active_color == 'w' else 'w'

    def undo_last_move(self):
        if not self.move_log:
            print("No moves to undo.")
            return
        prev_fen, prev_move = self.move_log.pop()
        self._restore_from_fen(prev_fen)
        print(f"Undid move: {prev_move}")
        self.active_color = 'b' if self.active_color == 'w' else 'w'

    def _restore_from_fen(self, fen_str):
        rank_strings = fen_str.split(' ')[0].split('/')
        for rank_idx, rank_data in enumerate(rank_strings):
            file_idx = 0
            for ch in rank_data:
                if ch.isdigit():
                    for _ in range(int(ch)):
                        self.piece_grid[7 - rank_idx][file_idx] = 'empty'
                        file_idx += 1
                else:
                    self.piece_grid[7 - rank_idx][file_idx] = ch
                    file_idx += 1

    def _find_move_squares(self, detection_grid):
        changed_squares = []
        for r in range(8):
            for c in range(8):
                current_state = 'empty' if self.piece_grid[r][c] == 'empty' else (
                    'piece-white' if self.piece_grid[r][c].isupper() else 'piece-black'
                )
                if current_state != detection_grid[r][c]:
                    changed_squares.append((r, c))

        if len(changed_squares) != 2:
            raise ValueError(f"Invalid move detected - expected exactly 2 squares to change, got {len(changed_squares)}")

        src_sq, dst_sq = None, None
        for r, c in changed_squares:
            current_state = 'empty' if self.piece_grid[r][c] == 'empty' else (
                'piece-white' if self.piece_grid[r][c].isupper() else 'piece-black'
            )
            if current_state != 'empty' and detection_grid[r][c] == 'empty':
                src_sq = self._indices_to_notation(r, c)
            elif current_state == 'empty' and detection_grid[r][c] != 'empty':
                dst_sq = self._indices_to_notation(r, c)

        return src_sq, dst_sq

    def _indices_to_notation(self, r, c):
        file_char = chr(ord('h') - c)
        rank_char = str(r + 1)
        return file_char + rank_char

    def _notation_to_indices(self, notation):
        file_idx = ord('h') - (ord(notation[0]) - ord('a'))
        rank_idx = int(notation[1]) - 1
        return rank_idx, file_idx

    def get_fen(self):
        fen_str = ""
        for r in range(7, -1, -1):
            empty_count = 0
            for c in range(8):
                piece_char = self.piece_grid[r][c]
                if piece_char == 'empty':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_str += str(empty_count)
                        empty_count = 0
                    fen_str += piece_char
            
            if empty_count > 0:
                fen_str += str(empty_count)
            if r > 0:
                fen_str += '/'
        
        fen_str += f" {self.active_color} KQkq - 0 1"
        return fen_str

    def print_history(self):
        for idx, (fen, move) in enumerate(self.move_log):
            print(f"Move {idx + 1}: {move} → {fen}")
