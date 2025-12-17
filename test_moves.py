import json
import requests
import chess
from moveDetection import BoardState
from boxCoordinates import BoxCoordinates

def get_lichess_best_move(fen):
    url = f"https://lichess.org/api/cloud-eval?fen={fen}&multiPv=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['pvs'][0]['moves'].split()[0]
    else:
        raise Exception(f"Lichess API error: {response.status_code}, {response.text}")

def detect_white_move_from_frame(board, detected_state):
    previous_board_state = [['empty' for _ in range(8)] for _ in range(8)]
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        # Convert chess square to our coordinate system
        file = chess.square_file(square)  # 0-7 (a-h)
        rank = chess.square_rank(square)  # 0-7 (1-8)
        # Convert to our coordinate system where (0,0) is h1
        col = 7 - file  # h->a maps to 0->7
        row = rank      # 1->8 maps to 0->7
        if piece:
            previous_board_state[row][col] = 'piece-white' if piece.color == chess.WHITE else 'piece-black'

    from_square = None
    to_square = None
    for row in range(8):
        for col in range(8):
            before = previous_board_state[row][col]
            after = detected_state[row][col]
            if before != after:
                # Convert our coordinates back to chess square
                file = 7 - col  # 0->7 maps to h->a
                rank = row      # 0->7 maps to 1->8
                square = chess.square(file, rank)
                if before != 'empty' and after == 'empty':
                    from_square = square
                elif before == 'empty' and after != 'empty':
                    to_square = square

    if from_square is None or to_square is None:
        raise Exception("Could not detect a valid move from the frame.")

    move_uci = chess.square_name(from_square) + chess.square_name(to_square)
    return move_uci

def main():
    board_state = BoardState()
    coords = BoxCoordinates()
    print(f"Starting position: {board_state.get_fen()}")
    print("\nInitial board state:")
    print(board_state.board)

    try:
        with open('detection.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading detection.json: {e}")
        return

    for frame in data:
        frame_id = frame.get('id')

        if frame_id == 1:
            print(f"Initializing board from frame {frame_id}")
            continue  # skip initial frame

        print(f"\nProcessing frame {frame_id}...")

        try:
            detected_state = [['empty' for _ in range(8)] for _ in range(8)]
            for cell in frame.get('results', []):
                detected_state[cell['r']][cell['c']] = cell['label']

            detected_white_move = detect_white_move_from_frame(board_state.board, detected_state)

            white_move_obj = chess.Move.from_uci(detected_white_move)
            if white_move_obj in board_state.board.legal_moves:
                # Get robot coordinates for White's move
                from_square = detected_white_move[:2]
                to_square = detected_white_move[2:]
                white_pickup, white_place = coords.get_move_coordinates(from_square, to_square)
                print(f"White move: {detected_white_move}")
                print(f"Robot coordinates - Pickup: {white_pickup}, Place: {white_place}")
                
                board_state.board.push(white_move_obj)
                print("\nBoard state after White move:")
                print(board_state.board)
            else:
                print(f"⚠ Illegal White move detected: {detected_white_move}")
                continue

            fen_after_white = board_state.board.fen()
            print(f"FEN after White move: {fen_after_white}")

            best_black_move = get_lichess_best_move(fen_after_white)
            print(f"Lichess best move for Black: {best_black_move}")

            black_move_obj = chess.Move.from_uci(best_black_move)
            if black_move_obj in board_state.board.legal_moves:
                # Get robot coordinates for Black's move
                from_square = best_black_move[:2]
                to_square = best_black_move[2:]
                black_pickup, black_place = coords.get_move_coordinates(from_square, to_square)
                print(f"Black move: {best_black_move}")
                print(f"Robot coordinates - Pickup: {black_pickup}, Place: {black_place}")
                
                board_state.board.push(black_move_obj)
                print("\nBoard state after Black move:")
                print(board_state.board)
            else:
                print(f"⚠ Illegal Black move detected: {best_black_move}")

            print(f"FEN after Black move: {board_state.board.fen()}")

        except Exception as e:
            print(f"Error processing frame {frame_id}: {e}")

    print("\n=== Final Move History ===")
    board_state.print_history()
    print(f"Final FEN: {board_state.board.fen()}")
    print("\nFinal board state:")
    print(board_state.board)

if __name__ == "__main__":
    main()
