import json
import requests
import chess
from moveDetection import BoardState
from boxCoordinates import BoxCoordinates

def get_lichess_best_move(fen_string):
    api_url = f"https://lichess.org/api/cloud-eval?fen={fen_string}&multiPv=1"
    response = requests.get(api_url)
    if response.status_code == 200:
        result = response.json()
        return result['pvs'][0]['moves'].split()[0]
    else:
        raise Exception(f"Lichess API error: {response.status_code}, {response.text}")

def find_move_from_detection(game_board, detection_grid):
    expected_grid = [['empty' for _ in range(8)] for _ in range(8)]
    for sq in chess.SQUARES:
        piece_obj = game_board.piece_at(sq)
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        c = 7 - f
        row = r
        if piece_obj:
            expected_grid[row][c] = 'piece-white' if piece_obj.color == chess.WHITE else 'piece-black'

    origin = None
    destination = None
    for row in range(8):
        for col in range(8):
            prev = expected_grid[row][col]
            curr = detection_grid[row][col]
            if prev != curr:
                f = 7 - col
                r = row
                sq = chess.square(f, r)
                if prev != 'empty' and curr == 'empty':
                    origin = sq
                elif prev == 'empty' and curr != 'empty':
                    destination = sq

    if origin is None or destination is None:
        raise Exception("Could not detect a valid move from the frame.")

    move_str = chess.square_name(origin) + chess.square_name(destination)
    return move_str

def run_test():
    game = BoardState()
    coord_mapper = BoxCoordinates()
    print(f"Starting position: {game.get_fen()}")
    print("\nInitial board state:")
    print(game.board)

    try:
        with open('detection.json', 'r') as fh:
            detection_data = json.load(fh)
    except Exception as err:
        print(f"Error reading detection.json: {err}")
        return

    for frame_data in detection_data:
        frame_num = frame_data.get('id')

        if frame_num == 1:
            print(f"Initializing board from frame {frame_num}")
            continue

        print(f"\nProcessing frame {frame_num}...")

        try:
            detection_grid = [['empty' for _ in range(8)] for _ in range(8)]
            for cell in frame_data.get('results', []):
                detection_grid[cell['r']][cell['c']] = cell['label']

            white_move = find_move_from_detection(game.board, detection_grid)

            move_obj = chess.Move.from_uci(white_move)
            if move_obj in game.board.legal_moves:
                src = white_move[:2]
                dst = white_move[2:]
                pickup_pos, drop_pos = coord_mapper.get_move_coordinates(src, dst)
                print(f"White move: {white_move}")
                print(f"Robot coordinates - Pickup: {pickup_pos}, Place: {drop_pos}")
                
                game.board.push(move_obj)
                print("\nBoard state after White move:")
                print(game.board)
            else:
                print(f"⚠ Illegal White move detected: {white_move}")
                continue

            current_fen = game.board.fen()
            print(f"FEN after White move: {current_fen}")

            best_response = get_lichess_best_move(current_fen)
            print(f"Lichess best move for Black: {best_response}")

            response_obj = chess.Move.from_uci(best_response)
            if response_obj in game.board.legal_moves:
                src = best_response[:2]
                dst = best_response[2:]
                pickup_pos, drop_pos = coord_mapper.get_move_coordinates(src, dst)
                print(f"Black move: {best_response}")
                print(f"Robot coordinates - Pickup: {pickup_pos}, Place: {drop_pos}")
                
                game.board.push(response_obj)
                print("\nBoard state after Black move:")
                print(game.board)
            else:
                print(f"⚠ Illegal Black move detected: {best_response}")

            print(f"FEN after Black move: {game.board.fen()}")

        except Exception as err:
            print(f"Error processing frame {frame_num}: {err}")

    print("\n=== Final Move History ===")
    game.print_history()
    print(f"Final FEN: {game.board.fen()}")
    print("\nFinal board state:")
    print(game.board)

def main():
    run_test()

if __name__ == "__main__":
    main()
