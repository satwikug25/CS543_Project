import cv2
import numpy as np
import os
import json
import pickle
import time
from PIL import Image
from google import genai
from moveDetection import BoardState
from test_moves import get_lichess_best_move
from boxCoordinates import BoxCoordinates
import chess

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def transform_detection_labels(raw_detection):
    transformed = [['empty' for _ in range(8)] for _ in range(8)]
    for r_idx in range(8):
        for c_idx in range(8):
            cell_label = raw_detection[r_idx][c_idx]
            if cell_label == 'piece-white':
                transformed[r_idx][c_idx] = 'piece-white'
            elif cell_label == 'piece-black':
                transformed[r_idx][c_idx] = 'piece-black'
            else:
                transformed[r_idx][c_idx] = 'empty'
    return transformed

def extract_move_from_detection(game_board, detection_matrix):
    normalized_detection = transform_detection_labels(detection_matrix)
    
    expected_state = [['empty' for _ in range(8)] for _ in range(8)]
    for sq in chess.SQUARES:
        piece_obj = game_board.piece_at(sq)
        f_idx = chess.square_file(sq)
        r_idx = chess.square_rank(sq)
        col_idx = 7 - f_idx
        row_idx = r_idx
        if piece_obj:
            expected_state[row_idx][col_idx] = 'piece-white' if piece_obj.color == chess.WHITE else 'piece-black'

    origin_sq = None
    dest_sq = None
    for row_idx in range(8):
        for col_idx in range(8):
            prev_val = expected_state[row_idx][col_idx]
            curr_val = normalized_detection[row_idx][col_idx]
            if prev_val != curr_val:
                f_idx = 7 - col_idx
                r_idx = row_idx
                sq = chess.square(f_idx, r_idx)
                if prev_val != 'empty' and curr_val == 'empty':
                    origin_sq = sq
                elif prev_val == 'empty' and curr_val != 'empty':
                    dest_sq = sq

    if origin_sq is None or dest_sq is None:
        raise Exception("Could not detect a valid move from the frame.")

    uci_move = chess.square_name(origin_sq) + chess.square_name(dest_sq)
    return uci_move

class ChessBoardAnalyzer:
    def __init__(self):
        with open("cameraMatrix.pkl", 'rb') as fh:
            self.cam_intrinsics = np.asarray(pickle.load(fh), dtype=float)
        with open("dist.pkl", 'rb') as fh:
            self.distortion_params = np.asarray(pickle.load(fh), dtype=float)

        self.tag_size = 5.0 / 100.0
        self.marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detect_params = cv2.aruco.DetectorParameters()
        self.id_to_corner = {0:2, 1:3, 2:0, 3:1}
        
        try:
            self.marker_finder = cv2.aruco.ArucoDetector(self.marker_dict, self.detect_params)
            self.modern_api = True
        except AttributeError:
            self.marker_finder = None
            self.modern_api = False

        self.game_state = BoardState()
        self.square_coords = BoxCoordinates()
        
        self.api_keys = [
            os.getenv("GEMINI_API_KEY"),
            os.getenv("GEMINI_API_KEY2"),
            os.getenv("GEMINI_API_KEY3"),
            os.getenv("GEMINI_API_KEY4"),
        ]
        self.active_key_idx = 0
        
        available_keys = [k for k in self.api_keys if k]
        if not available_keys:
            print("WARNING: No Gemini API keys found in environment variables!")
            print("Please set at least one of: geminiApiKey, geminiApiKey2, geminiApiKey3, geminiApiKey4")
            print("Make sure your .env file is in the project root and contains these variables.")
        else:
            print(f"Loaded {len(available_keys)} API key(s)")

        os.makedirs("grid_piece", exist_ok=True)

    def find_board_markers(self, img):
        if self.modern_api:
            marker_corners, marker_ids, _ = self.marker_finder.detectMarkers(img)
        else:
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                img, self.marker_dict, parameters=self.detect_params
            )
        
        if marker_ids is None:
            return None, None, None

        marker_ids = marker_ids.flatten()
        
        if not self.modern_api:
            try:
                rot_vecs, trans_vecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    marker_corners, self.tag_size,
                    self.cam_intrinsics, self.distortion_params
                )
            except AttributeError:
                pass

        corner_points = np.zeros((4,2), dtype=np.float32)
        for i, tag_id in enumerate(marker_ids):
            tag_id = int(tag_id)
            if tag_id not in self.id_to_corner:
                continue
            corner_pixels = marker_corners[i].reshape(4,2)
            mapped_idx = self.id_to_corner[tag_id]
            corner_points[mapped_idx] = corner_pixels[mapped_idx]

        return corner_points, marker_corners, marker_ids

    def analyze_frame(self, img):
        corner_points, marker_corners, marker_ids = self.find_board_markers(img)
        if corner_points is None:
            print("No markers detected!")
            return img

        self.render_grid_overlay(img, corner_points)
        
        px_per_meter = np.linalg.norm(corner_points[0] - corner_points[1]) / self.tag_size
        px_per_inch = px_per_meter * 0.0254
        cell_px = int(round(2.2 * px_per_inch))
        board_dim = cell_px * 8

        target_corners = np.array([[0,0],[board_dim,0],[board_dim,board_dim],[0,board_dim]], dtype=np.float32)
        transform_mat = cv2.getPerspectiveTransform(corner_points, target_corners)
        warped_board = cv2.warpPerspective(img, transform_mat, (board_dim,board_dim))

        for row in range(8):
            for col in range(8):
                top,bottom = row*cell_px,(row+1)*cell_px
                left,right = col*cell_px,(col+1)*cell_px
                square_img = warped_board[top:bottom, left:right]
                cv2.imwrite(f"grid_piece/piece_r{row}_c{col}.png", square_img)

        classification_results = self.classify_all_squares()
        
        self.save_detection_results(classification_results)
        
        self.analyze_game_moves()

        return img

    def classify_all_squares(self):
        classification_list = []
        vlm_model = "gemini-2.5-flash-image"
        
        for row in range(8):
            for col in range(8):
                img_name = f"piece_r{row}_c{col}.png"
                img_path = os.path.join("grid_piece", img_name)
                
                if not os.path.isfile(img_path):
                    continue

                result_label = "error"
                try_count = 0
                max_tries = 3
                
                while try_count < len(self.api_keys) * max_tries:
                    current_key = self.api_keys[self.active_key_idx]
                    
                    if not current_key:
                        self.active_key_idx = (self.active_key_idx + 1) % len(self.api_keys)
                        try_count += 1
                        continue

                    try:
                        api_client = genai.Client(api_key=current_key)
                        square_img = Image.open(img_path)
                        api_response = api_client.models.generate_content(
                            model=vlm_model,
                            contents=[
                                square_img,
                                (
                                    "This is a top-down photo of a single chess board square. "
                                    "Classify what you see into exactly ONE of these three categories:\n\n"
                                    "- 'empty' = no chess piece on the square (you only see the board surface)\n"
                                    "- 'piece-white' = a WHITE/LIGHT colored chess piece is present (cream, beige, light wood, or white color)\n"
                                    "- 'piece-black' = a BLACK/DARK colored chess piece is present (dark brown, black, or very dark color)\n\n"
                                    "Look at the COLOR of the piece itself, not the square color. "
                                    "Chess pieces are typically cylindrical or have a circular top when viewed from above.\n\n"
                                    "Respond with ONLY one word: empty, piece-white, or piece-black"
                                )
                            ],
                        )
                        result_label = api_response.text.strip().lower()
                        if result_label not in ['empty', 'piece-white', 'piece-black']:
                            print(f"Warning: Invalid label '{result_label}' for piece_r{row}_c{col}.png, retrying...")
                            time.sleep(1)
                            try_count += 1
                            continue
                        break
                    except Exception as err:
                        err_msg = str(err)
                        if '429' in err_msg or 'RESOURCE_EXHAUSTED' in err_msg or 'quota' in err_msg.lower():
                            wait_time = 2.0
                            if 'retryDelay' in err_msg or 'retry in' in err_msg.lower():
                                wait_time = 3.0
                            
                            print(f"Rate limit hit for key ending in ...{current_key[-4:] if current_key else 'N/A'}, waiting {wait_time}s...")
                            time.sleep(wait_time)
                            self.active_key_idx = (self.active_key_idx + 1) % len(self.api_keys)
                            try_count += 1
                        else:
                            print(f"Error with key ending in ...{current_key[-4:] if current_key else 'N/A'}: {err_msg[:100]}")
                            self.active_key_idx = (self.active_key_idx + 1) % len(self.api_keys)
                            try_count += 1
                    
                    time.sleep(0.5)
                
                if result_label == "error":
                    print(f"Failed to classify piece_r{row}_c{col}.png after {try_count} attempts")
                
                classification_list.append((row, col, result_label))
                
                if (row * 8 + col + 1) % 8 == 0:
                    print(f"Progress: {row * 8 + col + 1}/64 squares classified")
        
        return classification_list

    def save_detection_results(self, classification_list):
        output_file = "detection.json"
        
        if os.path.exists(output_file):
            with open(output_file, "r") as fh:
                stored_data = json.load(fh)
        else:
            stored_data = []

        entry_id = len(stored_data) + 1
        new_entry = {
            "id": entry_id,
            "results": [
                {"r": row, "c": col, "label": lbl}
                for row, col, lbl in classification_list
            ]
        }

        stored_data.append(new_entry)
        with open(output_file, "w") as fh:
            json.dump(stored_data, fh, indent=2)

    def analyze_game_moves(self):
        try:
            with open('detection.json', 'r') as fh:
                stored_data = json.load(fh)
            
            if len(stored_data) < 1:
                return

            latest_frame = stored_data[-1]

            current_detection = [['empty' for _ in range(8)] for _ in range(8)]

            for square_data in latest_frame['results']:
                current_detection[square_data['r']][square_data['c']] = square_data['label']

            try:
                identified_move = extract_move_from_detection(self.game_state.board, current_detection)
            except Exception as err:
                print(f"Note: {str(err)}")
                return
            
            move_object = chess.Move.from_uci(identified_move)
            
            white_to_move = self.game_state.board.turn == chess.WHITE
            
            if move_object in self.game_state.board.legal_moves:
                src_square = identified_move[:2]
                dst_square = identified_move[2:]
                grab_pos, drop_pos = self.square_coords.get_move_coordinates(src_square, dst_square)
                
                if white_to_move:
                    print(f"White move detected: {identified_move}")
                    print(f"Robot coordinates - Pickup: {grab_pos}, Place: {drop_pos}")
                else:
                    print(f"Black move detected: {identified_move}")
                    print(f"Robot coordinates - Pickup: {grab_pos}, Place: {drop_pos}")
                
                self.game_state.board.push(move_object)
                print(f"\nBoard state after {'White' if white_to_move else 'Black'} move:")
                print(self.game_state.board)
                current_fen = self.game_state.board.fen()
                print(f"FEN after move: {current_fen}")
                
                if self.game_state.board.turn == chess.BLACK:
                    engine_suggestion = get_lichess_best_move(current_fen)
                    print(f"\nSuggested Black move: {engine_suggestion}")
                    suggested_move = chess.Move.from_uci(engine_suggestion)
                    if suggested_move in self.game_state.board.legal_moves:
                        src_sq = engine_suggestion[:2]
                        dst_sq = engine_suggestion[2:]
                        opp_grab, opp_drop = self.square_coords.get_move_coordinates(src_sq, dst_sq)
                        print(f"Robot coordinates for Black move - Pickup: {opp_grab}, Place: {opp_drop}")
                        print("(Move not applied - waiting for physical move on board)")
                    else:
                        print(f"⚠ Illegal Black move suggested: {engine_suggestion}")
                else:
                    print("\nIt's White's turn to move.")
            else:
                print(f"⚠ Illegal move detected: {identified_move}")

        except Exception as err:
            print(f"Error processing moves: {err}")

    def render_grid_overlay(self, img, corner_points, num_cells=8, line_color=(0,255,0), line_width=2):
        pt_tl, pt_tr, pt_br, pt_bl = corner_points
        for idx in range(1, num_cells):
            ratio = idx / float(num_cells)
            line_start = tuple((pt_tl + ratio*(pt_tr - pt_tl)).astype(int))
            line_end = tuple((pt_bl + ratio*(pt_br - pt_bl)).astype(int))
            cv2.line(img, line_start, line_end, line_color, line_width)
            line_start = tuple((pt_tl + ratio*(pt_bl - pt_tl)).astype(int))
            line_end = tuple((pt_tr + ratio*(pt_br - pt_tr)).astype(int))
            cv2.line(img, line_start, line_end, line_color, line_width)

ChessPipeline = ChessBoardAnalyzer

def run_application():
    analyzer = ChessBoardAnalyzer()
    video_capture = cv2.VideoCapture(0)

    print("Press 'S' to capture and process frame")
    print("Press 'Q' to quit")

    while True:
        success, current_frame = video_capture.read()
        if not success:
            break

        cv2.imshow('Chess Pipeline', current_frame)

        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            break
        if pressed_key == ord('s'):
            print('Processing...')
            processed = analyzer.analyze_frame(current_frame)
            cv2.imshow('Processed', processed)
            cv2.waitKey(0)
            cv2.destroyWindow('Processed')

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    run_application()

if __name__ == "__main__":
    main()
