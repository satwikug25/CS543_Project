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

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, but that's okay - user might have set env vars manually
    pass

def convert_detected_to_board_state(detected_state):
    """Convert detected labels ('object', 'object-black', 'empty') to board state labels ('piece-white', 'piece-black', 'empty')"""
    board_state = [['empty' for _ in range(8)] for _ in range(8)]
    for row in range(8):
        for col in range(8):
            label = detected_state[row][col]
            if label == 'piece-white':
                board_state[row][col] = 'piece-white'
            elif label == 'piece-black':
                board_state[row][col] = 'piece-black'
            else:  # 'empty' or 'error'
                board_state[row][col] = 'empty'
    return board_state

def detect_white_move_from_frame(board, detected_state):
    # Convert detected state labels to board state labels
    detected_state_converted = convert_detected_to_board_state(detected_state)
    
    previous_board_state = [['empty' for _ in range(8)] for _ in range(8)]
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        file = chess.square_file(square)  # 0-7 (a-h)
        rank = chess.square_rank(square)  # 0-7 (1-8)
        col = 7 - file  # h->a maps to 0->7
        row = rank      # 1->8 maps to 0->7
        if piece:
            previous_board_state[row][col] = 'piece-white' if piece.color == chess.WHITE else 'piece-black'

    from_square = None
    to_square = None
    for row in range(8):
        for col in range(8):
            before = previous_board_state[row][col]
            after = detected_state_converted[row][col]
            if before != after:
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

class ChessPipeline:
    def __init__(self):
        # Load camera calibration
        with open("cameraMatrix.pkl", 'rb') as f:
            self.camera_matrix = np.asarray(pickle.load(f), dtype=float)
        with open("dist.pkl", 'rb') as f:
            self.dist_coeffs = np.asarray(pickle.load(f), dtype=float)

        # Initialize ArUco detector
        self.marker_length = 5.0 / 100.0  # 5cm in meters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.corner_map = {0:2, 1:3, 2:0, 3:1}
        
        # Try to initialize new API (OpenCV 4.7+), fall back to old API if not available
        try:
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
            self.use_new_api = True
        except AttributeError:
            self.aruco_detector = None
            self.use_new_api = False

        # Initialize board state
        self.board_state = BoardState()
        self.coords = BoxCoordinates()
        
        # Initialize Gemini API keys
        self.keys = [
            os.getenv("GEMINI_API_KEY"),
            os.getenv("GEMINI_API_KEY2"),
            os.getenv("GEMINI_API_KEY3"),
            os.getenv("GEMINI_API_KEY4"),
        ]
        self.current_key_index = 0
        
        # Check if any keys are loaded
        valid_keys = [key for key in self.keys if key]
        if not valid_keys:
            print("WARNING: No Gemini API keys found in environment variables!")
            print("Please set at least one of: geminiApiKey, geminiApiKey2, geminiApiKey3, geminiApiKey4")
            print("Make sure your .env file is in the project root and contains these variables.")
        else:
            print(f"Loaded {len(valid_keys)} API key(s)")

        # Create output directories
        os.makedirs("grid_piece", exist_ok=True)

    def detect_markers(self, frame):
        # Support both old and new OpenCV ArUco API
        if self.use_new_api:
            # New API (OpenCV 4.7+)
            corners_list, ids, _ = self.aruco_detector.detectMarkers(frame)
        else:
            # Old API (OpenCV < 4.7)
            corners_list, ids, _ = cv2.aruco.detectMarkers(
                frame, self.aruco_dict, parameters=self.parameters
            )
        
        if ids is None:
            return None, None, None

        ids = ids.flatten()
        
        # Pose estimation (not used in this function, but kept for compatibility)
        # Only compute if old API is available
        if not self.use_new_api:
            try:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners_list, self.marker_length,
                    self.camera_matrix, self.dist_coeffs
                )
            except AttributeError:
                # If estimatePoseSingleMarkers doesn't exist, skip pose estimation
                pass

        # Get board corners
        src_pts = np.zeros((4,2), dtype=np.float32)
        for idx, mid in enumerate(ids):
            mid = int(mid)
            if mid not in self.corner_map:
                continue
            pix = corners_list[idx].reshape(4,2)
            bidx = self.corner_map[mid]
            src_pts[bidx] = pix[bidx]

        return src_pts, corners_list, ids

    def process_frame(self, frame):
        # Detect markers
        src_pts, corners_list, ids = self.detect_markers(frame)
        if src_pts is None:
            print("No markers detected!")
            return frame

        # Draw grid on frame
        self.draw_grid(frame, src_pts)
        
        # Compute grid size
        pixel_per_m = np.linalg.norm(src_pts[0] - src_pts[1]) / self.marker_length
        PIX_PER_INCH = pixel_per_m * 0.0254
        GRID_PIX = int(round(2.2 * PIX_PER_INCH))
        size = GRID_PIX * 8

        # Warp perspective
        dst_pts = np.array([[0,0],[size,0],[size,size],[0,size]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        flat_grid = cv2.warpPerspective(frame, H, (size,size))

        # Save grid cells
        for r in range(8):
            for c in range(8):
                y0,y1 = r*GRID_PIX,(r+1)*GRID_PIX
                x0,x1 = c*GRID_PIX,(c+1)*GRID_PIX
                cell = flat_grid[y0:y1, x0:x1]
                cv2.imwrite(f"grid_piece/piece_r{r}_c{c}.png", cell)

        # Classify pieces using LLM
        results = self.classify_pieces()
        
        # Update detection.json
        self.update_detection_json(results)
        
        # Detect moves and get best response
        self.process_moves()

        return frame

    def classify_pieces(self):
        results = []
        # Use gemini-2.5-flash-image for higher rate limits (as suggested by API)
        model_name = "gemini-2.5-flash-image"
        
        for r in range(8):
            for c in range(8):
                filename = f"piece_r{r}_c{c}.png"
                image_path = os.path.join("grid_piece", filename)
                
                if not os.path.isfile(image_path):
                    continue

                label = "error"
                attempts = 0
                max_retries = 3  # Maximum retries per key
                
                while attempts < len(self.keys) * max_retries:
                    key = self.keys[self.current_key_index]
                    
                    if not key:
                        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                        attempts += 1
                        continue

                    try:
                        client = genai.Client(api_key=key)
                        img = Image.open(image_path)
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[
                                img,
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
                        label = response.text.strip().lower()
                        # Validate response
                        if label not in ['empty', 'piece-white', 'piece-black']:
                            print(f"Warning: Invalid label '{label}' for piece_r{r}_c{c}.png, retrying...")
                            time.sleep(1)  # Brief delay before retry
                            attempts += 1
                            continue
                        break
                    except Exception as e:
                        error_str = str(e)
                        # Check if it's a rate limit error
                        if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str or 'quota' in error_str.lower():
                            # Extract retry delay if available
                            retry_delay = 2.0  # Default delay
                            if 'retryDelay' in error_str or 'retry in' in error_str.lower():
                                # Try to extract delay (simplified - could be improved)
                                retry_delay = 3.0
                            
                            print(f"Rate limit hit for key ending in ...{key[-4:] if key else 'N/A'}, waiting {retry_delay}s...")
                            time.sleep(retry_delay)
                            # Try next key
                            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                            attempts += 1
                        else:
                            print(f"Error with key ending in ...{key[-4:] if key else 'N/A'}: {error_str[:100]}")
                            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
                            attempts += 1
                    
                    # Add delay between API calls to avoid rate limits
                    time.sleep(0.5)  # 500ms delay between requests
                
                if label == "error":
                    print(f"Failed to classify piece_r{r}_c{c}.png after {attempts} attempts")
                
                results.append((r, c, label))
                
                # Progress indicator
                if (r * 8 + c + 1) % 8 == 0:
                    print(f"Progress: {r * 8 + c + 1}/64 squares classified")
        
        return results

    def update_detection_json(self, results):
        detection_file = "detection.json"
        
        if os.path.exists(detection_file):
            with open(detection_file, "r") as f:
                data = json.load(f)
        else:
            data = []

        new_id = len(data) + 1
        run_entry = {
            "id": new_id,
            "results": [
                {"r": r, "c": c, "label": label}
                for r, c, label in results
            ]
        }

        data.append(run_entry)
        with open(detection_file, "w") as f:
            json.dump(data, f, indent=2)

    def process_moves(self):
        try:
            with open('detection.json', 'r') as f:
                data = json.load(f)
            
            if len(data) < 1:
                return

            # Get the latest frame
            curr_frame = data[-1]

            # Convert frame to board state
            curr_state = [['empty' for _ in range(8)] for _ in range(8)]

            for cell in curr_frame['results']:
                curr_state[cell['r']][cell['c']] = cell['label']

            # Detect move (works for both white and black)
            try:
                detected_move = detect_white_move_from_frame(self.board_state.board, curr_state)
            except Exception as e:
                # No valid move detected (expected when board states are identical or no move occurred)
                print(f"Note: {str(e)}")
                return
            
            move_obj = chess.Move.from_uci(detected_move)
            
            # Determine whose turn it is
            is_white_turn = self.board_state.board.turn == chess.WHITE
            
            if move_obj in self.board_state.board.legal_moves:
                from_square = detected_move[:2]
                to_square = detected_move[2:]
                pickup, place = self.coords.get_move_coordinates(from_square, to_square)
                
                # Apply the detected move
                if is_white_turn:
                    print(f"White move detected: {detected_move}")
                    print(f"Robot coordinates - Pickup: {pickup}, Place: {place}")
                else:
                    print(f"Black move detected: {detected_move}")
                    print(f"Robot coordinates - Pickup: {pickup}, Place: {place}")
                
                self.board_state.board.push(move_obj)
                print(f"\nBoard state after {'White' if is_white_turn else 'Black'} move:")
                print(self.board_state.board)
                fen_after_move = self.board_state.board.fen()
                print(f"FEN after move: {fen_after_move}")
                
                # Get opponent's best move (suggestion only, don't apply it)
                if self.board_state.board.turn == chess.BLACK:
                    best_black_move = get_lichess_best_move(fen_after_move)
                    print(f"\nSuggested Black move: {best_black_move}")
                    black_move_obj = chess.Move.from_uci(best_black_move)
                    if black_move_obj in self.board_state.board.legal_moves:
                        from_square = best_black_move[:2]
                        to_square = best_black_move[2:]
                        black_pickup, black_place = self.coords.get_move_coordinates(from_square, to_square)
                        print(f"Robot coordinates for Black move - Pickup: {black_pickup}, Place: {black_place}")
                        print("(Move not applied - waiting for physical move on board)")
                    else:
                        print(f"⚠ Illegal Black move suggested: {best_black_move}")
                else:
                    print("\nIt's White's turn to move.")
            else:
                print(f"⚠ Illegal move detected: {detected_move}")

        except Exception as e:
            print(f"Error processing moves: {e}")

    def draw_grid(self, img, src_pts, grid_size=8, color=(0,255,0), thickness=2):
        tl, tr, br, bl = src_pts
        for i in range(1, grid_size):
            α = i / float(grid_size)
            # vertical
            start = tuple((tl + α*(tr - tl)).astype(int))
            end = tuple((bl + α*(br - bl)).astype(int))
            cv2.line(img, start, end, color, thickness)
            # horizontal
            start = tuple((tl + α*(bl - tl)).astype(int))
            end = tuple((tr + α*(br - tr)).astype(int))
            cv2.line(img, start, end, color, thickness)

def main():
    pipeline = ChessPipeline()
    cap = cv2.VideoCapture(0)

    print("Press 'S' to capture and process frame")
    print("Press 'Q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame
        cv2.imshow('Chess Pipeline', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            print('Processing...')
            out = pipeline.process_frame(frame)
            cv2.imshow('Processed', out)
            cv2.waitKey(0)
            cv2.destroyWindow('Processed')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 