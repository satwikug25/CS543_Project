import cv2
import os
import sys
from chess_pipeline import ChessPipeline

def scan_cameras():
    found_cameras = []
    print("Scanning for available cameras...")
    for idx in range(10):
        capture = cv2.VideoCapture(idx)
        if capture.isOpened():
            success, _ = capture.read()
            if success:
                backend_name = capture.getBackendName()
                found_cameras.append((idx, backend_name))
                print(f"  Camera {idx}: {backend_name}")
            capture.release()
    return found_cameras

def run_iphone_pipeline():
    analyzer = ChessPipeline()
    
    cam_idx = 0
    if len(sys.argv) > 1:
        try:
            cam_idx = int(sys.argv[1])
        except ValueError:
            print(f"Invalid camera index: {sys.argv[1]}")
            print("Usage: python chess_pipeline_iphone.py [camera_index]")
            print("\nAvailable cameras:")
            scan_cameras()
            return
    
    available_cams = scan_cameras()
    if not available_cams:
        print("No cameras found!")
        print("\nTo use iPhone camera:")
        print("1. Option A - Continuity Camera (macOS):")
        print("   - Connect iPhone via USB or ensure both devices are on same Wi-Fi")
        print("   - iPhone should appear as a camera automatically")
        print("   - Make sure iPhone is unlocked")
        print("\n2. Option B - EpocCam or similar app:")
        print("   - Install EpocCam on iPhone and Mac")
        print("   - Connect via USB or Wi-Fi")
        print("\n3. Option C - Use network camera:")
        print("   - Run: python chess_pipeline_network.py")
        return
    
    print(f"\nUsing camera index {cam_idx}")
    print("If this is not your iPhone, try different indices:")
    for i, backend in available_cams:
        print(f"  python chess_pipeline_iphone.py {i}")
    
    video_stream = cv2.VideoCapture(cam_idx)
    
    if not video_stream.isOpened():
        print(f"Error: Could not open camera {cam_idx}")
        return
    
    video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    video_stream.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    print("\n" + "="*50)
    print("iPhone Camera Chess Pipeline")
    print("="*50)
    print("Press 'S' or SPACE to capture and process frame")
    print("Press 'Q' or ESC to quit")
    print("="*50 + "\n")
    
    capture_count = 0
    
    while True:
        success, current_frame = video_stream.read()
        if not success:
            print("Error: Could not read frame from camera")
            break
        
        cv2.imshow('iPhone Camera - Press S to capture', current_frame)
        
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q') or key_pressed == 27:
            break
        if key_pressed == ord('s') or key_pressed == ord(' '):
            capture_count += 1
            print(f"\n[{capture_count}] Capturing frame...")
            print("-" * 50)
            
            try:
                result = analyzer.analyze_frame(current_frame)
                print("-" * 50)
                print(f"Frame {capture_count} processed successfully!")
                
                cv2.imshow('Processed Frame', result)
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyWindow('Processed Frame')
                
            except Exception as err:
                print(f"Error processing frame: {err}")
                import traceback
                traceback.print_exc()
    
    video_stream.release()
    cv2.destroyAllWindows()
    print("\nCamera released. Goodbye!")

def main():
    run_iphone_pipeline()

if __name__ == "__main__":
    main()
