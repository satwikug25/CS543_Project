import cv2
import os
import sys
from chess_pipeline import ChessPipeline

def list_available_cameras():
    """List all available camera devices"""
    available_cameras = []
    print("Scanning for available cameras...")
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                # Try to get camera name if possible
                backend = cap.getBackendName()
                available_cameras.append((i, backend))
                print(f"  Camera {i}: {backend}")
            cap.release()
    return available_cameras

def main():
    pipeline = ChessPipeline()
    
    # Check for command line argument for camera index
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print(f"Invalid camera index: {sys.argv[1]}")
            print("Usage: python chess_pipeline_iphone.py [camera_index]")
            print("\nAvailable cameras:")
            list_available_cameras()
            return
    
    # List available cameras
    cameras = list_available_cameras()
    if not cameras:
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
    
    print(f"\nUsing camera index {camera_index}")
    print("If this is not your iPhone, try different indices:")
    for idx, backend in cameras:
        print(f"  python chess_pipeline_iphone.py {idx}")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    print("\n" + "="*50)
    print("iPhone Camera Chess Pipeline")
    print("="*50)
    print("Press 'S' or SPACE to capture and process frame")
    print("Press 'Q' or ESC to quit")
    print("="*50 + "\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Display the frame
        cv2.imshow('iPhone Camera - Press S to capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        if key == ord('s') or key == ord(' '):  # 's' or SPACE
            frame_count += 1
            print(f"\n[{frame_count}] Capturing frame...")
            print("-" * 50)
            
            # Process the frame
            try:
                out = pipeline.process_frame(frame)
                print("-" * 50)
                print(f"Frame {frame_count} processed successfully!")
                
                # Show processed frame
                cv2.imshow('Processed Frame', out)
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyWindow('Processed Frame')
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                import traceback
                traceback.print_exc()
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera released. Goodbye!")

if __name__ == "__main__":
    main()



