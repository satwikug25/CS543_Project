import cv2
import os
from chess_pipeline import ChessPipeline

def main():
    pipeline = ChessPipeline()
    
    # Load image from file instead of webcam
    image_path = "chesstest.jpg"  # Change this to any image file you want to process
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        print("Available image files in the directory:")
        for file in os.listdir("."):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"  - {file}")
        return
    
    print(f"Loading image from: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not load image from '{image_path}'")
        return
    
    print("Processing image...")
    out = pipeline.process_frame(frame)
    
    # Display the processed frame
    cv2.imshow('Processed Image', out)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Processing complete!")

if __name__ == "__main__":
    main()

