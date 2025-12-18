import cv2
import os
from chess_pipeline import ChessPipeline

def process_static_image():
    analyzer = ChessPipeline()
    
    input_file = "chesstest.jpg"
    
    if not os.path.exists(input_file):
        print(f"Error: Image file '{input_file}' not found!")
        print("Available image files in the directory:")
        for f in os.listdir("."):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"  - {f}")
        return
    
    print(f"Loading image from: {input_file}")
    input_img = cv2.imread(input_file)
    
    if input_img is None:
        print(f"Error: Could not load image from '{input_file}'")
        return
    
    print("Processing image...")
    result_img = analyzer.analyze_frame(input_img)
    
    cv2.imshow('Processed Image', result_img)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Processing complete!")

def main():
    process_static_image()

if __name__ == "__main__":
    main()
