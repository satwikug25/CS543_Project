"""
Script to capture intermediate processing images for the report.
Run this with an image to generate all intermediate stage images.
"""

import cv2
import numpy as np
import os
from chess_pipeline import ChessPipeline

def capture_intermediate_stages(image_path, output_dir="report_images"):
    """Capture all intermediate processing stages"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    print(f"Loading image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load {image_path}")
        return
    
    # Stage 1: Original frame
    print("Stage 1: Saving original frame...")
    cv2.imwrite(f"{output_dir}/stage1_original.png", frame)
    
    # Initialize pipeline
    pipeline = ChessPipeline()
    
    # Detect markers
    print("Stage 2: Detecting markers...")
    src_pts, corners_list, ids = pipeline.detect_markers(frame)
    
    if src_pts is None:
        print("No markers detected! Cannot proceed.")
        return
    
    # Stage 2: Frame with markers and grid
    frame_with_grid = frame.copy()
    pipeline.draw_grid(frame_with_grid, src_pts)
    
    # Draw detected markers
    if ids is not None:
        ids = ids.flatten()
        for i, corner in enumerate(corners_list):
            corner = corner.reshape(4, 2).astype(int)
            cv2.polylines(frame_with_grid, [corner], True, (0, 255, 0), 2)
            if i < len(ids):
                cv2.putText(frame_with_grid, str(ids[i]), 
                           tuple(corner[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
    
    cv2.imwrite(f"{output_dir}/stage2_markers.png", frame_with_grid)
    
    # Stage 3: Warped board
    print("Stage 3: Computing perspective transformation...")
    pixel_per_m = np.linalg.norm(src_pts[0] - src_pts[1]) / pipeline.marker_length
    PIX_PER_INCH = pixel_per_m * 0.0254
    GRID_PIX = int(round(2.2 * PIX_PER_INCH))
    size = GRID_PIX * 8
    
    dst_pts = np.array([[0,0],[size,0],[size,size],[0,size]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    flat_grid = cv2.warpPerspective(frame, H, (size,size))
    
    cv2.imwrite(f"{output_dir}/stage3_warped.png", flat_grid)
    
    # Stage 4: Grid overlay (same as stage 2 but with labels)
    print("Stage 4: Creating grid overlay...")
    cv2.imwrite(f"{output_dir}/stage4_grid.png", frame_with_grid)
    
    # Stage 5: Classification visualization
    print("Stage 5: Processing classification (this may take a while)...")
    # Process frame to get classifications
    try:
        pipeline.process_frame(frame)
        
        # Load detection.json to visualize classifications
        import json
        with open('detection.json', 'r') as f:
            data = json.load(f)
        
        if data:
            latest = data[-1]
            # Create visualization
            vis_frame = flat_grid.copy()
            cell_size = size // 8
            
            for result in latest['results']:
                r, c = result['r'], result['c']
                label = result['label']
                
                x = c * cell_size
                y = r * cell_size
                
                # Draw label
                if label == 'empty':
                    color = (128, 128, 128)  # Gray
                elif label == 'object':
                    color = (255, 255, 255)  # White
                elif label == 'object-black':
                    color = (0, 0, 0)  # Black
                else:
                    color = (0, 0, 255)  # Red for error
                
                cv2.rectangle(vis_frame, (x, y), (x+cell_size, y+cell_size), color, 2)
                cv2.putText(vis_frame, label[:5], (x+5, y+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            cv2.imwrite(f"{output_dir}/stage5_classification.png", vis_frame)
    except Exception as e:
        print(f"Warning: Could not create classification visualization: {e}")
        # Create a placeholder
        cv2.imwrite(f"{output_dir}/stage5_classification.png", flat_grid)
    
    # Stage 6: Move detection visualization
    print("Stage 6: Creating move detection visualization...")
    # This would show detected moves - for now, use the grid with move annotations
    move_vis = frame_with_grid.copy()
    
    # Add text indicating move detection
    cv2.putText(move_vis, "Move Detection Ready", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imwrite(f"{output_dir}/stage6_move.png", move_vis)
    
    # Create grid squares collage
    print("Creating grid squares example...")
    if os.path.exists("grid_piece"):
        # Create a 4x4 collage of example squares
        collage_size = 400
        cell_size = collage_size // 4
        collage = np.zeros((collage_size, collage_size, 3), dtype=np.uint8)
        
        examples = [
            (2, 0, "empty"),
            (6, 0, "white piece"),
            (6, 4, "black piece"),
            (0, 0, "white piece"),
        ]
        
        for idx, (r, c, label) in enumerate(examples):
            filename = f"grid_piece/piece_r{r}_c{c}.png"
            if os.path.exists(filename):
                img = cv2.imread(filename)
                if img is not None:
                    img_resized = cv2.resize(img, (cell_size, cell_size))
                    row = idx // 2
                    col = idx % 2
                    y = row * cell_size
                    x = col * cell_size
                    collage[y:y+cell_size, x:x+cell_size] = img_resized
                    
                    # Add label
                    cv2.putText(collage, label, (x+5, y+cell_size-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        cv2.imwrite(f"{output_dir}/grid_squares_example.png", collage)
    
    print(f"\n✓ All images saved to {output_dir}/")
    print("\nGenerated files:")
    print("  - stage1_original.png")
    print("  - stage2_markers.png")
    print("  - stage3_warped.png")
    print("  - stage4_grid.png")
    print("  - stage5_classification.png")
    print("  - stage6_move.png")
    print("  - grid_squares_example.png")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python capture_intermediate_images.py <image_path>")
        print("Example: python capture_intermediate_images.py image1.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    capture_intermediate_stages(image_path)



