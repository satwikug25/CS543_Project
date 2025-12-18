import cv2
import numpy as np
import os
from chess_pipeline import ChessPipeline

def generate_report_images(source_img_path, dest_folder="report_images"):
    
    os.makedirs(dest_folder, exist_ok=True)
    
    print(f"Loading image: {source_img_path}")
    source_img = cv2.imread(source_img_path)
    if source_img is None:
        print(f"Error: Could not load {source_img_path}")
        return
    
    print("Stage 1: Saving original frame...")
    cv2.imwrite(f"{dest_folder}/stage1_original.png", source_img)
    
    analyzer = ChessPipeline()
    
    print("Stage 2: Detecting markers...")
    corner_pts, marker_corners, marker_ids = analyzer.find_board_markers(source_img)
    
    if corner_pts is None:
        print("No markers detected! Cannot proceed.")
        return
    
    annotated_img = source_img.copy()
    analyzer.render_grid_overlay(annotated_img, corner_pts)
    
    if marker_ids is not None:
        marker_ids = marker_ids.flatten()
        for i, corners in enumerate(marker_corners):
            corners = corners.reshape(4, 2).astype(int)
            cv2.polylines(annotated_img, [corners], True, (0, 255, 0), 2)
            if i < len(marker_ids):
                cv2.putText(annotated_img, str(marker_ids[i]), 
                           tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
    
    cv2.imwrite(f"{dest_folder}/stage2_markers.png", annotated_img)
    
    print("Stage 3: Computing perspective transformation...")
    px_per_m = np.linalg.norm(corner_pts[0] - corner_pts[1]) / analyzer.tag_size
    px_per_inch = px_per_m * 0.0254
    cell_size = int(round(2.2 * px_per_inch))
    board_size = cell_size * 8
    
    target_pts = np.array([[0,0],[board_size,0],[board_size,board_size],[0,board_size]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corner_pts, target_pts)
    warped_img = cv2.warpPerspective(source_img, transform, (board_size,board_size))
    
    cv2.imwrite(f"{dest_folder}/stage3_warped.png", warped_img)
    
    print("Stage 4: Creating grid overlay...")
    cv2.imwrite(f"{dest_folder}/stage4_grid.png", annotated_img)
    
    print("Stage 5: Processing classification (this may take a while)...")
    try:
        analyzer.analyze_frame(source_img)
        
        import json
        with open('detection.json', 'r') as fh:
            detection_data = json.load(fh)
        
        if detection_data:
            latest_detection = detection_data[-1]
            visualization = warped_img.copy()
            sq_size = board_size // 8
            
            for entry in latest_detection['results']:
                r, c = entry['r'], entry['c']
                lbl = entry['label']
                
                x_pos = c * sq_size
                y_pos = r * sq_size
                
                if lbl == 'empty':
                    draw_color = (128, 128, 128)
                elif lbl == 'object':
                    draw_color = (255, 255, 255)
                elif lbl == 'object-black':
                    draw_color = (0, 0, 0)
                else:
                    draw_color = (0, 0, 255)
                
                cv2.rectangle(visualization, (x_pos, y_pos), (x_pos+sq_size, y_pos+sq_size), draw_color, 2)
                cv2.putText(visualization, lbl[:5], (x_pos+5, y_pos+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)
            
            cv2.imwrite(f"{dest_folder}/stage5_classification.png", visualization)
    except Exception as err:
        print(f"Warning: Could not create classification visualization: {err}")
        cv2.imwrite(f"{dest_folder}/stage5_classification.png", warped_img)
    
    print("Stage 6: Creating move detection visualization...")
    move_visualization = annotated_img.copy()
    
    cv2.putText(move_visualization, "Move Detection Ready", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imwrite(f"{dest_folder}/stage6_move.png", move_visualization)
    
    print("Creating grid squares example...")
    if os.path.exists("grid_piece"):
        collage_dim = 400
        cell_dim = collage_dim // 4
        collage_img = np.zeros((collage_dim, collage_dim, 3), dtype=np.uint8)
        
        sample_squares = [
            (2, 0, "empty"),
            (6, 0, "white piece"),
            (6, 4, "black piece"),
            (0, 0, "white piece"),
        ]
        
        for idx, (r, c, description) in enumerate(sample_squares):
            img_file = f"grid_piece/piece_r{r}_c{c}.png"
            if os.path.exists(img_file):
                sq_img = cv2.imread(img_file)
                if sq_img is not None:
                    resized = cv2.resize(sq_img, (cell_dim, cell_dim))
                    grid_row = idx // 2
                    grid_col = idx % 2
                    y = grid_row * cell_dim
                    x = grid_col * cell_dim
                    collage_img[y:y+cell_dim, x:x+cell_dim] = resized
                    
                    cv2.putText(collage_img, description, (x+5, y+cell_dim-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        cv2.imwrite(f"{dest_folder}/grid_squares_example.png", collage_img)
    
    print(f"\n✓ All images saved to {dest_folder}/")
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
    
    img_file = sys.argv[1]
    generate_report_images(img_file)
