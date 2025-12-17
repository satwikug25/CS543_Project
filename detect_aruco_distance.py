#!/usr/bin/env python3
"""
detect_and_display.py

1. Load camera intrinsics & distortion from pickles
2. Detect ArUco markers, pick the one board‐corner from each
3. Draw an 8×8 grid on the original image (for visualization only)
4. Compute and print 3D distances between two corners (cm)
5. Warp both the “grid-overlay” and “no-grid” versions to a flat, top-down view
6. On the warped “grid” image overlay the 2.2″ grid and save it
7. Slice the warped “no-grid” image into clean cells
8. Display a selected isolated cell in a window
"""

import os
import sys
import pickle
import cv2
import numpy as np

# ——— Configuration ———
IMAGE_PATH       = "/Users/keval/Documents/VSCode/ChessBot/chesstest.jpg"
CAM_MAT_PKL      = "cameraMatrix.pkl"
DIST_COEFFS_PKL  = "dist.pkl"

MARKER_LENGTH_CM = 5       # marker side length in cm
MARKER_LENGTH_M  = MARKER_LENGTH_CM / 100.0

OUTPUT_IMAGE     = "flattened_grid.png"
PIECE_FOLDER     = "grid_piece"

MARKER_ID_A      = 0
MARKER_ID_B      = 1

DISPLAY_ROW      = 0
DISPLAY_COL      = 0
# ————————————————————

class FourMarkerPose:
    def __init__(self, marker_length_m):
        with open(CAM_MAT_PKL, 'rb') as f:
            self.camera_matrix = np.asarray(pickle.load(f), dtype=float)
        with open(DIST_COEFFS_PKL, 'rb') as f:
            self.dist_coeffs  = np.asarray(pickle.load(f), dtype=float)

        self.marker_length = marker_length_m
        self.aruco_dict    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters    = cv2.aruco.DetectorParameters()

        # marker ID → which corner is the board’s corner?
        self.corner_map = {0:2, 1:3, 2:0, 3:1}

    def detect_and_compute(self, img):
        corners_list, ids, _ = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.parameters
        )
        if ids is None:
            print("Error: no markers detected.", file=sys.stderr)
            return {}, None, None, None, None

        ids = ids.flatten()
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners_list, self.marker_length,
            self.camera_matrix, self.dist_coeffs
        )

        half = self.marker_length / 2.0
        obj_corners = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ])

        results = {}
        for idx, mid in enumerate(ids):
            mid = int(mid)
            if mid not in self.corner_map:
                continue

            pix = corners_list[idx].reshape(4,2)
            bidx = self.corner_map[mid]
            px, py = pix[bidx]

            R, _   = cv2.Rodrigues(rvecs[idx])
            obj_pt = obj_corners[bidx].reshape(3,1)
            cam_pt = (R @ obj_pt + tvecs[idx].reshape(3,1)).flatten()

            results[mid] = {'pixel': (px, py), 'camera_pt_m': cam_pt}

        return results, corners_list, ids, rvecs, tvecs

def draw_grid(img, src_pts, grid_size=8, color=(0,255,0), thickness=2):
    tl, tr, br, bl = src_pts
    for i in range(1, grid_size):
        α = i / float(grid_size)
        # vertical
        start = tuple((tl + α*(tr - tl)).astype(int))
        end   = tuple((bl + α*(br - bl)).astype(int))
        cv2.line(img, start, end, color, thickness)
        # horizontal
        start = tuple((tl + α*(bl - tl)).astype(int))
        end   = tuple((tr + α*(br - tr)).astype(int))
        cv2.line(img, start, end, color, thickness)

def main():
    # 1) Load image and keep a no-grid copy
    img_grid      = cv2.imread(IMAGE_PATH)
    if img_grid is None:
        print(f"Error: could not read `{IMAGE_PATH}`", file=sys.stderr)
        sys.exit(1)
    img_no_grid   = img_grid.copy()

    # 2) Detect markers & get pixel corners
    estimator, corners, raw_corners, ids = FourMarkerPose(MARKER_LENGTH_M), *([None]*3)
    corners, raw_corners, ids, _, _ = estimator.detect_and_compute(img_no_grid)
    if not corners:
        sys.exit(1)

    # assemble src_pts in TL,TR,BR,BL
    src_pts = np.zeros((4,2), dtype=np.float32)
    for mid, info in corners.items():
        idx = estimator.corner_map[mid]
        src_pts[idx] = info['pixel']

    # 3) Draw green 8×8 on img_grid only
    draw_grid(img_grid, src_pts, grid_size=8)
    cv2.imwrite("grid_on_original.png", img_grid)
    print("Grid overlay saved as 'grid_on_original.png'")

    # 4) Print 3D distances...
    print("\nBoard-corners (pixel → 3D-cm):")
    for mid, d in corners.items():
        px, py = d['pixel']
        cx, cy, cz = d['camera_pt_m'] * 100
        print(f" Marker {mid}: pixel=({px:.1f},{py:.1f}) → ({cx:.1f},{cy:.1f},{cz:.1f}) cm")

    if MARKER_ID_A in corners and MARKER_ID_B in corners:
        a = corners[MARKER_ID_A]['camera_pt_m']
        b = corners[MARKER_ID_B]['camera_pt_m']
        dist = np.linalg.norm(a - b) * 100
        print(f"Distance {MARKER_ID_A}↔{MARKER_ID_B}: {dist:.2f} cm")
    else:
        print(f"Need markers {MARKER_ID_A} & {MARKER_ID_B}, found {list(corners)}", file=sys.stderr)

    # 5) Compute grid-pixel size
    pixel_per_m = None
    for idx, mid in enumerate(ids):
        if int(mid)==MARKER_ID_A:
            pts = raw_corners[idx].reshape(4,2)
            pixel_per_m = np.linalg.norm(pts[0]-pts[1]) / MARKER_LENGTH_M
            break
    if pixel_per_m is None:
        print("Error: cannot compute pixel_per_m", file=sys.stderr)
        sys.exit(1)

    PIX_PER_INCH = pixel_per_m * 0.0254
    GRID_PIX     = int(round(2.2 * PIX_PER_INCH))
    size         = GRID_PIX * 8

    # 6) Warp both versions
    dst_pts = np.array([[0,0],[size,0],[size,size],[0,size]],dtype=np.float32)
    H       = cv2.getPerspectiveTransform(src_pts, dst_pts)

    flat_grid   = cv2.warpPerspective(img_grid,    H, (size,size))
    flat_nogrid = cv2.warpPerspective(img_no_grid, H, (size,size))

    # on flat_grid draw blue 2.2″ lines
    for i in range(9):
        x = i * GRID_PIX
        cv2.line(flat_grid,   (x,0),      (x,size), (255,0,0), 1)
        cv2.line(flat_grid,   (0,x),      (size,x), (255,0,0), 1)

    # 7) save the flattened grid with lines
    cv2.imwrite(OUTPUT_IMAGE, flat_grid)
    print(f"\nFlattened grid saved as '{OUTPUT_IMAGE}'")

    # 8) slice flat_nogrid into cells
    if not os.path.exists(PIECE_FOLDER):
        os.makedirs(PIECE_FOLDER)
    for r in range(8):
        for c in range(8):
            y0,y1 = r*GRID_PIX,(r+1)*GRID_PIX
            x0,x1 = c*GRID_PIX,(c+1)*GRID_PIX
            cell = flat_nogrid[y0:y1, x0:x1]
            cv2.imwrite(os.path.join(PIECE_FOLDER, f"piece_r{r}_c{c}.png"), cell)
    print(f"All 8×8 grid cells saved in '{PIECE_FOLDER}'")

    # display one
    fname = f"piece_r{DISPLAY_ROW}_c{DISPLAY_COL}.png"
    piece = cv2.imread(os.path.join(PIECE_FOLDER,fname))
    if piece is None:
        print(f"Error: cannot load '{fname}'", file=sys.stderr)
        sys.exit(1)
    cv2.imshow(f"Cell {DISPLAY_ROW},{DISPLAY_COL}", piece)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
