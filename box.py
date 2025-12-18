class BoxCoordinates:
    def __init__(self):
        self.position_map = {
            'h1': (0, 0),
            'g1': (0, 1),
            'f1': (0, 2),
            'e1': (0, 3),
            'd1': (0, 4),
            'c1': (0, 5),
            'b1': (0, 6),
            'a1': (0, 7),
            
            'h2': (1, 0),
            'g2': (1, 1),
            'f2': (1, 2),
            'e2': (1, 3),
            'd2': (1, 4),
            'c2': (1, 5),
            'b2': (1, 6),
            'a2': (1, 7),
            
            'h3': (2, 0),
            'g3': (2, 1),
            'f3': (2, 2),
            'e3': (2, 3),
            'd3': (2, 4),
            'c3': (2, 5),
            'b3': (2, 6),
            'a3': (2, 7),
            
            'h4': (3, 0),
            'g4': (3, 1),
            'f4': (3, 2),
            'e4': (3, 3),
            'd4': (3, 4),
            'c4': (3, 5),
            'b4': (3, 6),
            'a4': (3, 7),
            
            'h5': (4, 0),
            'g5': (4, 1),
            'f5': (4, 2),
            'e5': (4, 3),
            'd5': (4, 4),
            'c5': (4, 5),
            'b5': (4, 6),
            'a5': (4, 7),
            
            'h6': (5, 0),
            'g6': (5, 1),
            'f6': (5, 2),
            'e6': (5, 3),
            'd6': (5, 4),
            'c6': (5, 5),
            'b6': (5, 6),
            'a6': (5, 7),
            
            'h7': (6, 0),
            'g7': (6, 1),
            'f7': (6, 2),
            'e7': (6, 3),
            'd7': (6, 4),
            'c7': (6, 5),
            'b7': (6, 6),
            'a7': (6, 7),
            
            'h8': (7, 0),
            'g8': (7, 1),
            'f8': (7, 2),
            'e8': (7, 3),
            'd8': (7, 4),
            'c8': (7, 5),
            'b8': (7, 6),
            'a8': (7, 7),
        }
    
    def get_coordinates(self, square_name):
        if square_name not in self.position_map:
            raise ValueError(f"Invalid square: {square_name}")
        return self.position_map[square_name]
    
    def get_move_coordinates(self, origin_square, target_square):
        origin_pos = self.get_coordinates(origin_square)
        target_pos = self.get_coordinates(target_square)
        return origin_pos, target_pos

if __name__ == "__main__":
    coord_helper = BoxCoordinates()
    
    sample_moves = [
        ("e2", "e4"),
        ("e7", "e5"),
        ("g1", "f3"),
    ]
    
    for origin, target in sample_moves:
        origin_coords, target_coords = coord_helper.get_move_coordinates(origin, target)
        print(f"Move {origin} to {target}:")
        print(f"  From: {origin_coords}")
        print(f"  To: {target_coords}")
