class BoxCoordinates:
    def __init__(self):
        # Initialize coordinates for each square
        # Format: (row, col) where:
        # row: 0-7 (bottom to top)
        # col: 0-7 (right to left)
        self.coordinates = {
            'h1': (0, 0),    # Bottom right
            'g1': (0, 1),
            'f1': (0, 2),
            'e1': (0, 3),
            'd1': (0, 4),
            'c1': (0, 5),
            'b1': (0, 6),
            'a1': (0, 7),    # Bottom left
            
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
            'a8': (7, 7),    # Top left
        }
    
    def get_coordinates(self, square):
        """Get the matrix coordinates for a given chess square."""
        if square not in self.coordinates:
            raise ValueError(f"Invalid square: {square}")
        return self.coordinates[square]
    
    def get_move_coordinates(self, from_square, to_square):
        """Get the coordinates for a complete move."""
        from_coords = self.get_coordinates(from_square)
        to_coords = self.get_coordinates(to_square)
        return from_coords, to_coords

# Example usage
if __name__ == "__main__":
    coords = BoxCoordinates()
    
    # Test some moves
    test_moves = [
        ("e2", "e4"),  # Common opening move
        ("e7", "e5"),  # Common response
        ("g1", "f3"),  # Knight development
    ]
    
    for from_square, to_square in test_moves:
        from_coords, to_coords = coords.get_move_coordinates(from_square, to_square)
        print(f"Move {from_square} to {to_square}:")
        print(f"  From: {from_coords}")
        print(f"  To: {to_coords}") 