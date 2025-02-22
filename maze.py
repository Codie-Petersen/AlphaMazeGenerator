import random

# Define wall combinations based on your rules
WALL_COMBINATIONS = {
    0: "no_wall",
    1: "up_wall",
    2: "down_wall",
    3: "up_down_wall",
    4: "left_wall",
    5: "up_left_wall",
    6: "down_left_wall",
    7: "up_down_left_wall",
    8: "right_wall",
    9: "up_right_wall",
    10: "down_right_wall",
    11: "up_down_right_wall",
    12: "left_right_wall",
    13: "up_left_right_wall",
    14: "down_left_right_wall",
    15: "up_down_left_right_wall"
}

def break_wall(grid, x, y, nx, ny):
    """Break walls between two cells."""
    if nx == x - 1:  # Moving up
        grid[x][y] -= 1
        grid[nx][ny] -= 2
    elif nx == x + 1:  # Moving down
        grid[x][y] -= 2
        grid[nx][ny] -= 1
    elif ny == y - 1:  # Moving left
        grid[x][y] -= 4
        grid[nx][ny] -= 8
    elif ny == y + 1:  # Moving right
        grid[x][y] -= 8
        grid[nx][ny] -= 4

def recursive_backtracking(grid, x, y):
    """Recursive backtracking algorithm to generate a maze."""
    directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # Left, Up, Right, Down
    random.shuffle(directions)  # Randomize directions to ensure variety

    for dx, dy in directions:
        nx, ny = x + dx, y + dy

        # Check if the neighbor is within bounds and unvisited (all walls intact)
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 15:
            break_wall(grid, x, y, nx, ny)   # Break the wall between current cell and neighbor
            recursive_backtracking(grid, nx, ny)   # Recur on the neighbor

def generate_maze(rows, cols):
    """Generate a maze using recursive backtracking."""
    # Initialize the grid with all walls (15 means all walls are present)
    grid = [[15 for _ in range(cols)] for _ in range(rows)]

    # Start maze generation from a random cell
    start_x, start_y = random.randint(0, rows - 1), random.randint(0, cols - 1)
    recursive_backtracking(grid, start_x, start_y)

    # Select origin and target points
    origin_coords = (start_x, start_y)
    
    # Ensure the target is far from the origin for better paths
    while True:
        target_x = random.randint(0, rows - 1)
        target_y = random.randint(0, cols - 1)
        if abs(target_x - start_x) + abs(target_y - start_y) > max(rows // 2, cols // 2):
            target_coords = (target_x, target_y)
            break

    return grid, origin_coords, target_coords

def convert_to_alpha_format(grid, origin_coords, target_coords):
    """Convert the maze grid into the dumb format."""
    rows, cols = len(grid), len(grid[0])
    
    def cell_to_walls(cell_value):
        """Convert a cell's integer value to its wall type."""
        return WALL_COMBINATIONS[cell_value]

    alpha_format = ""
    
    for r in range(rows):
        for c in range(cols):
            coord = f"<|{r}-{c}|>"
            
            wall_type = cell_to_walls(grid[r][c])
            
            # Add origin or target markers **AFTER** the wall type
            if (r, c) == origin_coords:
                marker = "<|origin|>"
            elif (r, c) == target_coords:
                marker = "<|target|>"
            else:
                marker = "<|blank|>"
            
            alpha_format += f"{coord}<|{wall_type}|>{marker}"
        
        alpha_format += "\n"
    
    return alpha_format



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a maze.")

    parser.add_argument("-r", "--rows", type=int, default=5, help="Number of rows in the maze.")
    parser.add_argument("-c", "--cols", type=int, default=5, help="Number of columns in the maze.")
    parser.add_argument("--rc", "--cr", type=int, default=0, help="Square size for the maze.")
    args = parser.parse_args()
    
    if args.rc != 0:
        args.rows = args.rc
        args.cols = args.rc

    maze_grid, origin_coords_finalized , target_coords_finalized = generate_maze(args.rows, args.cols)
    dumb_maze_output = convert_to_alpha_format(maze_grid ,origin_coords_finalized ,target_coords_finalized)

print(dumb_maze_output)
