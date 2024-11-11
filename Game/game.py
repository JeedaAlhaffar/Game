import numpy as np
import copy

class Square:
    def __init__(self, color, start_pos, goal_pos):
        self.color = color
        self.x, self.y = start_pos
        self.goal_x, self.goal_y = goal_pos
        self.active = True  


class GridGame:
    def __init__(self, grid):
        self.original_grid = grid  
        self.grid = copy.deepcopy(grid)  
        self.rows, self.cols = self.grid.shape

        self.history = [copy.deepcopy(self.grid)]
        
        self.squares = []
        for color in ['R', 'B', 'G', 'Y']:  
            pos = np.argwhere(self.grid == color)
            goal = np.argwhere(self.grid == color.lower())
            if len(pos) > 0 and len(goal) > 0:
                start_pos = tuple(pos[0])

                goal_pos = tuple(goal[0])
                self.squares.append(Square(color, start_pos, goal_pos))

    def print_grid(self):
        for row in self.grid:
            print(" ".join(row))
        print("\n")


    def is_within_bounds(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols

    def can_move(self, x, y, dx, dy, occupied_positions):
        new_x, new_y = x + dx, y + dy
        return (self.is_within_bounds(new_x, new_y) and
                self.grid[new_x][new_y] != '#' and
                (new_x, new_y) not in occupied_positions)

    def get_possible_moves(self):
        directions = {
            'w': (-1, 0),  
            's': (1, 0),   
            'a': (0, -1),  
            'd': (0, 1)    
        }

        possible_moves = set()
        occupied_positions = {(sq.x, sq.y) for sq in self.squares if sq.active}
        
        for square in self.squares:
            if square.active:
                for direction, (dx, dy) in directions.items():

                    if self.can_move(square.x, square.y, dx, dy, occupied_positions):
                        possible_moves.add(direction)

        return list(possible_moves)

    def move_square(self, square, dx, dy, occupied_positions):
        x, y = square.x, square.y
        while self.is_within_bounds(x + dx, y + dy) and \
              self.grid[x + dx][y + dy] != '#' and \
              (x + dx, y + dy) not in occupied_positions:
            x += dx
            y += dy
            if (x, y) == (square.goal_x, square.goal_y):
                break
        return x, y


    def move_both(self, direction):
        dx, dy = 0, 0
        if direction == 'w': dx, dy = -1, 0
        elif direction == 's': dx, dy = 1, 0
        elif direction == 'a': dx, dy = 0, -1
        elif direction == 'd': dx, dy = 0, 1

        occupied_positions = {(sq.x, sq.y) for sq in self.squares if sq.active}
        goals_reached = 0

        for square in self.squares:
            if square.active:
                new_x, new_y = self.move_square(square, dx, dy, occupied_positions)
                
                if (new_x, new_y) != (square.x, square.y):
                    self.grid[square.x][square.y] = ' '
                    square.x, square.y = new_x, new_y
                    occupied_positions.add((new_x, new_y))
                    if (square.x, square.y) == (square.goal_x, square.goal_y):
                        square.active = False
                        self.grid[square.goal_x][square.goal_y] = ' '
                        print(f"{square.color} reached its goal and disappeared!")
                        goals_reached += 1

                    if square.active:
                        self.grid[square.x][square.y] = square.color

        self.history.append(copy.deepcopy(self.grid))

        if all(not square.active for square in self.squares):

            print("Congratulations, all squares reached their goals!")
            return True

        return False

    def get_history(self):
        return self.history
    
    def generate_next_states(self):
        next_states = []

        possible_moves = self.get_possible_moves()
        

        for move in possible_moves:
            new_game = copy.deepcopy(self)
            new_game.move_both(move)
            next_states.append((move, new_game.grid))
        
        return next_states


grid1 = np.array([
    [' ', '#', '#', '#', '#', '#', ' ', ' ', ' ', ' ', ' '],
    ['#', '#', 'R', ' ', ' ', '#', '#', '#', '#', '#', ' '],
    ['#', ' ', ' ', ' ', ' ', '#', '#', 'b', ' ', '#', ' '],
    ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', '#'],
    ['#', ' ', ' ', ' ', '#', '#', '#', ' ', ' ', 'r', '#'],
    ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', '#'],
    ['#', '#', 'B', ' ', '#', '#', '#', '#', '#', '#', ' '],
    [' ', '#', '#', '#', '#', ' ', ' ', ' ', ' ', ' ', ' ']
])



grid3 = np.array([
    [' ', ' ', ' ', ' ', '#', '#', '#', '#', '#'],
    [' ', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#'],
    ['#', '#', '#', '#', '#', 'r', '#', ' ', '#'],
    ['#', ' ', ' ', ' ', 'R', ' ', '#', ' ', '#'],
    ['#', ' ', ' ', 'b', '#', 'B', '#', ' ', '#'],
    ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
    ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
    ['#', '#', '#', '#', '#', '#', '#', '#', '#']
])


grid4 = np.array([
    ['#', '#', '#', '#', '#', '#', ' ', ' ', ' '],
    ['#', 'r', ' ', ' ', ' ', '#', '#', ' ', ' '],
    ['#', ' ', ' ', 'b', ' ', ' ', '#', '#', '#'],
    ['#', ' ', ' ', ' ', ' ', ' ', 'B', 'R', '#'],
    ['#', '#', '#', '#', '#', '#', '#', '#', '#']
])
grid5 = np.array([
    [' ', '#', '#', '#', '#', '#', ' ', ' ', ' ', ' ', ' '],
    ['#', '#', 'R', ' ', ' ', '#', '#', '#', '#', '#', ' '],
    ['#', ' ', ' ', ' ', ' ', '#', '#', 'b', ' ', '#', ' '],
    ['#', ' ', 'G', ' ', ' ', ' ', ' ', ' ', 'g', '#', '#'],
    ['#', ' ', ' ', ' ', '#', '#', '#', ' ', ' ', 'r', '#'],
    ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#', '#'],
    ['#', '#', 'B', ' ', '#', '#', '#', '#', '#', '#', ' '],
    [' ', '#', '#', '#', '#', ' ', ' ', ' ', ' ', ' ', ' ']
])

game = GridGame(grid5)

next_states = game.generate_next_states()

for move, state in next_states:


    print(f"Move: {move}")

    for row in state:
        print(" ".join(row))
    print("\n")


