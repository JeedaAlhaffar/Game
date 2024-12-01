import numpy as np
import copy
from collections import deque
import heapq

class Square:
    def __init__(self, color, start_pos, goal_pos):
        self.color = color
        self.x, self.y = start_pos
        self.goal_x, self.goal_y = goal_pos
        self.active = True  


class Gridgame:
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


    def is_withinBound(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols

    def can_move(self, x, y, dx, dy, occupied_positions):
        new_x, new_y = x + dx, y + dy
        return (self.is_withinBound(new_x, new_y) and
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
        while self.is_withinBound(x + dx, y + dy) and \
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
                        goals_reached += 1

                    if square.active:
                        self.grid[square.x][square.y] = square.color

        self.history.append(copy.deepcopy(self.grid))
        


        if all(not square.active for square in self.squares):
            
            return True

        return False

    def generate_next_states(self):
        next_states = []

        possible_moves = self.get_possible_moves()
        

        for move in possible_moves:
            new_game = copy.deepcopy(self)
            new_game.move_both(move)
            next_states.append((move, new_game))
        print(f"Possible moves from current state: {possible_moves}")

        return next_states
    def is_goal(self):
        return all(not square.active for square in self.squares)
    def dfs_solve(self):
        stack = [(self, [])]  
        visited = set()
        states_explored = 0

        while stack:
            current_state, path = stack.pop()
            state_repr = self.grid_repr(current_state.grid)
            if state_repr in visited:
                continue

            visited.add(state_repr)
            states_explored += 1

            if current_state.is_goal():
                print("Solution found!")
                print("Path to goal:")
                for move, state in path:
                    print(f"Move: {move}")
                    for row in state.grid:
                        print(" ".join(row))
                    print("\n")
                print(f"Total states explored: {states_explored}")
                return path

            for move, next_state in current_state.generate_next_states():
                if self.grid_repr(next_state.grid) not in visited:
                    stack.append((next_state, path + [(move, next_state)]))

        print("No solution found.")
        return None
    def grid_repr(self, grid):
        return tuple(tuple(row) for row in grid)
    def bfs_solve(self):
        queue = deque([(self, [])])  
        visited = set()
        states_explored = 0

        while queue:
            current_state, path = queue.popleft()
            state_repr = self.grid_repr(current_state.grid)
            if state_repr in visited:
                continue

            visited.add(state_repr)
            states_explored += 1

            if current_state.is_goal():
                print("Solution found!")
                print("Path to goal:")
                for move, state in path:
                    print(f"Move: {move}")
                    for row in state.grid:
                        print(" ".join(row))
                    print("\n")
                print(f"Total states explored: {states_explored}")
                return path

            for move, next_state in current_state.generate_next_states():
                if self.grid_repr(next_state.grid) not in visited:
                    queue.append((next_state, path + [(move, next_state)]))

        print("No solution found.")
        return None
    def dfs_solve_recursive(self, current_state=None, path=None, visited=None, states_explored=0):
        if current_state is None:
            current_state = self
        if path is None:
            path = []
        if visited is None:
            visited = set()

        state_repr = self.grid_repr(current_state.grid)
        if state_repr in visited:
            return None, states_explored

        visited.add(state_repr)
        states_explored += 1

        if current_state.is_goal():
            print("Solution found!")
            print("Path to goal:")
            for move, state in path:
                print(f"Move: {move}")
                for row in state.grid:
                    print(" ".join(row))
                print("\n")
            print(f"Total states explored: {states_explored}")
            return path, states_explored

        for move, next_state in current_state.generate_next_states():
            if self.grid_repr(next_state.grid) not in visited:
                result_path, result_states = self.dfs_solve_recursive(
                    next_state, path + [(move, next_state)], visited, states_explored
                )
                if result_path is not None:  
                    return result_path, result_states

        return None, states_explored 
    def grid_repr(self, grid):
        return tuple(tuple(row) for row in grid)
        
    def ucs_solve(self):

        priority_queue = []
        heapq.heappush(priority_queue, (0, self, [])) 

        visited = set()
        states_explored = 0

        while priority_queue:
            cost, current_state, path = heapq.heappop(priority_queue)
            state_repr = self.grid_repr(current_state.grid)

            if state_repr in visited:
                continue

            visited.add(state_repr)
            states_explored += 1

            if current_state.is_goal():
                print("Solution found using UCS!")
                print("Path to goal:")
                for move, state in path:
                    print(f"Move: {move}")
                    for row in state.grid:
                        print(" ".join(row))
                    print("\n")
                print(f"Total states explored: {states_explored}")
                return path


            for move, next_state in current_state.generate_next_states():
                if self.grid_repr(next_state.grid) not in visited:
                    new_cost = cost + 1  
                    heapq.heappush(priority_queue, (new_cost, next_state, path + [(move, next_state)]))

        print("No solution found using UCS.")
        return None
    def a_star_solve(self):
        def heuristic(square):
            return abs(square.x - square.goal_x) + abs(square.y - square.goal_y)

        priority_queue = []
        heapq.heappush(priority_queue, (0, 0, id(self), self, []))  

        visited = set()
        states_explored = 0

        while priority_queue:
            f, g, _, current_state, path = heapq.heappop(priority_queue)  
            state_repr = self.grid_repr(current_state.grid)
            print(f"f: {f}")
            print(f"h: {g}")

            if state_repr in visited:
                continue

            visited.add(state_repr)
            states_explored += 1

            if current_state.is_goal():
                print("Solution found using A*!")
                print("Path to goal:")
                for move, state in path:
                    print(f"Move: {move}")
                    for row in state.grid:
                        print(" ".join(row))
                    print("\n")
                print(f"Total states explored: {states_explored}")
                return path

            for move, next_state in current_state.generate_next_states():
                if self.grid_repr(next_state.grid) not in visited:
                    h = sum(heuristic(sq) for sq in next_state.squares if sq.active)
                    new_cost = g + 1 



                    heapq.heappush(priority_queue, (new_cost + h, new_cost, id(next_state), next_state, path + [(move, next_state)]))

        print("No solution found using A*.")
        return None

    def hill_climbing_solve(self):
        def heuristic(square):
            return abs(square.x - square.goal_x) + abs(square.y - square.goal_y)

        def evaluate_state(state):
            return sum(heuristic(sq) for sq in state.squares if sq.active)

        current_state = self
        path = []
        states_explored = 0

        while not current_state.is_goal():
            states_explored += 1
            current_heuristic = evaluate_state(current_state)

            neighbors = current_state.generate_next_states()
            if not neighbors:
                print("No more neighbors to explore. Stuck at a local maximum.")
                break

            best_move, best_state = None, None
            best_heuristic = float('inf')
            
            for move, neighbor in neighbors:
                neighbor_heuristic = evaluate_state(neighbor)
                print(f"neighbor_heuristic: {neighbor_heuristic}")

                if neighbor_heuristic < best_heuristic:
                    best_heuristic = neighbor_heuristic
                    best_move, best_state = move, neighbor

            if best_heuristic >= current_heuristic:
                print("No better neighbor found. Stuck at a local maximum.")
                break
            print(f"best_heuristic: {best_heuristic}")

            path.append((best_move, best_state))
            current_state = best_state

        if current_state.is_goal():
            print("Solution found using Hill Climbing!")
            print("Path to goal:")
            for move, state in path:
                print(f"Move: {move}")
                for row in state.grid:
                    print(" ".join(row))
                print("\n")
            print(f"Total states explored: {states_explored}")
            return path

        print("No solution found using Hill Climbing.")
        return None




grid5 = np.array([
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

game = Gridgame(grid5)
# game.dfs_solve_recursive()
# game.bfs_solve()
# game.a_star_solve()
game.hill_climbing_solve()
