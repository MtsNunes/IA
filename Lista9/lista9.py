import heapq
from collections import deque
import time

def get_blank_pos(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return (i, j)
    return None

def generate_moves(state):
    blank_i, blank_j = get_blank_pos(state)
    moves = []
    dirs = [('up', -1, 0), ('down', 1, 0), ('left', 0, -1), ('right', 0, 1)]
    
    for action, di, dj in dirs:
        new_i, new_j = blank_i + di, blank_j + dj
        if 0 <= new_i < 3 and 0 <= new_j < 3:
            new_state = [list(row) for row in state]
            new_state[blank_i][blank_j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[blank_i][blank_j]
            new_state = tuple(map(tuple, new_state))
            moves.append((new_state, action))
    return moves

def bfs(initial, goal):
    initial_tuple = tuple(map(tuple, initial))
    goal_tuple = tuple(map(tuple, goal))
    
    visited = set()
    queue = deque([(initial_tuple, [])])
    nodes_visited = 0
    
    while queue:
        current, path = queue.popleft()
        nodes_visited += 1
        
        if current == goal_tuple:
            return path, nodes_visited
        
        if current in visited:
            continue
        visited.add(current)
        
        for next_state, move in generate_moves(current):
            if next_state not in visited:
                queue.append((next_state, path + [move]))
    
    return None, nodes_visited

def dfs(initial, goal):
    initial_tuple = tuple(map(tuple, initial))
    goal_tuple = tuple(map(tuple, goal))
    
    visited = set()
    stack = [(initial_tuple, [])]
    nodes_visited = 0
    
    while stack:
        current, path = stack.pop()
        nodes_visited += 1
        
        if current == goal_tuple:
            return path, nodes_visited
        
        if current in visited:
            continue
        visited.add(current)
        
        for next_state, move in generate_moves(current):
            if next_state not in visited:
                stack.append((next_state, path + [move]))
    
    return None, nodes_visited

def misplaced_tiles(state, goal):
    goal_tuple = tuple(map(tuple, goal))
    return sum(
        1 for i in range(3) 
        for j in range(3) 
        if state[i][j] != goal_tuple[i][j]
    )

def manhattan_distance(state, goal):
    goal_pos = {}
    for i in range(3):
        for j in range(3):
            goal_pos[goal[i][j]] = (i, j)
    
    distance = 0
    for i in range(3):
        for j in range(3):
            tile = state[i][j]
            target_i, target_j = goal_pos[tile]
            distance += abs(i - target_i) + abs(j - target_j)
    return distance

def a_star(initial, goal, heuristic):
    initial_tuple = tuple(map(tuple, initial))
    goal_tuple = tuple(map(tuple, goal))
    
    open_heap = []
    heapq.heappush(open_heap, (0, 0, initial_tuple, []))
    
    closed = dict()
    nodes_visited = 0
    
    while open_heap:
        current_f, current_g, current_state, path = heapq.heappop(open_heap)
        nodes_visited += 1
        
        if current_state == goal_tuple:
            return path, nodes_visited
        
        if current_state in closed and closed[current_state] <= current_g:
            continue
        closed[current_state] = current_g
        
        for next_state, move in generate_moves(current_state):
            next_g = current_g + 1
            next_h = heuristic(next_state, goal)
            next_f = next_g + next_h
            
            if next_state not in closed or next_g < closed.get(next_state, float('inf')):
                heapq.heappush(open_heap, (next_f, next_g, next_state, path + [move]))
    
    return None, nodes_visited

# Configuração do teste
initial = [
    [1, 2, 3],
    [4, 0, 5],
    [7, 8, 6]
]

goal = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

def run_algorithm(algorithm, heuristic=None):
    start_time = time.time()
    
    if algorithm == 'BFS':
        path, nodes = bfs(initial, goal)
    elif algorithm == 'DFS':
        path, nodes = dfs(initial, goal)
    elif algorithm == 'A*':
        path, nodes = a_star(initial, goal, heuristic)
    
    elapsed_time = (time.time() - start_time) * 1000
    
    return {
        'algorithm': algorithm + (f' ({heuristic.__name__})' if heuristic else ''),
        'nodes_visited': nodes,
        'time_ms': round(elapsed_time, 2),
        'path_length': len(path) if path else None
    }

results = []
results.append(run_algorithm('BFS'))
results.append(run_algorithm('DFS'))
results.append(run_algorithm('A*', misplaced_tiles))
results.append(run_algorithm('A*', manhattan_distance))

for res in results:
    print(f"{res['algorithm']}:")
    print(f"  Nós visitados: {res['nodes_visited']}")
    print(f"  Tempo: {res['time_ms']} ms")
    print(f"  Comprimento do caminho: {res['path_length']}\n")