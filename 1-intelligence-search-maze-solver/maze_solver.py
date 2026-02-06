"""
Maze Solver: Graph Search Algorithm Analysis Module
Author: Meet Ahalpara

This module implements and evaluates foundational Graph Search algorithms, including 
Uninformed Search (Depth-First Search, Breadth-First Search) and Informed Search (A*).
It is designed to analyze path optimality, computational efficiency, and state-space 
exploration metrics within a discrete maze topology.
"""

from collections import deque
import heapq

def load_maze(path):
    """
    Parses an external text-based maze configuration file. 
    The boundaries, structural walls, and exact coordinate positions of the 
    Start (S) and Goal (G) vertices are structurally identified and extracted.
    """
    with open(path) as f:
        # Trailing newlines are stripped to ensure uniform row configuration lengths across the spatial matrix.
        lines = [line.rstrip("\n") for line in f if line.strip()]  
        
    start = goal = None
    for r, row in enumerate(lines):              
        for c, ch in enumerate(row):             
            if ch == "S":
                start = (r, c)                   # Source vertex coordinate mapping
            elif ch == "G":
                goal = (r, c)                    # Target vertex coordinate mapping
                
    if start is None or goal is None:
        raise ValueError("Invalid topology: Maze configuration must contain explicit 'S' and 'G' markers.")
    return lines, start, goal                   

def in_bounds(maze, r, c):
    """Verifies whether an arbitrary coordinate pair falls within the established spatial matrix dimensions."""
    return 0 <= r < len(maze) and 0 <= c < len(maze[0])  

def neighbors(maze, pos):
    """
    Retrieves adjacent, accessible nodes using a 4-connectivity spatial grid model.
    Boundary violations and structural obstacles (#) are systematically filtered out.
    """
    r, c = pos                                    
    # Directional vectors defining orthogonal translations: Up, Down, Left, Right
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]      
    valid_transitions = []
    
    for dr, dc in steps:                         
        nr, nc = r + dr, c + dc                  
        if in_bounds(maze, nr, nc) and maze[nr][nc] != "#":  
            valid_transitions.append((nr, nc))              
    return valid_transitions

def reconstruct_path(came_from, start, goal):
    """
    Performs backward traversal from the target destination node back to the source vertex.
    The recorded parent node mappings are utilized to synthesize the exact chronological solution path.
    """
    if goal not in came_from:                    
        return None
    cur = goal                                   
    path = [cur]                                 
    while cur != start:                          
        cur = came_from[cur]
        path.append(cur)
    path.reverse()                               # The sequence is inverted to yield a chronological path from Start to Goal.
    return path

def dfs(maze, start, goal):
    """
    Executes Depth-First Search (DFS) using an explicit Last-In, First-Out (LIFO) stack architecture.
    This algorithm explores paths deeply along single branches before evaluating alternative trajectories, 
    which does not guarantee optimal path lengths.
    """
    stack = [start]                              # LIFO computational processing mechanism
    came_from = {}                               # Node tracking map for trajectory reconstruction
    visited = set([start])                       # Hash set tracking the evaluated node frontier
    explored_count = 0                                 

    while stack:                                 
        current = stack.pop()                    # Extracts the deepest active node from the stack
        explored_count += 1                            
        if current == goal:                      
            break
        for nxt in neighbors(maze, current):     
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = current        
                stack.append(nxt)                

    path = reconstruct_path(came_from, start, goal)
    path_length = len(path) - 1 if path else None  
    return path, path_length, explored_count

def bfs(maze, start, goal):
    """
    Executes Breadth-First Search (BFS) utilizing a First-In, First-Out (FIFO) queue architecture.
    This approach systematically evaluates states level-by-level, guaranteeing structural path 
    optimality within unweighted graph environments.
    """
    q = deque([start])                           # FIFO computational processing queue
    came_from = {}
    visited = set([start])
    explored_count = 0

    while q:                                     
        current = q.popleft()                    # Processes the earliest discovered vertex
        explored_count += 1
        if current == goal:
            break
        for nxt in neighbors(maze, current):
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = current
                q.append(nxt)                    

    path = reconstruct_path(came_from, start, goal)
    path_length = len(path) - 1 if path else None
    return path, path_length, explored_count

def manhattan(a, b):
    """Calculates the L1 norm (Manhattan Distance) heuristic between two discrete matrix coordinates."""
    (r1, c1), (r2, c2) = a, b
    return abs(r1 - r2) + abs(c1 - c2)           

def astar(maze, start, goal):
    """
    Executes A* Informed Search leveraging a heuristic evaluation function f(n) = g(n) + h(n).
    A Priority Queue (min-heap) is maintained to heavily optimize state-space exploration 
    metrics while guaranteeing path optimality.
    """
    open_heap = []                               # Priority Queue tracking the lowest f(n) cost
    heapq.heappush(open_heap, (0, start))        
    came_from = {}
    g_score = {start: 0}                         # Exact structural step distance from the source vertex
    in_open = {start}                            # Tracks active open set elements across the frontier
    explored_count = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)    # Dequeues the node presenting the minimal f(n) evaluation score
        explored_count += 1
        in_open.discard(current)                 # Transitions the node into the closed set space

        if current == goal:
            break

        for nxt in neighbors(maze, current):
            tentative_g = g_score[current] + 1    
            if tentative_g < g_score.get(nxt, float("inf")):  # A structurally lower cost path trajectory has been discovered
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + manhattan(nxt, goal)  # f(n) evaluation calculation
                if nxt not in in_open:           
                    heapq.heappush(open_heap, (f_score, nxt))  
                    in_open.add(nxt)

    path = reconstruct_path(came_from, start, goal)
    path_length = len(path) - 1 if path else None
    return path, path_length, explored_count

def print_results_table(results):
    """Generates a structured, tabular performance metrics comparison matrix directed to standard output."""
    valid_lengths = [length for _, length, _ in results.values() if length]
    optimal_length = min(valid_lengths) if valid_lengths else 0
    
    print("\n" + "="*65)
    print("      GRAPH SEARCH ALGORITHM PERFORMANCE METRICS COMPARISON")
    print("="*65)
    print(f"{'Algorithm':<15} {'Path Length':<15} {'Explored States':<18} {'Optimal?':<10}")
    print("-"*65)
    
    for name, (_, length, explored) in results.items():
        optimal_status = "YES" if length == optimal_length else "NO"
        len_display = length if length is not None else "N/A"
        print(f"{name:<15} {len_display:<15} {explored:<18} {optimal_status:<10}")
    
    print("="*65 + "\n")

def main():
    maze_path = "maze.txt"
    try:
        maze, start, goal = load_maze(maze_path)    
    except FileNotFoundError:
        print(f"Error: The configuration file '{maze_path}' could not be located.")
        return
    except ValueError as e:
        print(e)
        return

    results = {}
    for name, algo in [("DFS", dfs), ("BFS", bfs), ("A*", astar)]:
        path, length, explored = algo(maze, start, goal)  
        results[name] = (path, length, explored)
        print(f"=== System Execution Protocol: {name} ===")
        print(f"Computed Solution Path Length: {length}")
        print(f"Total State Expansion Count  : {explored}")
        print(f"Sequence Coordinates Map     : {path}\n")
    
    print_results_table(results)                 

if __name__ == "__main__":
    main()