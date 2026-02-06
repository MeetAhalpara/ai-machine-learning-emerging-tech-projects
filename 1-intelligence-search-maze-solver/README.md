# Intelligence Search Maze Solver

Finding a way out of a maze is more than a puzzle; it is a foundational problem in computational logic. The challenge is not merely locating an exit, but discovering the most elegant and efficient route through a constrained, imperfect environment filled with dead ends. This project is an analytical exploration of discrete state-space pathfinding, designed to reveal how different algorithmic philosophies navigate complex topologies.

To understand the mechanics of digital navigation, three distinct architectures were engineered and evaluated. Depth-First Search was deployed to explore deeply and blindly. Breadth-First Search was implemented to map every possibility systematically. Finally, A* Search was introduced to bring something profound to the process: foresight. 

By tracking the execution of each approach, the system reveals stark contrasts in computational effort, architectural trade-offs, and path optimality.

## Performance Benchmarks

The results speak to the power of heuristic intelligence. Below is the comparative data recorded during deterministic execution across the spatial matrix:

| Algorithm | Path Length (Steps) | Explored States | Computational Structure | Optimality Guarantee |
| :--- | :---: | :---: | :--- | :---: |
| Breadth-First Search (BFS) | 23 | 66 | FIFO Queue | Yes |
| A* Search (Heuristic-Driven) | 23 | 46 | Priority Min-Heap | Yes |
| Depth-First Search (DFS) | 35 | 36 | LIFO Stack | No |

### Engineering Insights

The contrast between informed and uninformed search is clear. Both Breadth-First Search and A* successfully converged on the absolute mathematical shortest path of 23 steps. However, A* achieved this precise outcome while evaluating 30.3% fewer operational states. It demonstrates the immense efficiency gained when a system uses an evaluation function to discard useless paths before they are fully explored. 

Conversely, the blind persistence of Depth-First Search resulted in a bloated, sub-optimal path. By pushing deeply along single branches without a holistic strategy, it incurred a penalty of 12 unnecessary steps over the optimal alternative.

---

## System Architecture

At the core of the A* implementation is a mathematical model known as the Manhattan Distance, or L1 Norm. This heuristic measures the exact spatial difference between any given position and the final destination:

$$h(a) = |a_{\text{row}} - b_{\text{row}}| + |a_{\text{col}} - b_{\text{col}}|$$

Because this equation never overestimates the remaining distance to the goal, it is strictly admissible. This mathematical constraint is what guarantees the algorithm will always find the perfect route without unnecessary deviation.

### Project Layout
The project is structured with simplicity in mind, separating the computational logic from the physical environment.
* `maze_solver.py`: The central Python engine containing the algorithmic state evaluation and analytical logic.
* `maze.txt`: A plain text matrix defining the structural walls, walkable passageways, the starting vector, and the target destination.

---

## Local Execution Setup

To initiate the analysis pipeline, activate the project-level virtual environment and execute the core solver script.

```bash
source .venv/Scripts/activate
python maze_solver.py