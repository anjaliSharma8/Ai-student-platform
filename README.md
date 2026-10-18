# Optimizing Holistic Engineering Student Performance Using Advanced AI Search, Planning and Probabilistic Reasoning

A complete, production-grade AI case study and optimization system tailored for engineering students. The system balances academic workload, prioritizes subjects, generates feasible personalized timetables, visualizes search spaces, and infers deadline-miss risks using genuine Python AI algorithms.

---

## 1. System Architecture

```
                                  [ User Interface ]
                       Streamlit Dashboard (Port 8501)
                                      │
                                      ▼
             ┌───────────────── AI Engine Orchestrator ────────────────┐
             │                                                         │
             ├───────────────┬─────────────────┬───────────────────────┤
             ▼               ▼                 ▼                       ▼
    [ AI Planning ]     [ A* Search ]    [ CSP Solver ]     [ Bayesian Network ]
   Topological Sort    Admissible H(n)   MRV + LCV + AC3     Exact VE Inference
    Task Decomposer     Priority Queue     Backtracking       Risk Posteriors
             │               │                 │                       │
             └───────────────┼─────────────────┴───────────────────────┘
                             ▼
                  [ SQLite Database Layer ]
                    `student_optimizer.db`
```

---

## 2. Core Python AI Algorithms & Formulations

### A. A* Search Algorithm (`astar_scheduler.py`)
- **State Space Representation $s$**:
  $$s = (\text{slot\_index}, \text{assigned\_sessions}, \text{unscheduled\_tasks}, \text{last\_subject\_id})$$
- **Path Cost $g(s)$**:
  $$g(s) = \sum \text{duration} + \text{context\_switch\_penalty} + \text{preference\_mismatch\_penalty} + \text{fatigue\_penalty}$$
- **Heuristic Function $h(s)$**:
  $$h(s) = \sum_{t \in \text{unscheduled}} \text{remaining\_hours}(t) + \text{urgency\_penalty}(t)$$
  *Admissibility & Consistency:* Because scheduling every remaining task requires at least its minimum duration, $h(s) \le h^*(s)$, ensuring no overestimation of remaining cost and yielding optimal timetable paths.
- **Priority Queue**: Implemented using `heapq` with visited-state memoization (`closed_set`).
- **Complexity**:
  - Time Complexity: $O(b^d)$ bounded with state pruning ($b \le \text{tasks}$, $d \le \text{slots}$).
  - Space Complexity: $O(b \cdot d)$ stored in priority queue and closed set.

### B. Constraint Satisfaction Problem (CSP) Solver (`csp_solver.py`)
- **Variables**: Study task blocks requiring scheduling.
- **Domains**: Discrete calendar time slots across a multi-day horizon.
- **Constraints**:
  1. *Non-Overlapping*: No two sessions can share identical time blocks.
  2. *Daily Capacity*: Total scheduled duration per day $\le$ student available hours.
  3. *Deadlines*: Session completion $\le$ deadline calendar date.
  4. *Prerequisites*: Prerequisite tasks must conclude strictly before dependent tasks commence.
  5. *Fixed Commitments*: Cannot conflict with student's lectures, labs, or work.
- **Heuristics**:
  - **MRV (Minimum Remaining Values)**: Selects variable with fewest valid domain slots.
  - **Degree Heuristic**: Breaks MRV ties by prioritizing variables with highest dependency degree.
  - **LCV (Least Constraining Value)**: Orders domain choices to preserve maximum freedom for unassigned variables.
  - **Forward Checking**: Propagates arc consistency to detect domain wipeouts immediately.
- **Complexity**:
  - Time Complexity: $O(d^n)$ in worst case, reduced by orders of magnitude via MRV + LCV + Forward Checking.

### C. AI Planning & Topological Sorting (`planning.py`)
- **Prerequisite Graph**: Directed Acyclic Graph (DAG) of task dependencies.
- **Topological Sorting**: Kahn's algorithm with cycle detection.
- **Multi-Factor Urgency Scoring**:
  $$\text{Priority Score} = 0.35 \cdot \text{Urgency} + 0.20 \cdot \text{Difficulty} + 0.25 \cdot \text{Importance} + 0.20 \cdot \text{Remaining Work}$$
- **Dynamic Replanning**: On student progress updates, re-computes critical paths and remaining workload.

### D. Bayesian Network & Probabilistic Inference (`bayesian_network.py`)
- **DAG Nodes**:
  - `Workload` $\in \{\text{Low}, \text{Medium}, \text{High}\}$
  - `Difficulty` $\in \{\text{Easy}, \text{Medium}, \text{Hard}\}$
  - `TimeRemaining` $\in \{\text{Sufficient}, \text{Limited}, \text{Critical}\}$
  - `Progress` $\in \{\text{Low}, \text{Medium}, \text{High}\}$
  - `DeadlineRisk` $\in \{\text{Low}, \text{Medium}, \text{High}\}$
- **Conditional Probability Tables (CPTs)**:
  Fully normalized probability distributions satisfying $\sum P(\cdot) = 1.0$.
- **Exact Inference**: Factor-based Variable Elimination calculating $P(\text{DeadlineRisk} \mid \text{Evidence})$.
- **Contribution Sensitivity**: Evaluates counterfactual shifts to isolate top risk drivers.

---

## 3. Installation & Local Execution

### Prerequisites
- Python 3.10+
- pip

### Setup
```bash
# Clone and navigate to repository
cd ai-student-optimizer

# Install Python dependencies
pip install -r requirements.txt

# Run Unit Tests
python3 -m unittest discover tests

# Launch Streamlit Application
streamlit run app.py
```

The Streamlit interface opens at `http://localhost:8501`.

---

## 4. Demonstration Dataset

Preloaded with 4 foundational engineering courses:
1. **Data Structures and Algorithms (CS-201)**: Hard (Graph traversal, DP, Advanced trees).
2. **Artificial Intelligence (CS-301)**: Hard (A* search, CSP backtracking, Bayesian inference).
3. **Database Management Systems (CS-204)**: Medium (Normalization, B+ tree indexing, SQL tuning).
4. **Computer Networks (CS-302)**: Medium (TCP/IP architecture, Socket programming with Wireshark).

---

## 5. Automated Unit Tests

Run the test suite:
```bash
python3 -m unittest discover tests
```
Tests verify:
- Kahn's algorithm topological sorting and cycle detection.
- A* priority queue scheduling and prerequisite satisfaction.
- CSP backtracking consistency and non-overlapping slots.
- Bayesian network exact probability distribution normalization ($\sum P = 1.0$) and monotonic risk reduction under high progress.
- SQLite persistence operations.
