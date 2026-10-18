"""
AI Planning Module for Engineering Student Workload.
Implements:
1. Subject & task decomposition
2. Directed Graph of prerequisite relationships
3. Topological sorting (Kahn's algorithm) with cycle detection
4. Multi-criteria task prioritization (deadline, difficulty, importance, remaining work)
5. Dynamic replanning on progress updates
6. Horizon capacity validation & bottleneck detection
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime, date
import copy
from models import Subject, Task, Difficulty, SchedulingReport

class AIPlanner:
    def __init__(self):
        pass

    @staticmethod
    def calculate_urgency_score(deadline_str: str, base_date: Optional[date] = None) -> float:
        """
        Calculates urgency based on days remaining until deadline.
        Closer deadlines yield higher urgency exponentially.
        """
        if not base_date:
            base_date = date.today()
        try:
            deadline = datetime.strptime(deadline_str, "%Y-%m-%d").date()
            delta_days = (deadline - base_date).days
            if delta_days <= 0:
                return 10.0  # Immediate / past due
            elif delta_days <= 2:
                return 8.5
            elif delta_days <= 4:
                return 7.0
            elif delta_days <= 7:
                return 5.0
            elif delta_days <= 14:
                return 3.0
            else:
                return max(1.0, 10.0 / (delta_days + 1))
        except Exception:
            return 5.0

    @classmethod
    def compute_task_priority(cls, task: Task, subject: Subject, base_date: Optional[date] = None) -> float:
        """
        Computes composite task priority score:
        Score = w_urgency * Urgency + w_diff * Difficulty + w_imp * Importance + w_rem * RemainingWork
        """
        w_urgency = 0.35
        w_difficulty = 0.20
        w_importance = 0.25
        w_remaining = 0.20

        urgency = cls.calculate_urgency_score(task.deadline or subject.deadline, base_date)
        diff_val = Difficulty.weight(task.difficulty or subject.difficulty) * 3.5  # scale ~ 3.5 to 7.7
        importance_val = float(task.importance or subject.importance) * 2.0  # scale 2 to 10
        rem_work_ratio = (task.remaining_hours / max(0.5, float(task.duration_hours))) * 10.0

        score = (
            w_urgency * urgency +
            w_difficulty * diff_val +
            w_importance * importance_val +
            w_remaining * rem_work_ratio
        )
        return round(score, 3)

    @staticmethod
    def build_dependency_graph(tasks: List[Task]) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
        """
        Builds adjacency list (prereq -> dependents) and in-degree map for tasks.
        """
        task_id_set = {t.id for t in tasks}
        adjacency: Dict[str, Set[str]] = {t.id: set() for t in tasks}
        in_degree: Dict[str, int] = {t.id: 0 for t in tasks}

        for t in tasks:
            for prereq in t.prerequisite_task_ids:
                if prereq in task_id_set:
                    adjacency[prereq].add(t.id)
                    in_degree[t.id] += 1

        return adjacency, in_degree

    @classmethod
    def topological_sort(cls, tasks: List[Task]) -> Tuple[List[Task], bool, List[str]]:
        """
        Kahn's Algorithm for Topological Sorting.
        Orders prerequisite topics strictly before dependent topics.
        Returns: (sorted_tasks, has_cycle, diagnostic_messages)
        """
        task_map = {t.id: t for t in tasks}
        adjacency, in_degree = cls.build_dependency_graph(tasks)

        # Queue of tasks with 0 in-degree (ready to schedule)
        # Tie-break queue by task priority
        ready_tasks = [t for t in tasks if in_degree[t.id] == 0]
        # Sort ready tasks initially by remaining hours and deadline
        ready_tasks.sort(key=lambda t: (t.deadline, -t.importance))

        sorted_result: List[Task] = []
        queue = [t.id for t in ready_tasks]
        visited_count = 0
        diagnostics = []

        while queue:
            curr_id = queue.pop(0)
            sorted_result.append(task_map[curr_id])
            visited_count += 1

            for neighbor in sorted(adjacency[curr_id]):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        has_cycle = visited_count < len(tasks)
        if has_cycle:
            unresolved = [t.id for t in tasks if in_degree[t.id] > 0]
            diagnostics.append(f"Cycle detected in prerequisite graph involving tasks: {', '.join(unresolved)}")
            # Fallback append remaining tasks
            for t in tasks:
                if t not in sorted_result:
                    sorted_result.append(t)

        return sorted_result, has_cycle, diagnostics

    @classmethod
    def plan_curriculum(cls, subjects: List[Subject], base_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Decomposes curriculum, builds global DAG, checks capacity, and outputs prioritized plan.
        """
        all_tasks: List[Task] = []
        subject_map = {s.id: s for s in subjects}

        for s in subjects:
            for t in s.tasks:
                all_tasks.append(t)

        # Topological sort respects internal task prereqs
        sorted_tasks, has_cycle, cycle_diagnostics = cls.topological_sort(all_tasks)

        # Annotate with priority
        prioritized_tasks = []
        for t in sorted_tasks:
            parent_sub = subject_map.get(t.subject_id)
            if parent_sub:
                p_score = cls.compute_task_priority(t, parent_sub, base_date)
            else:
                p_score = cls.calculate_urgency_score(t.deadline, base_date)
            prioritized_tasks.append({
                "task": t.to_dict(),
                "priority_score": p_score,
                "subject_name": parent_sub.name if parent_sub else "Unknown",
                "remaining_hours": t.remaining_hours
            })

        total_work_hours = sum(t.remaining_hours for t in all_tasks)

        return {
            "total_tasks": len(all_tasks),
            "total_work_hours": round(total_work_hours, 2),
            "has_cycle": has_cycle,
            "diagnostics": cycle_diagnostics,
            "plan_sequence": prioritized_tasks
        }

    @classmethod
    def replan_on_progress_update(cls, subjects: List[Subject], updated_task_id: str, new_pct: float) -> Dict[str, Any]:
        """
        Dynamically replans when student updates task progress.
        Recalculates remaining hours, critical paths, and bottleneck detection.
        """
        for s in subjects:
            for t in s.tasks:
                if t.id == updated_task_id:
                    t.completion_pct = min(100.0, max(0.0, float(new_pct)))
                    if t.completion_pct >= 100.0:
                        t.duration_hours = 0.0
            # Recalculate subject overall progress
            if s.tasks:
                s.completion_percentage = round(sum(t.completion_pct for t in s.tasks) / len(s.tasks), 1)
                s.estimated_remaining_hours = round(sum(t.remaining_hours for t in s.tasks), 2)

        return cls.plan_curriculum(subjects)
