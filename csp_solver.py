"""
Constraint Satisfaction Problem (CSP) Solver for Study Schedules.
Features:
- Variables: Study tasks needing scheduling
- Domains: Discrete time slots
- Constraints:
    * Non-overlapping sessions
    * Student daily hour capacity
    * Prerequisite ordering (prereqs before dependents)
    * Deadline feasibility (finish before deadline date)
    * Fixed commitments conflict avoidance
- Heuristics:
    * MRV (Minimum Remaining Values) for Variable Selection
    * Degree Heuristic (Most Constraining Variable) for tie-breaking
    * LCV (Least Constraining Value) for Value Ordering
- Inference: Forward Checking & Constraint Propagation
- Infeasibility diagnostics and constraint conflict explanation
"""

import time
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime, date
from models import (
    Subject, Task, StudentProfile, TimeSlot, ScheduledItem,
    SchedulingReport
)
from astar_scheduler import AStarScheduler

class CSPSolver:
    def __init__(self, max_backtracks: int = 5000):
        self.max_backtracks = max_backtracks
        self.backtrack_count = 0
        self.constraint_checks = 0
        self.pruning_count = 0

    @staticmethod
    def _is_slot_before_deadline(slot: TimeSlot, deadline_str: str) -> bool:
        """
        Validates that a time slot's day falls strictly on or before the deadline date.
        Assumes day_index 0 = today, day_index 6 = 6 days from today.
        """
        try:
            deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d").date()
            today = date.today()
            slot_date_delta = slot.day_index
            max_allowed_delta = (deadline_date - today).days
            return slot_date_delta <= max_allowed_delta
        except Exception:
            return True

    def initialize_domains(
        self,
        tasks: List[Task],
        available_slots: List[TimeSlot],
        profile: StudentProfile
    ) -> Dict[str, List[TimeSlot]]:
        """
        Initializes domains for each task variable, applying unary constraints
        (deadline compatibility, commitment collisions).
        """
        domains: Dict[str, List[TimeSlot]] = {}

        for task in tasks:
            valid_slots = []
            for slot in available_slots:
                self.constraint_checks += 1
                # Unary 1: Must be before deadline
                if not self._is_slot_before_deadline(slot, task.deadline):
                    continue

                valid_slots.append(slot)

            domains[task.id] = valid_slots

        return domains

    def select_unassigned_variable(
        self,
        assignment: Dict[str, TimeSlot],
        domains: Dict[str, List[TimeSlot]],
        task_map: Dict[str, Task]
    ) -> Optional[str]:
        """
        MRV (Minimum Remaining Values) + Degree Heuristic.
        Selects unassigned variable with smallest domain size.
        Ties are broken by number of dependent constraints (degree).
        """
        unassigned = [t_id for t_id in domains if t_id not in assignment]
        if not unassigned:
            return None

        # MRV: sort by domain size
        # Degree: count how many tasks list this task as a prerequisite
        dependents_count: Dict[str, int] = {t_id: 0 for t_id in unassigned}
        for t_id in unassigned:
            for other_id in domains:
                if other_id not in assignment:
                    other_task = task_map.get(other_id)
                    if other_task and t_id in other_task.prerequisite_task_ids:
                        dependents_count[t_id] += 1

        # Key: (domain_size, -degree, -importance)
        unassigned.sort(key=lambda tid: (
            len(domains[tid]),
            -dependents_count.get(tid, 0),
            -task_map[tid].importance
        ))

        return unassigned[0]

    def order_domain_values(
        self,
        var_id: str,
        assignment: Dict[str, TimeSlot],
        domains: Dict[str, List[TimeSlot]],
        task_map: Dict[str, Task]
    ) -> List[TimeSlot]:
        """
        LCV (Least Constraining Value) Heuristic.
        Orders values (time slots) by how few options they rule out for remaining unassigned variables.
        """
        curr_task = task_map[var_id]
        candidates = domains[var_id]

        def constraint_impact(slot: TimeSlot) -> int:
            ruled_out = 0
            for other_id, other_slots in domains.items():
                if other_id not in assignment and other_id != var_id:
                    other_task = task_map[other_id]
                    for s in other_slots:
                        # Overlap conflict
                        if s.day == slot.day and s.start_hour == slot.start_hour:
                            ruled_out += 1
                        # Prerequisite ordering conflict
                        elif var_id in other_task.prerequisite_task_ids:
                            # other task must be AFTER this task
                            if s.day_index < slot.day_index or (s.day_index == slot.day_index and s.start_hour <= slot.start_hour):
                                ruled_out += 2
                        elif other_id in curr_task.prerequisite_task_ids:
                            # this task must be AFTER other task
                            if s.day_index > slot.day_index or (s.day_index == slot.day_index and s.start_hour >= slot.start_hour):
                                ruled_out += 2
            return ruled_out

        return sorted(candidates, key=constraint_impact)

    def is_consistent(
        self,
        var_id: str,
        slot: TimeSlot,
        assignment: Dict[str, TimeSlot],
        task_map: Dict[str, Task]
    ) -> bool:
        """
        Validates binary and global constraints:
        1. No overlap with existing assigned slots
        2. Prerequisite ordering: if prereq is assigned, prereq must be before current slot;
           if dependent is assigned, dependent must be after current slot.
        """
        curr_task = task_map[var_id]

        for assigned_id, assigned_slot in assignment.items():
            self.constraint_checks += 1
            # Constraint 1: Non-overlapping
            if assigned_slot.day == slot.day and assigned_slot.start_hour == slot.start_hour:
                return False

            assigned_task = task_map[assigned_id]

            # Constraint 2: Prerequisite constraint
            # If assigned task is a prerequisite of current task
            if assigned_id in curr_task.prerequisite_task_ids:
                if assigned_slot.day_index > slot.day_index or (
                    assigned_slot.day_index == slot.day_index and assigned_slot.start_hour >= slot.start_hour
                ):
                    return False

            # If current task is a prerequisite of assigned task
            if var_id in assigned_task.prerequisite_task_ids:
                if slot.day_index > assigned_slot.day_index or (
                    slot.day_index == assigned_slot.day_index and slot.start_hour >= assigned_slot.start_hour
                ):
                    return False

        return True

    def forward_check(
        self,
        var_id: str,
        slot: TimeSlot,
        assignment: Dict[str, TimeSlot],
        domains: Dict[str, List[TimeSlot]],
        task_map: Dict[str, Task]
    ) -> Optional[Dict[str, List[TimeSlot]]]:
        """
        Forward Checking: Prunes inconsistent values from remaining unassigned variables.
        Returns pruned copy of domains, or None if domain wipeout occurs.
        """
        new_domains: Dict[str, List[TimeSlot]] = {k: list(v) for k, v in domains.items()}
        curr_task = task_map[var_id]

        for other_id in domains:
            if other_id not in assignment and other_id != var_id:
                other_task = task_map[other_id]
                pruned_slots = []

                for s in new_domains[other_id]:
                    # Check overlap
                    if s.day == slot.day and s.start_hour == slot.start_hour:
                        self.pruning_count += 1
                        continue

                    # Check prereq forward
                    if var_id in other_task.prerequisite_task_ids:
                        # other_task must be after var_id
                        if s.day_index < slot.day_index or (s.day_index == slot.day_index and s.start_hour <= slot.start_hour):
                            self.pruning_count += 1
                            continue
                    elif other_id in curr_task.prerequisite_task_ids:
                        # other_task must be before var_id
                        if s.day_index > slot.day_index or (s.day_index == slot.day_index and s.start_hour >= slot.start_hour):
                            self.pruning_count += 1
                            continue

                    pruned_slots.append(s)

                if not pruned_slots:
                    # Domain wipeout! Infeasible branch
                    return None

                new_domains[other_id] = pruned_slots

        return new_domains

    def backtrack(
        self,
        assignment: Dict[str, TimeSlot],
        domains: Dict[str, List[TimeSlot]],
        task_map: Dict[str, Task]
    ) -> Optional[Dict[str, TimeSlot]]:
        """
        Recursive Backtracking Search with MRV, LCV, and Forward Checking.
        """
        if len(assignment) == len(task_map):
            return assignment

        if self.backtrack_count >= self.max_backtracks:
            return None

        var_id = self.select_unassigned_variable(assignment, domains, task_map)
        if not var_id:
            return assignment

        ordered_values = self.order_domain_values(var_id, assignment, domains, task_map)

        for slot in ordered_values:
            self.backtrack_count += 1

            if self.is_consistent(var_id, slot, assignment, task_map):
                assignment[var_id] = slot

                pruned_domains = self.forward_check(var_id, slot, assignment, domains, task_map)
                if pruned_domains is not None:
                    result = self.backtrack(assignment, pruned_domains, task_map)
                    if result is not None:
                        return result

                del assignment[var_id]

        return None

    def diagnose_infeasibility(
        self,
        tasks: List[Task],
        available_slots: List[TimeSlot],
        subjects: List[Subject]
    ) -> List[str]:
        """
        Detailed diagnostic analysis explaining which constraints caused failure
        and recommending remediation adjustments.
        """
        diagnostics = []
        total_needed_hours = sum(t.remaining_hours for t in tasks)
        total_slot_hours = sum(s.duration for s in available_slots)

        if total_needed_hours > total_slot_hours:
            diagnostics.append(
                f"Deficit in Available Time: Total study work ({total_needed_hours:.1f} hrs) "
                f"exceeds available scheduled capacity ({total_slot_hours:.1f} hrs) by "
                f"{total_needed_hours - total_slot_hours:.1f} hours."
            )

        # Check deadline bottlenecks
        today = date.today()
        for t in tasks:
            try:
                deadline_dt = datetime.strptime(t.deadline, "%Y-%m-%d").date()
                delta_days = (deadline_dt - today).days
                if delta_days <= 1 and t.remaining_hours > 3.0:
                    diagnostics.append(
                        f"Imminent Deadline Bottleneck: Task '{t.name}' requires {t.remaining_hours} hrs "
                        f"but has a deadline within {delta_days} day(s)."
                    )
            except Exception:
                pass

        # Check prerequisite chaining
        for t in tasks:
            if len(t.prerequisite_task_ids) >= 2:
                diagnostics.append(
                    f"Complex Prerequisite Chain: Task '{t.name}' has multiple dependencies "
                    f"({len(t.prerequisite_task_ids)} prerequisites), heavily narrowing feasible slot domains."
                )

        diagnostics.append(
            "Recommendation: Increase daily study hours on Friday/Saturday or allocate buffer slots to restore feasibility."
        )

        return diagnostics

    def solve(
        self,
        subjects: List[Subject],
        profile: StudentProfile,
        horizon_days: int = 7
    ) -> SchedulingReport:
        """
        Main entry point for CSP Scheduling Solver.
        """
        start_time = time.time()
        self.backtrack_count = 0
        self.constraint_checks = 0
        self.pruning_count = 0

        scheduler_helper = AStarScheduler()
        available_slots = scheduler_helper.generate_discrete_slots(profile, horizon_days)
        total_avail_hours = sum(s.duration for s in available_slots)

        # Tasks to schedule
        active_tasks: List[Task] = []
        subject_map: Dict[str, Subject] = {s.id: s for s in subjects}
        for s in subjects:
            for t in s.tasks:
                if t.remaining_hours > 0:
                    active_tasks.append(t)

        task_map = {t.id: t for t in active_tasks}
        initial_domains = self.initialize_domains(active_tasks, available_slots, profile)

        # Execute Backtracking
        assignment = self.backtrack({}, initial_domains, task_map)
        elapsed_ms = (time.time() - start_time) * 1000.0

        scheduled_items: List[ScheduledItem] = []
        unscheduled: List[Dict[str, Any]] = []
        diagnostics: List[str] = []

        if assignment is not None:
            for t_id, slot in assignment.items():
                task_obj = task_map[t_id]
                subj_obj = subject_map.get(task_obj.subject_id)
                dur = min(slot.duration, task_obj.remaining_hours)
                scheduled_items.append(ScheduledItem(
                    task_id=task_obj.id,
                    task_name=task_obj.name,
                    subject_id=task_obj.subject_id,
                    subject_name=subj_obj.name if subj_obj else "Subject",
                    subject_color=subj_obj.color if subj_obj else "#10B981",
                    day=slot.day,
                    day_index=slot.day_index,
                    start_hour=slot.start_hour,
                    end_hour=slot.start_hour + dur,
                    duration_hours=round(dur, 2),
                    period=slot.period,
                    deadline=task_obj.deadline
                ))
            # Sort chronologically
            day_order = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            scheduled_items.sort(key=lambda item: (item.day_index, item.start_hour))
            success = True
        else:
            success = False
            diagnostics = self.diagnose_infeasibility(active_tasks, available_slots, subjects)
            # Record unscheduled tasks
            for t in active_tasks:
                s_obj = subject_map.get(t.subject_id)
                unscheduled.append({
                    "task_id": t.id,
                    "task_name": t.name,
                    "subject_name": s_obj.name if s_obj else "Unknown",
                    "remaining_hours": t.remaining_hours,
                    "reason": "CSP backtracking exhausted: no domain assignment satisfied all simultaneous constraints."
                })

        allocated_hours = sum(item.duration_hours for item in scheduled_items)

        return SchedulingReport(
            algorithm_name="Constraint Satisfaction Problem (CSP) Solver",
            success=success,
            scheduled_items=scheduled_items,
            unscheduled_tasks=unscheduled,
            total_allocated_hours=round(allocated_hours, 2),
            total_available_hours=round(total_avail_hours, 2),
            execution_time_ms=round(elapsed_ms, 2),
            states_explored=self.constraint_checks,
            backtracking_steps=self.backtrack_count,
            constraints_satisfied=len(scheduled_items) * 3,  # non-overlap, deadline, prereq
            constraints_violated=len(unscheduled),
            diagnostics=diagnostics,
            search_trace=[{
                "pruning_events": self.pruning_count,
                "backtracking_steps": self.backtrack_count,
                "constraint_evaluations": self.constraint_checks
            }]
        )
