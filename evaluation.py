"""
Evaluation Module for Experimental Benchmarking and Comparative Analysis.
Compares:
1. A* Search Scheduler (Admissible Heuristic + Priority Queue)
2. CSP Solver (MRV + LCV + Forward Checking)
3. Baseline Greedy FIFO Scheduler (Naive baseline without search optimization)
Measures:
- Task scheduling completeness (%)
- Constraint violations & satisfied counts
- Execution latency (milliseconds)
- State expansions & pruning events
- Bayesian risk shifts across progress transitions
"""

import time
from typing import List, Dict, Any, Tuple
from datetime import date
import copy
from models import Subject, Task, StudentProfile, SchedulingReport, ScheduledItem
from astar_scheduler import AStarScheduler
from csp_solver import CSPSolver
from bayesian_network import StudentRiskBayesianNetwork

class BaselineScheduler:
    """
    Simple naive baseline scheduler:
    Iterates through tasks chronologically by deadline, blindly assigns first available slot
    without optimizing for cognitive context switches, student fatigue, or prerequisite graphs.
    """
    def schedule(self, subjects: List[Subject], profile: StudentProfile, horizon_days: int = 7) -> SchedulingReport:
        start_time = time.time()
        helper = AStarScheduler()
        available_slots = helper.generate_discrete_slots(profile, horizon_days)
        total_avail_hours = sum(s.duration for s in available_slots)

        all_tasks: List[Task] = []
        subject_map = {s.id: s for s in subjects}
        for s in subjects:
            for t in s.tasks:
                if t.remaining_hours > 0:
                    all_tasks.append(t)

        # Naive sort strictly by deadline
        all_tasks.sort(key=lambda t: t.deadline)

        scheduled: List[ScheduledItem] = []
        unscheduled: List[Dict[str, Any]] = []
        occupied_slots = set()

        violations = 0
        satisfied = 0

        for t in all_tasks:
            assigned = False
            for slot in available_slots:
                slot_key = (slot.day, slot.start_hour)
                if slot_key in occupied_slots:
                    continue

                # Naive scheduler ignores prerequisite ordering or checks it late!
                # If prereq is violated:
                prereqs_met = True
                for prereq_id in t.prerequisite_task_ids:
                    # Check if prereq was scheduled before this slot
                    prereq_session = next((item for item in scheduled if item.task_id == prereq_id), None)
                    if not prereq_session or prereq_session.day_index > slot.day_index:
                        prereqs_met = False
                        violations += 1
                        break

                if prereqs_met:
                    satisfied += 1

                s_obj = subject_map.get(t.subject_id)
                dur = min(slot.duration, t.remaining_hours)
                scheduled.append(ScheduledItem(
                    task_id=t.id,
                    task_name=t.name,
                    subject_id=t.subject_id,
                    subject_name=s_obj.name if s_obj else "Subject",
                    subject_color=s_obj.color if s_obj else "#94A3B8",
                    day=slot.day,
                    day_index=slot.day_index,
                    start_hour=slot.start_hour,
                    end_hour=slot.start_hour + dur,
                    duration_hours=round(dur, 2),
                    period=slot.period,
                    deadline=t.deadline,
                    satisfies_prerequisites=prereqs_met
                ))
                occupied_slots.add(slot_key)
                assigned = True
                break

            if not assigned:
                unscheduled.append({
                    "task_id": t.id,
                    "task_name": t.name,
                    "remaining_hours": t.remaining_hours,
                    "reason": "Baseline slot depletion"
                })

        elapsed_ms = (time.time() - start_time) * 1000.0
        allocated_hours = sum(i.duration_hours for i in scheduled)

        return SchedulingReport(
            algorithm_name="Baseline Greedy (EDF)",
            success=(len(unscheduled) == 0 and violations == 0),
            scheduled_items=scheduled,
            unscheduled_tasks=unscheduled,
            total_allocated_hours=round(allocated_hours, 2),
            total_available_hours=round(total_avail_hours, 2),
            execution_time_ms=round(elapsed_ms, 2),
            states_explored=len(all_tasks) * len(available_slots),
            backtracking_steps=0,
            constraints_satisfied=satisfied,
            constraints_violated=violations + len(unscheduled),
            diagnostics=["Baseline scheduler does not prune or backtrack when prerequisite order is inverted."]
        )

class Evaluator:
    @classmethod
    def run_comparative_benchmark(
        cls,
        subjects: List[Subject],
        profile: StudentProfile,
        horizon_days: int = 7
    ) -> Dict[str, Any]:
        """
        Runs comprehensive comparative benchmark across A*, CSP, and Baseline.
        """
        astar = AStarScheduler(max_expansions=3000)
        csp = CSPSolver(max_backtracks=3000)
        baseline = BaselineScheduler()

        # Run A*
        report_astar = astar.schedule(subjects, profile, horizon_days)

        # Run CSP
        report_csp = csp.solve(subjects, profile, horizon_days)

        # Run Baseline
        report_baseline = baseline.schedule(subjects, profile, horizon_days)

        total_tasks = sum(len(s.tasks) for s in subjects)

        benchmark_table = [
            {
                "Algorithm": "A* Search (Optimal Heuristic)",
                "Tasks Scheduled": len(report_astar.scheduled_items),
                "Completion Rate (%)": round((len(report_astar.scheduled_items) / max(1, total_tasks)) * 100, 1),
                "Allocated Hours": report_astar.total_allocated_hours,
                "Constraints Satisfied": report_astar.constraints_satisfied,
                "Constraints Violated": report_astar.constraints_violated,
                "States / Checks": report_astar.states_explored,
                "Execution Time (ms)": report_astar.execution_time_ms,
                "Feasibility": "Feasible (Optimal)" if report_astar.success else "Partial Fallback"
            },
            {
                "Algorithm": "CSP Solver (MRV + LCV + AC)",
                "Tasks Scheduled": len(report_csp.scheduled_items),
                "Completion Rate (%)": round((len(report_csp.scheduled_items) / max(1, total_tasks)) * 100, 1),
                "Allocated Hours": report_csp.total_allocated_hours,
                "Constraints Satisfied": report_csp.constraints_satisfied,
                "Constraints Violated": report_csp.constraints_violated,
                "States / Checks": report_csp.states_explored,
                "Execution Time (ms)": report_csp.execution_time_ms,
                "Feasibility": "Satisfied" if report_csp.success else "Infeasible / Pruned"
            },
            {
                "Algorithm": "Baseline Greedy (Earliest Deadline First)",
                "Tasks Scheduled": len(report_baseline.scheduled_items),
                "Completion Rate (%)": round((len(report_baseline.scheduled_items) / max(1, total_tasks)) * 100, 1),
                "Allocated Hours": report_baseline.total_allocated_hours,
                "Constraints Satisfied": report_baseline.constraints_satisfied,
                "Constraints Violated": report_baseline.constraints_violated,
                "States / Checks": report_baseline.states_explored,
                "Execution Time (ms)": report_baseline.execution_time_ms,
                "Feasibility": "Heuristic Only (Suboptimal)"
            }
        ]

        return {
            "benchmark_table": benchmark_table,
            "astar_report": report_astar.to_dict(),
            "csp_report": report_csp.to_dict(),
            "baseline_report": report_baseline.to_dict()
        }

    @classmethod
    def evaluate_bayesian_sensitivity(cls) -> List[Dict[str, Any]]:
        """
        Evaluates Bayesian risk sensitivity when student updates task progress.
        Simulates progress transitions: Low -> Medium -> High under differing workloads.
        """
        bn = StudentRiskBayesianNetwork()
        scenarios = [
            {
                "Scenario": "Heavy Exam Week (Initial State)",
                "Evidence": {"Workload": "High", "Difficulty": "Hard", "TimeRemaining": "Critical", "Progress": "Low"}
            },
            {
                "Scenario": "Mid-Week Progress Boost (After Dynamic Replan)",
                "Evidence": {"Workload": "High", "Difficulty": "Hard", "TimeRemaining": "Limited", "Progress": "Medium"}
            },
            {
                "Scenario": "Final Stretch (High Task Completion)",
                "Evidence": {"Workload": "Medium", "Difficulty": "Medium", "TimeRemaining": "Limited", "Progress": "High"}
            },
            {
                "Scenario": "Ideal Paced Preparation",
                "Evidence": {"Workload": "Low", "Difficulty": "Easy", "TimeRemaining": "Sufficient", "Progress": "High"}
            }
        ]

        results = []
        for s in scenarios:
            post = bn.infer_risk_posterior(s["Evidence"])
            results.append({
                "Scenario": s["Scenario"],
                "Evidence": s["Evidence"],
                "P(Low Risk)": post.get("Low", 0.0),
                "P(Medium Risk)": post.get("Medium", 0.0),
                "P(High Risk)": post.get("High", 0.0)
            })

        return results
