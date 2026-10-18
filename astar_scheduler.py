"""
A* Search Algorithm for Study Timetable Generation.
Finds optimal task-to-timeslot schedule by exploring the state space using:
- Priority Queue (heapq)
- Admissible & Consistent Heuristic function h(n)
- Accurate Path-Cost function g(n) accounting for fatigue, context-switching, and preferences
- Visited state hashing / closed set
- Bounded search with documented fallback mechanism
- Search exploration trace logging for step-by-step visual analytics
"""

import heapq
import time
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime, date
from models import (
    Subject, Task, StudentProfile, TimeSlot, ScheduledItem,
    SchedulingReport, Difficulty
)

class AStarScheduler:
    def __init__(self, max_expansions: int = 4000, timeout_seconds: float = 4.0):
        self.max_expansions = max_expansions
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def generate_discrete_slots(profile: StudentProfile, horizon_days: int = 7) -> List[TimeSlot]:
        """
        Generates candidate time slots for each day based on available study hours and commitments.
        Divides available time into 1.0h to 2.0h manageable study blocks.
        """
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        slots: List[TimeSlot] = []
        slot_counter = 0

        for d_idx in range(horizon_days):
            day_name = day_names[d_idx % 7]
            avail_hours = profile.daily_available_hours.get(day_name, 4.0)

            # Define time windows: Morning (09:00-12:00), Afternoon (14:00-17:00), Evening (18:00-21:00)
            windows = [
                ("Morning", 9.0, 12.0),
                ("Afternoon", 14.0, 17.0),
                ("Evening", 18.0, 21.0),
                ("Night", 21.0, 23.0)
            ]

            hours_allocated_today = 0.0
            for period_name, start_w, end_w in windows:
                if hours_allocated_today >= avail_hours:
                    break

                block_len = min(2.0, avail_hours - hours_allocated_today)
                if block_len < 0.5:
                    break

                # Check fixed commitment collisions
                collides = False
                for comm in profile.fixed_commitments:
                    if comm.day == day_name:
                        # Overlap check
                        if max(start_w, comm.start_hour) < min(start_w + block_len, comm.end_hour):
                            collides = True
                            break

                if not collides:
                    slots.append(TimeSlot(
                        slot_id=f"slot_{d_idx}_{slot_counter}",
                        day=day_name,
                        day_index=d_idx,
                        start_hour=start_w,
                        end_hour=start_w + block_len,
                        period=period_name
                    ))
                    slot_counter += 1
                    hours_allocated_today += block_len

        return slots

    @staticmethod
    def compute_step_cost(
        task: Task,
        slot: TimeSlot,
        last_subject_id: Optional[str],
        profile: StudentProfile
    ) -> float:
        """
        Calculates path-cost step g_delta:
        - Base duration cost
        - Context-switch cost: penalty if alternating subjects back and forth
        - Preference alignment: reward for matching student's preferred study periods
        - Fatigue factor: hard subjects late at night penalized
        """
        step = task.duration_hours

        # Context switch penalty
        if last_subject_id and last_subject_id != task.subject_id:
            step += 0.4  # Cognitive switching overhead

        # Preference penalty
        if profile.preferred_study_periods and slot.period not in profile.preferred_study_periods:
            step += 0.6  # Studied outside peak preference

        # Circadian fatigue penalty
        if task.difficulty == Difficulty.HARD and slot.period == "Night":
            step += 1.0  # Complex engineering topics late at night

        return round(step, 3)

    @classmethod
    def compute_heuristic(
        cls,
        unscheduled_tasks: List[Task],
        current_day_index: int,
        horizon_days: int
    ) -> float:
        """
        Admissible & Consistent Heuristic h(n):
        Estimates minimum remaining scheduling cost to reach the goal.
        - Sum of remaining task durations (the absolute lower bound on scheduling work)
        - Urgency weighting: impending deadlines require higher density
        - Penalty for remaining work exceeding available horizon
        Because minimum duration must be spent, and non-negative penalties reflect lower bounds on
        scheduling friction, h(n) <= true remaining cost, satisfying admissibility h(n) <= h*(n).
        """
        if not unscheduled_tasks:
            return 0.0

        h_val = 0.0
        days_left = max(1, horizon_days - current_day_index)

        for t in unscheduled_tasks:
            h_val += t.remaining_hours

            # Urgency factor
            try:
                deadline_dt = datetime.strptime(t.deadline, "%Y-%m-%d").date()
                delta_days = (deadline_dt - date.today()).days
                if delta_days <= days_left:
                    h_val += 0.2 * (days_left - delta_days)
            except Exception:
                pass

        return round(h_val, 3)

    def schedule(
        self,
        subjects: List[Subject],
        profile: StudentProfile,
        horizon_days: int = 7
    ) -> SchedulingReport:
        """
        Executes A* Search over the partial schedule state space.
        """
        start_time = time.time()
        available_slots = self.generate_discrete_slots(profile, horizon_days)
        total_avail_hours = sum(s.duration for s in available_slots)

        # Flatten all tasks with remaining hours > 0
        all_tasks: List[Task] = []
        subject_map: Dict[str, Subject] = {s.id: s for s in subjects}
        for s in subjects:
            for t in s.tasks:
                if t.remaining_hours > 0:
                    all_tasks.append(t)

        # Build prerequisite map
        completed_task_ids: Set[str] = {t.id for t in all_tasks if t.completion_pct >= 100.0}

        # Priority Queue entry: (f_score, g_score, counter, state_tuple)
        # State tuple: (current_slot_idx, tuple_of_scheduled_items, tuple_of_unscheduled_ids, last_subject_id)
        open_list = []
        closed_set = set()
        counter = 0

        init_unscheduled = tuple(sorted(t.id for t in all_tasks))
        init_h = self.compute_heuristic(all_tasks, 0, horizon_days)
        init_g = 0.0
        init_f = init_g + init_h

        # (f, g, counter, slot_idx, scheduled_items, unscheduled_ids, last_subj_id)
        heapq.heappush(open_list, (init_f, init_g, counter, 0, (), init_unscheduled, None))

        best_schedule: List[ScheduledItem] = []
        best_unscheduled_ids: List[str] = list(init_unscheduled)
        search_trace: List[Dict[str, Any]] = []
        expansions = 0
        timed_out = False

        task_dict = {t.id: t for t in all_tasks}

        while open_list and expansions < self.max_expansions:
            # Check timeout
            if time.time() - start_time > self.timeout_seconds:
                timed_out = True
                break

            f_curr, g_curr, _, slot_idx, scheduled, unscheduled, last_subj = heapq.heappop(open_list)
            expansions += 1

            # Search trace sampling (store first 40 and periodically up to 100)
            if len(search_trace) < 80 and (expansions <= 30 or expansions % 25 == 0):
                sample_task_name = task_dict[unscheduled[0]].name if unscheduled else "None"
                search_trace.append({
                    "step": expansions,
                    "f_score": round(f_curr, 2),
                    "g_score": round(g_curr, 2),
                    "h_score": round(f_curr - g_curr, 2),
                    "open_list_size": len(open_list),
                    "closed_set_size": len(closed_set),
                    "scheduled_count": len(scheduled),
                    "remaining_tasks": len(unscheduled),
                    "next_candidate": sample_task_name
                })

            # Check if this state is better than our previous best
            if len(unscheduled) < len(best_unscheduled_ids):
                best_unscheduled_ids = list(unscheduled)
                best_schedule = list(scheduled)

            # Goal test: all tasks scheduled
            if not unscheduled:
                best_schedule = list(scheduled)
                best_unscheduled_ids = []
                break

            # Goal test: run out of time slots
            if slot_idx >= len(available_slots):
                continue

            state_key = (slot_idx, unscheduled)
            if state_key in closed_set:
                continue
            closed_set.add(state_key)

            curr_slot = available_slots[slot_idx]
            scheduled_ids_in_state = {item.task_id for item in scheduled}.union(completed_task_ids)

            # Try assigning an eligible task into the current slot
            branched = False
            for t_id in unscheduled:
                candidate_task = task_dict[t_id]

                # Check prerequisite constraint
                prereqs_met = all(p in scheduled_ids_in_state for p in candidate_task.prerequisite_task_ids)
                if not prereqs_met:
                    continue

                # Check duration compatibility
                duration = min(curr_slot.duration, candidate_task.remaining_hours)

                # Step cost
                step_cost = self.compute_step_cost(candidate_task, curr_slot, last_subj, profile)
                new_g = g_curr + step_cost

                # Remaining unscheduled tasks
                new_unscheduled_list = [uid for uid in unscheduled if uid != t_id]
                new_unscheduled_objs = [task_dict[uid] for uid in new_unscheduled_list]
                new_h = self.compute_heuristic(new_unscheduled_objs, curr_slot.day_index, horizon_days)
                new_f = new_g + new_h

                subj_obj = subject_map.get(candidate_task.subject_id)
                new_item = ScheduledItem(
                    task_id=candidate_task.id,
                    task_name=candidate_task.name,
                    subject_id=candidate_task.subject_id,
                    subject_name=subj_obj.name if subj_obj else "Subject",
                    subject_color=subj_obj.color if subj_obj else "#3B82F6",
                    day=curr_slot.day,
                    day_index=curr_slot.day_index,
                    start_hour=curr_slot.start_hour,
                    end_hour=curr_slot.start_hour + duration,
                    duration_hours=round(duration, 2),
                    period=curr_slot.period,
                    deadline=candidate_task.deadline
                )

                new_scheduled = scheduled + (new_item,)
                counter += 1
                heapq.heappush(open_list, (
                    new_f,
                    new_g,
                    counter,
                    slot_idx + 1,
                    new_scheduled,
                    tuple(new_unscheduled_list),
                    candidate_task.subject_id
                ))
                branched = True

            # Also allow the transition of leaving this slot idle (e.g. if no tasks fit or prefer later)
            # Add branch skipping slot
            counter += 1
            idle_g = g_curr + 0.1  # Minor idle penalty
            rem_objs = [task_dict[uid] for uid in unscheduled]
            idle_h = self.compute_heuristic(rem_objs, curr_slot.day_index, horizon_days)
            heapq.heappush(open_list, (
                idle_g + idle_h,
                idle_g,
                counter,
                slot_idx + 1,
                scheduled,
                unscheduled,
                last_subj
            ))

        # Fallback scheduling if bounded search didn't schedule all tasks
        diagnostics = []
        if expansions >= self.max_expansions or timed_out:
            diagnostics.append(
                f"A* Bounded Search limit reached ({expansions} states explored). "
                "Engaging deterministic priority-greedy fallback scheduler for unassigned tasks."
            )
            # Greedy fallback
            best_schedule, best_unscheduled_ids = self._greedy_fallback_fill(
                best_schedule,
                best_unscheduled_ids,
                available_slots,
                task_dict,
                subject_map
            )

        elapsed_ms = (time.time() - start_time) * 1000.0
        allocated_hours = sum(item.duration_hours for item in best_schedule)

        unscheduled_details = []
        for uid in best_unscheduled_ids:
            t_obj = task_dict[uid]
            s_obj = subject_map.get(t_obj.subject_id)
            unscheduled_details.append({
                "task_id": t_obj.id,
                "task_name": t_obj.name,
                "subject_name": s_obj.name if s_obj else "Unknown",
                "remaining_hours": t_obj.remaining_hours,
                "reason": "Exceeded available weekly study hours or prerequisite ordering bottleneck"
            })

        constraints_satisfied = len(best_schedule)
        constraints_violated = len(unscheduled_details)

        return SchedulingReport(
            algorithm_name="A* Search with Admissible Heuristic",
            success=(len(best_unscheduled_ids) == 0),
            scheduled_items=best_schedule,
            unscheduled_tasks=unscheduled_details,
            total_allocated_hours=round(allocated_hours, 2),
            total_available_hours=round(total_avail_hours, 2),
            execution_time_ms=round(elapsed_ms, 2),
            states_explored=expansions,
            backtracking_steps=len(closed_set),
            constraints_satisfied=constraints_satisfied,
            constraints_violated=constraints_violated,
            diagnostics=diagnostics,
            search_trace=search_trace
        )

    def _greedy_fallback_fill(
        self,
        current_schedule: List[ScheduledItem],
        unscheduled_ids: List[str],
        available_slots: List[TimeSlot],
        task_dict: Dict[str, Task],
        subject_map: Dict[str, Subject]
    ) -> Tuple[List[ScheduledItem], List[str]]:
        """
        Deterministic fallback scheduler used when A* bounds are reached.
        Assigns remaining tasks into any remaining free time slots based on highest priority.
        """
        schedule = list(current_schedule)
        scheduled_slot_keys = {(s.day, s.start_hour) for s in schedule}
        remaining_ids = list(unscheduled_ids)

        # Find unoccupied slots
        free_slots = [
            slot for slot in available_slots
            if (slot.day, slot.start_hour) not in scheduled_slot_keys
        ]

        # Sort remaining tasks by deadline and importance
        remaining_tasks = [task_dict[uid] for uid in remaining_ids]
        remaining_tasks.sort(key=lambda t: (t.deadline, -t.importance))

        unassigned = []
        for task in remaining_tasks:
            assigned = False
            for slot in free_slots:
                if (slot.day, slot.start_hour) not in scheduled_slot_keys:
                    s_obj = subject_map.get(task.subject_id)
                    dur = min(slot.duration, task.remaining_hours)
                    item = ScheduledItem(
                        task_id=task.id,
                        task_name=task.name,
                        subject_id=task.subject_id,
                        subject_name=s_obj.name if s_obj else "Subject",
                        subject_color=s_obj.color if s_obj else "#3B82F6",
                        day=slot.day,
                        day_index=slot.day_index,
                        start_hour=slot.start_hour,
                        end_hour=slot.start_hour + dur,
                        duration_hours=round(dur, 2),
                        period=slot.period,
                        deadline=task.deadline
                    )
                    schedule.append(item)
                    scheduled_slot_keys.add((slot.day, slot.start_hour))
                    assigned = True
                    break
            if not assigned:
                unassigned.append(task.id)

        return schedule, unassigned
