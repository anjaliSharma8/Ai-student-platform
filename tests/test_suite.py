"""
Comprehensive unit tests for:
- A* Scheduler
- CSP Solver
- AI Planner
- Bayesian Network
- Database persistence
"""

import unittest
from datetime import date, timedelta
from models import Subject, Task, StudentProfile, Difficulty, TimeSlot
from astar_scheduler import AStarScheduler
from csp_solver import CSPSolver
from planning import AIPlanner
from bayesian_network import StudentRiskBayesianNetwork
from database import Database
import os

class TestAIPlanner(unittest.TestCase):
    def setUp(self):
        self.today = date.today()
        self.task1 = Task(
            id="t1", subject_id="s1", name="Foundations",
            duration_hours=2.0, deadline=(self.today + timedelta(days=3)).isoformat(),
            prerequisite_task_ids=[]
        )
        self.task2 = Task(
            id="t2", subject_id="s1", name="Advanced Topics",
            duration_hours=3.0, deadline=(self.today + timedelta(days=5)).isoformat(),
            prerequisite_task_ids=["t1"]
        )

    def test_topological_sort_orders_prerequisites_first(self):
        sorted_tasks, has_cycle, diagnostics = AIPlanner.topological_sort([self.task2, self.task1])
        self.assertFalse(has_cycle)
        self.assertEqual(len(sorted_tasks), 2)
        self.assertEqual(sorted_tasks[0].id, "t1")
        self.assertEqual(sorted_tasks[1].id, "t2")

    def test_cycle_detection(self):
        # Create cyclic dependency
        cyclic_t1 = Task(id="c1", subject_id="s1", name="A", duration_hours=1.0, deadline="2026-12-01", prerequisite_task_ids=["c2"])
        cyclic_t2 = Task(id="c2", subject_id="s1", name="B", duration_hours=1.0, deadline="2026-12-01", prerequisite_task_ids=["c1"])
        sorted_tasks, has_cycle, diagnostics = AIPlanner.topological_sort([cyclic_t1, cyclic_t2])
        self.assertTrue(has_cycle)
        self.assertTrue(len(diagnostics) > 0)

    def test_urgency_calculation(self):
        urgent_date = (self.today + timedelta(days=1)).isoformat()
        distant_date = (self.today + timedelta(days=20)).isoformat()
        u_urgent = AIPlanner.calculate_urgency_score(urgent_date, self.today)
        u_distant = AIPlanner.calculate_urgency_score(distant_date, self.today)
        self.assertGreater(u_urgent, u_distant)

class TestAStarScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = AStarScheduler(max_expansions=1000)
        self.profile = StudentProfile(
            student_id="test_student",
            name="Test Student",
            daily_available_hours={"Monday": 4.0, "Tuesday": 4.0, "Wednesday": 4.0, "Thursday": 4.0, "Friday": 4.0, "Saturday": 4.0, "Sunday": 4.0}
        )
        today = date.today()
        self.subject = Subject(
            id="s_test", name="Algorithm Analysis", code="CS101", credits=3,
            difficulty=Difficulty.MEDIUM, importance=4, deadline=(today + timedelta(days=5)).isoformat(),
            tasks=[
                Task(id="tk1", subject_id="s_test", name="Greedy Proofs", duration_hours=2.0, deadline=(today + timedelta(days=3)).isoformat()),
                Task(id="tk2", subject_id="s_test", name="DP Optimization", duration_hours=2.0, deadline=(today + timedelta(days=4)).isoformat(), prerequisite_task_ids=["tk1"])
            ]
        )

    def test_astar_schedules_tasks_and_respects_prerequisites(self):
        report = self.scheduler.schedule([self.subject], self.profile, horizon_days=5)
        self.assertTrue(report.success or len(report.scheduled_items) > 0)
        self.assertGreaterEqual(report.total_allocated_hours, 2.0)
        self.assertGreaterEqual(report.states_explored, 1)

        # Verify prereq ordering in scheduled items
        item_tk1 = next((i for i in report.scheduled_items if i.task_id == "tk1"), None)
        item_tk2 = next((i for i in report.scheduled_items if i.task_id == "tk2"), None)
        if item_tk1 and item_tk2:
            self.assertLessEqual(item_tk1.day_index, item_tk2.day_index)

    def test_heuristic_is_non_negative(self):
        tasks = self.subject.tasks
        h = self.scheduler.compute_heuristic(tasks, 0, 7)
        self.assertGreater(h, 0.0)

class TestCSPSolver(unittest.TestCase):
    def setUp(self):
        self.solver = CSPSolver(max_backtracks=1000)
        self.profile = StudentProfile(
            student_id="test_csp",
            name="CSP Tester",
            daily_available_hours={"Monday": 3.0, "Tuesday": 3.0, "Wednesday": 3.0, "Thursday": 3.0, "Friday": 3.0, "Saturday": 3.0, "Sunday": 3.0}
        )
        today = date.today()
        self.subject = Subject(
            id="s_csp", name="Computer Systems", code="CS102", credits=4,
            difficulty=Difficulty.HARD, importance=5, deadline=(today + timedelta(days=6)).isoformat(),
            tasks=[
                Task(id="c_t1", subject_id="s_csp", name="Assembly Lab", duration_hours=2.0, deadline=(today + timedelta(days=4)).isoformat()),
                Task(id="c_t2", subject_id="s_csp", name="Pipelining", duration_hours=2.0, deadline=(today + timedelta(days=5)).isoformat(), prerequisite_task_ids=["c_t1"])
            ]
        )

    def test_csp_solver_finds_valid_assignment(self):
        report = self.solver.solve([self.subject], self.profile, horizon_days=5)
        self.assertTrue(report.success)
        self.assertEqual(len(report.scheduled_items), 2)
        # Ensure no overlap
        slots = [(i.day, i.start_hour) for i in report.scheduled_items]
        self.assertEqual(len(slots), len(set(slots)))

class TestBayesianNetwork(unittest.TestCase):
    def setUp(self):
        self.bn = StudentRiskBayesianNetwork()

    def test_cpt_sum_to_one(self):
        # Test parent CPT normalization
        for factor in self.bn.factors:
            if factor.variables == ["Workload"]:
                total = sum(factor.values.values())
                self.assertAlmostEqual(total, 1.0, places=4)

    def test_posterior_inference_sums_to_one(self):
        evidence = {"Workload": "High", "Difficulty": "Hard", "TimeRemaining": "Critical", "Progress": "Low"}
        posterior = self.bn.infer_risk_posterior(evidence)
        total = sum(posterior.values())
        self.assertAlmostEqual(total, 1.0, places=3)
        self.assertGreater(posterior["High"], posterior["Low"])

    def test_progress_reduces_deadline_risk(self):
        ev_low_prog = {"Workload": "High", "Difficulty": "Hard", "TimeRemaining": "Limited", "Progress": "Low"}
        ev_high_prog = {"Workload": "High", "Difficulty": "Hard", "TimeRemaining": "Limited", "Progress": "High"}
        post_low = self.bn.infer_risk_posterior(ev_low_prog)
        post_high = self.bn.infer_risk_posterior(ev_high_prog)
        self.assertGreater(post_low["High"], post_high["High"])

class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.test_db_path = "test_optimizer.db"
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        self.db = Database(self.test_db_path)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    def test_sample_dataset_loaded(self):
        subjects = self.db.get_all_subjects()
        self.assertEqual(len(subjects), 4)
        names = {s.name for s in subjects}
        self.assertIn("Artificial Intelligence", names)
        self.assertIn("Computer Networks", names)

if __name__ == "__main__":
    unittest.main()
