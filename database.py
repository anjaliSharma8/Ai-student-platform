
"""
SQLite Database module for persistent storage of subjects, tasks,
student preferences, and schedules.
"""

import sqlite3
import json
from contextlib import contextmanager
from typing import List
from datetime import date, timedelta

from models import Subject, Task, StudentProfile, Difficulty, Importance

DB_PATH = "student_optimizer.db"


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_tables()
        self._ensure_seed_data()

    @contextmanager
    def _get_connection(self):
        """Open a connection and always close it after use."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS subjects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    code TEXT NOT NULL,
                    credits INTEGER NOT NULL,
                    difficulty TEXT NOT NULL,
                    importance INTEGER NOT NULL,
                    deadline TEXT NOT NULL,
                    color TEXT NOT NULL,
                    completion_percentage REAL DEFAULT 0.0,
                    estimated_remaining_hours REAL DEFAULT 0.0,
                    prerequisite_subject_ids TEXT DEFAULT '[]'
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    duration_hours REAL NOT NULL,
                    deadline TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    importance INTEGER NOT NULL,
                    completion_pct REAL DEFAULT 0.0,
                    prerequisite_task_ids TEXT DEFAULT '[]',
                    preferred_period TEXT,
                    FOREIGN KEY (subject_id)
                        REFERENCES subjects(id) ON DELETE CASCADE
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS student_profile (
                    student_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    daily_available_hours TEXT NOT NULL,
                    preferred_study_periods TEXT NOT NULL,
                    break_duration_minutes INTEGER DEFAULT 15,
                    max_continuous_study_hours REAL DEFAULT 2.5,
                    fixed_commitments TEXT DEFAULT '[]'
                )
            """)

    def _ensure_seed_data(self):
        if not self.get_all_subjects():
            self.load_sample_dataset()

    def load_sample_dataset(self):
        """Load four demonstration subjects and their tasks."""
        today = date.today()

        sample_subjects = [
            Subject(
                id="subj_dsa",
                name="Data Structures and Algorithms",
                code="CS-201",
                credits=4,
                difficulty=Difficulty.HARD,
                importance=5,
                deadline=(today + timedelta(days=6)).isoformat(),
                color="#EF4444",
                completion_percentage=60.0,
                estimated_remaining_hours=8.0,
                prerequisite_subject_ids=[],
                tasks=[
                    Task(
                        id="task_dsa_1",
                        subject_id="subj_dsa",
                        name="Graph Algorithms & Dijkstra",
                        duration_hours=3.0,
                        deadline=(today + timedelta(days=4)).isoformat(),
                        task_type="Theory",
                        difficulty=Difficulty.HARD,
                        importance=5,
                        completion_pct=75.0,
                        prerequisite_task_ids=[]
                    ),
                    Task(
                        id="task_dsa_2",
                        subject_id="subj_dsa",
                        name="Dynamic Programming Problem Set",
                        duration_hours=4.0,
                        deadline=(today + timedelta(days=5)).isoformat(),
                        task_type="Problem Set",
                        difficulty=Difficulty.HARD,
                        importance=5,
                        completion_pct=40.0,
                        prerequisite_task_ids=["task_dsa_1"]
                    ),
                    Task(
                        id="task_dsa_3",
                        subject_id="subj_dsa",
                        name="B-Trees & Advanced Balanced Trees",
                        duration_hours=2.5,
                        deadline=(today + timedelta(days=6)).isoformat(),
                        task_type="Theory",
                        difficulty=Difficulty.MEDIUM,
                        importance=4,
                        completion_pct=30.0,
                        prerequisite_task_ids=[]
                    )
                ]
            ),
            Subject(
                id="subj_ai",
                name="Artificial Intelligence",
                code="CS-301",
                credits=4,
                difficulty=Difficulty.HARD,
                importance=5,
                deadline=(today + timedelta(days=7)).isoformat(),
                color="#8B5CF6",
                completion_percentage=45.0,
                estimated_remaining_hours=9.5,
                prerequisite_subject_ids=["subj_dsa"],
                tasks=[
                    Task(
                        id="task_ai_1",
                        subject_id="subj_ai",
                        name="A* Search & Admissible Heuristics",
                        duration_hours=3.0,
                        deadline=(today + timedelta(days=5)).isoformat(),
                        task_type="Theory",
                        difficulty=Difficulty.MEDIUM,
                        importance=5,
                        completion_pct=80.0,
                        prerequisite_task_ids=["task_dsa_1"]
                    ),
                    Task(
                        id="task_ai_2",
                        subject_id="subj_ai",
                        name="CSP Backtracking & MRV Implementation",
                        duration_hours=3.5,
                        deadline=(today + timedelta(days=6)).isoformat(),
                        task_type="Lab",
                        difficulty=Difficulty.HARD,
                        importance=4,
                        completion_pct=30.0,
                        prerequisite_task_ids=["task_ai_1"]
                    ),
                    Task(
                        id="task_ai_3",
                        subject_id="subj_ai",
                        name="Bayesian Networks & Variable Elimination",
                        duration_hours=4.0,
                        deadline=(today + timedelta(days=7)).isoformat(),
                        task_type="Theory",
                        difficulty=Difficulty.HARD,
                        importance=5,
                        completion_pct=25.0,
                        prerequisite_task_ids=[]
                    )
                ]
            ),
            Subject(
                id="subj_dbms",
                name="Database Management Systems",
                code="CS-204",
                credits=3,
                difficulty=Difficulty.MEDIUM,
                importance=4,
                deadline=(today + timedelta(days=9)).isoformat(),
                color="#10B981",
                completion_percentage=70.0,
                estimated_remaining_hours=5.5,
                prerequisite_subject_ids=["subj_dsa"],
                tasks=[
                    Task(
                        id="task_dbms_1",
                        subject_id="subj_dbms",
                        name="Relational Algebra & Normalization (BCNF)",
                        duration_hours=2.5,
                        deadline=(today + timedelta(days=7)).isoformat(),
                        task_type="Theory",
                        difficulty=Difficulty.MEDIUM,
                        importance=4,
                        completion_pct=90.0,
                        prerequisite_task_ids=[]
                    ),
                    Task(
                        id="task_dbms_2",
                        subject_id="subj_dbms",
                        name="Query Optimization & Indexing",
                        duration_hours=3.5,
                        deadline=(today + timedelta(days=9)).isoformat(),
                        task_type="Problem Set",
                        difficulty=Difficulty.HARD,
                        importance=4,
                        completion_pct=40.0,
                        prerequisite_task_ids=[
                            "task_dsa_3",
                            "task_dbms_1"
                        ]
                    )
                ]
            ),
            Subject(
                id="subj_cn",
                name="Computer Networks",
                code="CS-302",
                credits=4,
                difficulty=Difficulty.MEDIUM,
                importance=4,
                deadline=(today + timedelta(days=8)).isoformat(),
                color="#0284C7",
                completion_percentage=50.0,
                estimated_remaining_hours=6.5,
                prerequisite_subject_ids=[],
                tasks=[
                    Task(
                        id="task_cn_1",
                        subject_id="subj_cn",
                        name="TCP/IP Layer Architecture & Flow Control",
                        duration_hours=3.0,
                        deadline=(today + timedelta(days=6)).isoformat(),
                        task_type="Theory",
                        difficulty=Difficulty.MEDIUM,
                        importance=4,
                        completion_pct=60.0,
                        prerequisite_task_ids=[]
                    ),
                    Task(
                        id="task_cn_2",
                        subject_id="subj_cn",
                        name="Socket Programming & Wireshark Lab",
                        duration_hours=4.0,
                        deadline=(today + timedelta(days=8)).isoformat(),
                        task_type="Lab",
                        difficulty=Difficulty.HARD,
                        importance=5,
                        completion_pct=35.0,
                        prerequisite_task_ids=["task_cn_1"]
                    )
                ]
            )
        ]

        with self._get_connection() as conn:
            conn.execute("DELETE FROM tasks")
            conn.execute("DELETE FROM subjects")

        for subject in sample_subjects:
            self.save_subject(subject)

        default_profile = StudentProfile(
            student_id="default_student",
            name="Alex Mercer (3rd Year Computer Engineering)",
            daily_available_hours={
                "Monday": 4.5,
                "Tuesday": 4.0,
                "Wednesday": 4.5,
                "Thursday": 5.0,
                "Friday": 3.5,
                "Saturday": 6.5,
                "Sunday": 6.0
            },
            preferred_study_periods=["Morning", "Evening"],
            break_duration_minutes=15,
            max_continuous_study_hours=2.5,
            fixed_commitments=[]
        )
        self.save_student_profile(default_profile)

    def get_all_subjects(self) -> List[Subject]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM subjects"
            ).fetchall()

            subjects = []

            for row in rows:
                task_rows = conn.execute(
                    "SELECT * FROM tasks WHERE subject_id = ?",
                    (row["id"],)
                ).fetchall()

                tasks = [
                    Task(
                        id=task["id"],
                        subject_id=task["subject_id"],
                        name=task["name"],
                        duration_hours=task["duration_hours"],
                        deadline=task["deadline"],
                        task_type=task["task_type"],
                        difficulty=task["difficulty"],
                        importance=task["importance"],
                        completion_pct=task["completion_pct"],
                        prerequisite_task_ids=json.loads(
                            task["prerequisite_task_ids"] or "[]"
                        ),
                        preferred_period=task["preferred_period"]
                    )
                    for task in task_rows
                ]

                subjects.append(
                    Subject(
                        id=row["id"],
                        name=row["name"],
                        code=row["code"],
                        credits=row["credits"],
                        difficulty=row["difficulty"],
                        importance=row["importance"],
                        deadline=row["deadline"],
                        color=row["color"],
                        completion_percentage=row["completion_percentage"],
                        estimated_remaining_hours=row[
                            "estimated_remaining_hours"
                        ],
                        prerequisite_subject_ids=json.loads(
                            row["prerequisite_subject_ids"] or "[]"
                        ),
                        tasks=tasks
                    )
                )

            return subjects

    def save_subject(self, subject: Subject):
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO subjects (
                    id, name, code, credits, difficulty, importance,
                    deadline, color, completion_percentage,
                    estimated_remaining_hours, prerequisite_subject_ids
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                subject.id,
                subject.name,
                subject.code,
                subject.credits,
                subject.difficulty,
                subject.importance,
                subject.deadline,
                subject.color,
                subject.completion_percentage,
                subject.estimated_remaining_hours,
                json.dumps(subject.prerequisite_subject_ids)
            ))

            conn.execute(
                "DELETE FROM tasks WHERE subject_id = ?",
                (subject.id,)
            )

            for task in subject.tasks:
                conn.execute("""
                    INSERT INTO tasks (
                        id, subject_id, name, duration_hours, deadline,
                        task_type, difficulty, importance, completion_pct,
                        prerequisite_task_ids, preferred_period
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.id,
                    task.subject_id,
                    task.name,
                    task.duration_hours,
                    task.deadline,
                    task.task_type,
                    task.difficulty,
                    task.importance,
                    task.completion_pct,
                    json.dumps(task.prerequisite_task_ids),
                    task.preferred_period
                ))

    def delete_subject(self, subject_id: str):
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM tasks WHERE subject_id = ?",
                (subject_id,)
            )
            conn.execute(
                "DELETE FROM subjects WHERE id = ?",
                (subject_id,)
            )

    def get_student_profile(self) -> StudentProfile:
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM student_profile
                WHERE student_id = 'default_student'
            """).fetchone()

            if not row:
                return StudentProfile(
                    "default_student",
                    "Alex Mercer"
                )

            return StudentProfile(
                student_id=row["student_id"],
                name=row["name"],
                daily_available_hours=json.loads(
                    row["daily_available_hours"]
                ),
                preferred_study_periods=json.loads(
                    row["preferred_study_periods"]
                ),
                break_duration_minutes=row["break_duration_minutes"],
                max_continuous_study_hours=row[
                    "max_continuous_study_hours"
                ],
                fixed_commitments=json.loads(
                    row["fixed_commitments"] or "[]"
                )
            )

    def save_student_profile(self, profile: StudentProfile):
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO student_profile (
                    student_id, name, daily_available_hours,
                    preferred_study_periods, break_duration_minutes,
                    max_continuous_study_hours, fixed_commitments
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.student_id,
                profile.name,
                json.dumps(profile.daily_available_hours),
                json.dumps(profile.preferred_study_periods),
                profile.break_duration_minutes,
                profile.max_continuous_study_hours,
                json.dumps([
                    item.__dict__ if hasattr(item, "__dict__") else item
                    for item in profile.fixed_commitments
                ])
            ))
