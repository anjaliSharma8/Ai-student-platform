"""
Data models and validation structures for Holistic Engineering Student Performance Optimizer.
Implements robust serialization, validation, and domain-specific schemas.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime, date
import json

class Difficulty:
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"

    @classmethod
    def weight(cls, val: str) -> float:
        mapping = {cls.EASY: 1.0, cls.MEDIUM: 1.5, cls.HARD: 2.2}
        return mapping.get(val, 1.5)

class Importance:
    LOW = 1
    MEDIUM_LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

class WorkloadLevel:
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class TimeRemainingLevel:
    SUFFICIENT = "Sufficient"
    LIMITED = "Limited"
    CRITICAL = "Critical"

class ProgressLevel:
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class DeadlineRiskLevel:
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

@dataclass
class Task:
    id: str
    subject_id: str
    name: str
    duration_hours: float
    deadline: str  # YYYY-MM-DD
    task_type: str = "Theory"  # Theory, Problem Set, Lab, Revision
    difficulty: str = Difficulty.MEDIUM
    importance: int = 3  # 1-5
    completion_pct: float = 0.0
    prerequisite_task_ids: List[str] = field(default_factory=list)
    preferred_period: Optional[str] = None  # Morning, Afternoon, Evening, Night

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["remaining_hours"] = self.remaining_hours
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        clean = dict(data)
        if "prerequisite_task_ids" not in clean or clean["prerequisite_task_ids"] is None:
            clean["prerequisite_task_ids"] = []
        return cls(**clean)

    @property
    def remaining_hours(self) -> float:
        rem_pct = max(0.0, 100.0 - float(self.completion_pct)) / 100.0
        return round(float(self.duration_hours) * rem_pct, 2)

@dataclass
class Subject:
    id: str
    name: str
    code: str
    credits: int
    difficulty: str  # Easy, Medium, Hard
    importance: int  # 1 - 5
    deadline: str  # Upcoming exam or major project deadline YYYY-MM-DD
    color: str = "#3B82F6"
    completion_percentage: float = 0.0
    estimated_remaining_hours: float = 12.0
    prerequisite_subject_ids: List[str] = field(default_factory=list)
    tasks: List[Task] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tasks"] = [t.to_dict() if isinstance(t, Task) else t for t in self.tasks]
        if self.tasks:
            d["estimated_remaining_hours"] = round(sum(t.remaining_hours for t in self.tasks), 2)
            d["completion_percentage"] = round(sum(t.completion_pct for t in self.tasks) / len(self.tasks), 1)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Subject":
        raw_tasks = data.get("tasks", [])
        tasks = [Task.from_dict(t) if isinstance(t, dict) else t for t in raw_tasks]
        clean = dict(data)
        clean["tasks"] = tasks
        if "prerequisite_subject_ids" not in clean or clean["prerequisite_subject_ids"] is None:
            clean["prerequisite_subject_ids"] = []
        return cls(**clean)

@dataclass
class FixedCommitment:
    id: str
    name: str
    day: str  # Monday, Tuesday, ...
    start_hour: float  # e.g. 9.0 for 09:00
    end_hour: float    # e.g. 11.5 for 11:30
    commitment_type: str = "Class"  # Class, Lab, Work, Personal

@dataclass
class StudentProfile:
    student_id: str
    name: str
    daily_available_hours: Dict[str, float] = field(default_factory=lambda: {
        "Monday": 4.0,
        "Tuesday": 4.5,
        "Wednesday": 4.0,
        "Thursday": 5.0,
        "Friday": 3.5,
        "Saturday": 6.0,
        "Sunday": 6.0
    })
    preferred_study_periods: List[str] = field(default_factory=lambda: ["Morning", "Evening"])
    break_duration_minutes: int = 15
    max_continuous_study_hours: float = 2.5
    fixed_commitments: List[FixedCommitment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fixed_commitments"] = [asdict(c) for c in self.fixed_commitments]
        return d

@dataclass
class TimeSlot:
    slot_id: str
    day: str
    day_index: int  # 0 for Day 1, etc.
    start_hour: float
    end_hour: float
    period: str  # Morning (8-12), Afternoon (12-17), Evening (17-21), Night (21-24)

    @property
    def duration(self) -> float:
        return round(self.end_hour - self.start_hour, 2)

@dataclass
class ScheduledItem:
    task_id: str
    task_name: str
    subject_id: str
    subject_name: str
    subject_color: str
    day: str
    day_index: int
    start_hour: float
    end_hour: float
    duration_hours: float
    period: str
    deadline: str
    satisfies_prerequisites: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SchedulingReport:
    algorithm_name: str
    success: bool
    scheduled_items: List[ScheduledItem] = field(default_factory=list)
    unscheduled_tasks: List[Dict[str, Any]] = field(default_factory=list)
    total_allocated_hours: float = 0.0
    total_available_hours: float = 0.0
    execution_time_ms: float = 0.0
    states_explored: int = 0
    backtracking_steps: int = 0
    constraints_satisfied: int = 0
    constraints_violated: int = 0
    diagnostics: List[str] = field(default_factory=list)
    search_trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_name": self.algorithm_name,
            "success": self.success,
            "scheduled_items": [item.to_dict() for item in self.scheduled_items],
            "unscheduled_tasks": self.unscheduled_tasks,
            "total_allocated_hours": round(self.total_allocated_hours, 2),
            "total_available_hours": round(self.total_available_hours, 2),
            "execution_time_ms": round(self.execution_time_ms, 2),
            "states_explored": self.states_explored,
            "backtracking_steps": self.backtracking_steps,
            "constraints_satisfied": self.constraints_satisfied,
            "constraints_violated": self.constraints_violated,
            "diagnostics": self.diagnostics,
            "search_trace": self.search_trace
        }
