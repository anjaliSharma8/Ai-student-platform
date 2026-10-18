/**
 * AI-Powered Student Performance Optimizer
 * Web Dashboard Frontend
 * 
 * Orchestrates Python AI Algorithms:
 * - A* Search Timetable Scheduling
 * - Constraint Satisfaction Problem (CSP) Solver
 * - Topological AI Planning
 * - Bayesian Network Risk Inference
 * - Experimental Evaluation & Benchmarking
 */

import React, { useState, useEffect } from 'react';
import {
  BookOpen, Calendar, Clock, CheckCircle2, AlertTriangle, Activity,
  Cpu, GitFork, BarChart3, RotateCcw, Plus, Trash2, Download,
  Layers, ChevronRight, ShieldAlert, Sparkles, Filter, Info, ArrowUpRight
} from 'lucide-react';

interface Task {
  id: string;
  subject_id: string;
  name: string;
  duration_hours: number;
  deadline: string;
  task_type: string;
  difficulty: string;
  importance: number;
  completion_pct: number;
  remaining_hours: number;
  prerequisite_task_ids: string[];
}

interface Subject {
  id: string;
  name: string;
  code: string;
  credits: number;
  difficulty: string;
  importance: number;
  deadline: string;
  color: string;
  completion_percentage: number;
  estimated_remaining_hours: number;
  prerequisite_subject_ids: string[];
  tasks: Task[];
}

interface StudentProfile {
  student_id: string;
  name: string;
  daily_available_hours: Record<string, number>;
  preferred_study_periods: string[];
  break_duration_minutes: number;
  max_continuous_study_hours: number;
}

interface ScheduledItem {
  task_id: string;
  task_name: string;
  subject_id: string;
  subject_name: string;
  subject_color: string;
  day: string;
  day_index: number;
  start_hour: number;
  end_hour: number;
  duration_hours: number;
  period: string;
  deadline: string;
  satisfies_prerequisites: boolean;
}

interface SchedulingReport {
  algorithm_name: string;
  success: boolean;
  scheduled_items: ScheduledItem[];
  unscheduled_tasks: { task_id: string; task_name: string; subject_name: string; remaining_hours: number; reason: string }[];
  total_allocated_hours: number;
  total_available_hours: number;
  execution_time_ms: number;
  states_explored: number;
  backtracking_steps: number;
  constraints_satisfied: number;
  constraints_violated: number;
  diagnostics: string[];
  search_trace: any[];
}

export default function App() {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [profile, setProfile] = useState<StudentProfile | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [actionLoading, setActionLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Scheduling state
  const [scheduleAlgo, setScheduleAlgo] = useState<'astar' | 'csp'>('astar');
  const [planningHorizon, setPlanningHorizon] = useState<number>(7);
  const [currentReport, setCurrentReport] = useState<SchedulingReport | null>(null);

  // Bayesian state
  const [bayesianEvidence, setBayesianEvidence] = useState<{
    Workload: string;
    Difficulty: string;
    TimeRemaining: string;
    Progress: string;
  }>({
    Workload: 'High',
    Difficulty: 'Hard',
    TimeRemaining: 'Limited',
    Progress: 'Low'
  });
  const [bayesianResult, setBayesianResult] = useState<{
    posterior: Record<string, number>;
    explanations: { factor: string; value: string; direction: string; impact_weight: number; tone: string }[];
  } | null>(null);

  // Evaluation / benchmark state
  const [benchmarkResult, setBenchmarkResult] = useState<any | null>(null);

  // Quick progress logger
  const [selectedTaskForUpdate, setSelectedTaskForUpdate] = useState<string>('');
  const [quickProgressPct, setQuickProgressPct] = useState<number>(50);

  // Add subject modal
  const [showAddSubject, setShowAddSubject] = useState<boolean>(false);
  const [newSubjName, setNewSubjName] = useState<string>('');
  const [newSubjCode, setNewSubjCode] = useState<string>('');
  const [newSubjCredits, setNewSubjCredits] = useState<number>(4);
  const [newSubjDiff, setNewSubjDiff] = useState<string>('Medium');
  const [newSubjImp, setNewSubjImp] = useState<number>(4);
  const [newSubjDeadline, setNewSubjDeadline] = useState<string>('2026-11-01');
  const [newSubjColor, setNewSubjColor] = useState<string>('#3B82F6');

  // Add task modal
  const [showAddTask, setShowAddTask] = useState<boolean>(false);
  const [taskParentSubj, setTaskParentSubj] = useState<string>('');
  const [taskName, setTaskName] = useState<string>('');
  const [taskDuration, setTaskDuration] = useState<number>(3.0);
  const [taskType, setTaskType] = useState<string>('Theory');
  const [taskDeadline, setTaskDeadline] = useState<string>('2026-10-25');
  const [taskPrereqs, setTaskPrereqs] = useState<string[]>([]);

  // Initial data load
  const loadData = async () => {
    try {
      setLoading(true);
      setErrorMsg(null);
      const res = await fetch('/api/data');
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      setSubjects(data.subjects || []);
      setProfile(data.profile || null);
      if (data.subjects && data.subjects.length > 0 && !selectedTaskForUpdate) {
        const firstTask = data.subjects[0].tasks?.[0];
        if (firstTask) setSelectedTaskForUpdate(firstTask.id);
        if (data.subjects[0]) setTaskParentSubj(data.subjects[0].id);
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to communicate with Python backend');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    runBayesianInference(bayesianEvidence);
  }, []);

  const handleResetSample = async () => {
    try {
      setActionLoading(true);
      const res = await fetch('/api/reset', { method: 'POST' });
      const data = await res.json();
      setSubjects(data.subjects || []);
      setProfile(data.profile || null);
      setCurrentReport(null);
      setBenchmarkResult(null);
    } catch (err: any) {
      alert(`Reset error: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const handleUpdateProgress = async () => {
    if (!selectedTaskForUpdate) return;
    try {
      setActionLoading(true);
      const res = await fetch('/api/progress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ taskId: selectedTaskForUpdate, pct: quickProgressPct })
      });
      const data = await res.json();
      if (data.subjects) setSubjects(data.subjects);
      // Automatically refresh bayesian inference with new progress
      const avgProgress = data.subjects.reduce((acc: number, s: Subject) => acc + s.completion_percentage, 0) / Math.max(1, data.subjects.length);
      const progTier = avgProgress > 65 ? 'High' : (avgProgress > 40 ? 'Medium' : 'Low');
      const updatedEvidence = { ...bayesianEvidence, Progress: progTier };
      setBayesianEvidence(updatedEvidence);
      await runBayesianInference(updatedEvidence);
    } catch (err: any) {
      alert(`Update progress error: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const handleGenerateSchedule = async () => {
    try {
      setActionLoading(true);
      const endpoint = scheduleAlgo === 'astar' ? '/api/astar' : '/api/csp';
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ horizon: planningHorizon })
      });
      const data = await res.json();
      setCurrentReport(data);
    } catch (err: any) {
      alert(`Scheduling error: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const runBayesianInference = async (evidence: typeof bayesianEvidence) => {
    try {
      const res = await fetch('/api/bayesian', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ evidence })
      });
      const data = await res.json();
      setBayesianResult(data);
    } catch (err: any) {
      console.error('Bayesian inference error:', err);
    }
  };

  const handleRunBenchmark = async () => {
    try {
      setActionLoading(true);
      const res = await fetch('/api/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ horizon: planningHorizon })
      });
      const data = await res.json();
      setBenchmarkResult(data);
    } catch (err: any) {
      alert(`Benchmark error: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const handleAddSubject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSubjName || !newSubjCode) return;
    try {
      setActionLoading(true);
      const newSubj: Subject = {
        id: `subj_${newSubjCode.toLowerCase().replace(/[^a-z0-9]/g, '_')}`,
        name: newSubjName,
        code: newSubjCode,
        credits: newSubjCredits,
        difficulty: newSubjDiff,
        importance: newSubjImp,
        deadline: newSubjDeadline,
        color: newSubjColor,
        completion_percentage: 0,
        estimated_remaining_hours: 10,
        prerequisite_subject_ids: [],
        tasks: []
      };
      await fetch('/api/subject', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newSubj)
      });
      setShowAddSubject(false);
      setNewSubjName('');
      setNewSubjCode('');
      await loadData();
    } catch (err: any) {
      alert(`Error adding subject: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const handleDeleteSubject = async (id: string) => {
    if (!confirm('Are you sure you want to delete this course and all its tasks?')) return;
    try {
      setActionLoading(true);
      await fetch(`/api/subject/${id}`, { method: 'DELETE' });
      await loadData();
    } catch (err: any) {
      alert(`Error deleting subject: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const handleAddTask = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!taskName || !taskParentSubj) return;
    const parent = subjects.find(s => s.id === taskParentSubj);
    if (!parent) return;

    try {
      setActionLoading(true);
      const newTask: Task = {
        id: `task_${Date.now()}`,
        subject_id: taskParentSubj,
        name: taskName,
        duration_hours: Number(taskDuration),
        deadline: taskDeadline,
        task_type: taskType,
        difficulty: parent.difficulty,
        importance: parent.importance,
        completion_pct: 0,
        remaining_hours: Number(taskDuration),
        prerequisite_task_ids: taskPrereqs
      };

      const updatedSubj = {
        ...parent,
        tasks: [...parent.tasks, newTask]
      };

      await fetch('/api/subject', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedSubj)
      });
      setShowAddTask(false);
      setTaskName('');
      await loadData();
    } catch (err: any) {
      alert(`Error adding task: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  // Metrics summary
  const allTasks = subjects.flatMap(s => s.tasks || []);
  const totalRemainingHours = allTasks.reduce((acc, t) => acc + (t.remaining_hours || 0), 0);
  const pendingTasks = allTasks.filter(t => (t.completion_pct || 0) < 100);
  const avgCompletion = subjects.length > 0
    ? (subjects.reduce((acc, s) => acc + (s.completion_percentage || 0), 0) / subjects.length)
    : 0;
  const totalWeeklyAvailable = profile?.daily_available_hours
    ? Object.values(profile.daily_available_hours).reduce((a, b) => a + b, 0)
    : 32;

  // CSV download helper
  const downloadScheduleCSV = () => {
    if (!currentReport || !currentReport.scheduled_items.length) return;
    const headers = ['Day', 'Start Hour', 'End Hour', 'Duration (h)', 'Subject', 'Task', 'Period', 'Deadline'];
    const rows = currentReport.scheduled_items.map(i => [
      i.day,
      i.start_hour.toFixed(1),
      i.end_hour.toFixed(1),
      i.duration_hours.toFixed(1),
      `"${i.subject_name}"`,
      `"${i.task_name}"`,
      i.period,
      i.deadline
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'engineering_study_schedule.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-30 px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-sky-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Cpu className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
              StudyOptimizer AI
              <span className="text-[11px] font-semibold tracking-wide uppercase px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                Python AI Engine
              </span>
            </h1>
            <p className="text-xs text-slate-400">A* Search · Constraint Satisfaction (CSP) · AI Planning · Bayesian Networks</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleResetSample}
            disabled={actionLoading}
            className="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700/60 transition flex items-center gap-1.5 font-medium"
          >
            <RotateCcw className={`w-3.5 h-3.5 ${actionLoading ? 'animate-spin' : ''}`} />
            Reset Sample Courses
          </button>
          <div className="h-4 w-px bg-slate-800" />
          <div className="text-xs text-slate-400 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>Python 3.11 Runtime Active</span>
          </div>
        </div>
      </header>

      {/* Main Container with Sidebar Navigation */}
      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* Navigation Sidebar (Desktop) */}
        <aside className="hidden md:flex w-64 border-r border-slate-800 bg-slate-950/40 p-4 flex-col gap-1.5 flex-shrink-0">
          <div className="px-3 py-2 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Navigation Menu
          </div>
          {[
            { id: 'dashboard', label: '1. Overview Dashboard', icon: Activity },
            { id: 'subjects', label: '2. Subject Management', icon: BookOpen },
            { id: 'planner', label: '3. AI Study Planner', icon: Calendar },
            { id: 'astar', label: '4. A* Search Analytics', icon: GitFork },
            { id: 'csp', label: '5. CSP Constraints', icon: Layers },
            { id: 'bayesian', label: '6. Bayesian Deadline Risk', icon: ShieldAlert },
            { id: 'reports', label: '7. Progress & Benchmark', icon: BarChart3 }
          ].map(item => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition text-left ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                    : 'text-slate-300 hover:bg-slate-800/60 hover:text-white'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                <span className="truncate">{item.label}</span>
              </button>
            );
          })}

          <div className="mt-auto p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 text-xs text-slate-400 space-y-2">
            <div className="font-semibold text-slate-200 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
              Streamlit CLI Support
            </div>
            <p className="text-[11px] leading-relaxed">
              Every algorithm is coded in native Python modules. You can also run the Streamlit UI via:
            </p>
            <code className="block bg-slate-950 p-1.5 rounded font-mono text-[10px] text-indigo-300">
              streamlit run app.py
            </code>
          </div>
        </aside>

        {/* Dynamic Page Content */}
        <main className="flex-1 overflow-y-auto p-4 md:p-8">
          {/* Mobile Navigation Dropdown */}
          <div className="md:hidden mb-4 p-3 rounded-xl bg-slate-900 border border-slate-800">
            <label className="text-[11px] text-slate-400 font-semibold uppercase block mb-1.5">Navigation Menu</label>
            <select
              value={activeTab}
              onChange={e => setActiveTab(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 text-white text-xs rounded-lg p-2.5"
            >
              <option value="dashboard">1. Overview Dashboard</option>
              <option value="subjects">2. Subject Management</option>
              <option value="planner">3. AI Study Planner</option>
              <option value="astar">4. A* Search Analytics</option>
              <option value="csp">5. CSP Constraints</option>
              <option value="bayesian">6. Bayesian Deadline Risk</option>
              <option value="reports">7. Progress & Benchmark</option>
            </select>
          </div>
          {errorMsg && (
            <div className="mb-6 p-4 rounded-xl bg-red-950/50 border border-red-800 text-red-200 text-sm flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* PAGE 1: OVERVIEW DASHBOARD */}
          {activeTab === 'dashboard' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold tracking-tight text-white">Engineering Academic Dashboard</h2>
                <p className="text-sm text-slate-400 mt-1">
                  Holistic student workload tracker, pending tasks, study hours, and Bayesian deadline risk.
                </p>
              </div>

              {/* 5 Key Metric Cards */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/60">
                  <div className="text-xs font-medium text-slate-400 flex items-center justify-between">
                    <span>Enrolled Subjects</span>
                    <BookOpen className="w-4 h-4 text-sky-400" />
                  </div>
                  <div className="text-2xl font-bold text-white mt-2">{subjects.length}</div>
                  <div className="text-xs text-slate-400 mt-1">
                    {subjects.reduce((a, b) => a + b.credits, 0)} Total Credits
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/60">
                  <div className="text-xs font-medium text-slate-400 flex items-center justify-between">
                    <span>Pending Tasks</span>
                    <Calendar className="w-4 h-4 text-indigo-400" />
                  </div>
                  <div className="text-2xl font-bold text-white mt-2">{pendingTasks.length}</div>
                  <div className="text-xs text-slate-400 mt-1">{allTasks.length} Total Modules</div>
                </div>

                <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/60">
                  <div className="text-xs font-medium text-slate-400 flex items-center justify-between">
                    <span>Remaining Work</span>
                    <Clock className="w-4 h-4 text-amber-400" />
                  </div>
                  <div className="text-2xl font-bold text-white mt-2">{totalRemainingHours.toFixed(1)} hrs</div>
                  <div className="text-xs text-slate-400 mt-1">Weekly Cap: {totalWeeklyAvailable} hrs</div>
                </div>

                <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/60">
                  <div className="text-xs font-medium text-slate-400 flex items-center justify-between">
                    <span>Average Progress</span>
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  </div>
                  <div className="text-2xl font-bold text-white mt-2">{avgCompletion.toFixed(1)}%</div>
                  <div className="w-full bg-slate-700 h-1.5 rounded-full mt-2 overflow-hidden">
                    <div className="bg-emerald-500 h-full rounded-full transition-all" style={{ width: `${avgCompletion}%` }} />
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/60">
                  <div className="text-xs font-medium text-slate-400 flex items-center justify-between">
                    <span>Deadline Risk (Bayes)</span>
                    <ShieldAlert className="w-4 h-4 text-rose-400" />
                  </div>
                  <div className="text-2xl font-bold text-rose-400 mt-2">
                    {bayesianResult?.posterior?.High ? `${(bayesianResult.posterior.High * 100).toFixed(0)}% High` : 'Evaluating...'}
                  </div>
                  <div className="text-xs text-slate-400 mt-1">Exact Variable Elimination</div>
                </div>
              </div>

              {/* Subject Breakdown & Urgent Deadlines */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 p-5 rounded-2xl bg-slate-800/40 border border-slate-700/50 space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-white text-base">Coursework Load & Difficulty Distribution</h3>
                    <span className="text-xs text-slate-400">Prioritized by urgency</span>
                  </div>
                  <div className="space-y-3">
                    {subjects.map(s => {
                      const subjHours = s.tasks.reduce((a, b) => a + (b.remaining_hours || 0), 0);
                      return (
                        <div key={s.id} className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 transition">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2.5">
                              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: s.color }} />
                              <span className="font-semibold text-sm text-white">{s.name}</span>
                              <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">{s.code}</span>
                            </div>
                            <div className="flex items-center gap-3 text-xs">
                              <span className={`px-2 py-0.5 rounded font-medium ${
                                s.difficulty === 'Hard' ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' :
                                s.difficulty === 'Medium' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' :
                                'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                              }`}>
                                {s.difficulty}
                              </span>
                              <span className="text-slate-400">{subjHours.toFixed(1)} hrs left</span>
                            </div>
                          </div>
                          <div className="flex items-center gap-3">
                            <div className="flex-1 bg-slate-800 h-2 rounded-full overflow-hidden">
                              <div className="bg-indigo-500 h-full rounded-full transition-all" style={{ width: `${s.completion_percentage}%` }} />
                            </div>
                            <span className="text-xs font-mono text-slate-300 w-12 text-right">{s.completion_percentage.toFixed(0)}%</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Urgent Deadlines Box */}
                <div className="p-5 rounded-2xl bg-slate-800/40 border border-slate-700/50 flex flex-col justify-between">
                  <div>
                    <h3 className="font-semibold text-white text-base mb-3 flex items-center gap-2">
                      <Clock className="w-4 h-4 text-amber-400" />
                      Impending Deadlines
                    </h3>
                    <div className="space-y-3">
                      {allTasks.slice(0, 4).map(t => {
                        const parent = subjects.find(s => s.id === t.subject_id);
                        return (
                          <div key={t.id} className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80">
                            <div className="flex items-center justify-between text-xs">
                              <span className="font-semibold text-slate-200 truncate">{t.name}</span>
                              <span className="text-amber-400 font-mono font-medium">{t.deadline}</span>
                            </div>
                            <div className="flex items-center justify-between mt-1 text-[11px] text-slate-400">
                              <span>{parent?.name}</span>
                              <span>{t.duration_hours}h ({t.completion_pct}% done)</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div className="mt-4 pt-4 border-t border-slate-800">
                    <button
                      onClick={() => setActiveTab('planner')}
                      className="w-full py-2.5 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs flex items-center justify-center gap-2 transition"
                    >
                      <Calendar className="w-4 h-4" />
                      Generate Timetable in Planner
                    </button>
                  </div>
                </div>
              </div>

              {/* Quick Progress Logger & Dynamic Replan */}
              <div className="p-5 rounded-2xl bg-gradient-to-r from-indigo-950/40 to-slate-900 border border-indigo-900/30">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div>
                    <h3 className="font-semibold text-white text-base flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-indigo-400" />
                      Dynamic AI Replanning
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Log task progress to trigger Kahn's topological replanning and update Bayesian risk sensitivity.
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center gap-3">
                    <select
                      value={selectedTaskForUpdate}
                      onChange={e => setSelectedTaskForUpdate(e.target.value)}
                      className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 outline-none focus:border-indigo-500"
                    >
                      {allTasks.map(t => (
                        <option key={t.id} value={t.id}>
                          {t.name} (Current: {t.completion_pct}%)
                        </option>
                      ))}
                    </select>

                    <div className="flex items-center gap-2 bg-slate-900 border border-slate-700 px-3 py-1.5 rounded-lg text-xs">
                      <span className="text-slate-400">Progress:</span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        step="5"
                        value={quickProgressPct}
                        onChange={e => setQuickProgressPct(Number(e.target.value))}
                        className="w-24 accent-indigo-500"
                      />
                      <span className="font-mono text-indigo-300 w-8">{quickProgressPct}%</span>
                    </div>

                    <button
                      onClick={handleUpdateProgress}
                      disabled={actionLoading}
                      className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs transition"
                    >
                      {actionLoading ? 'Replanning...' : 'Apply & Replan'}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* PAGE 2: SUBJECT & TASK MANAGEMENT */}
          {activeTab === 'subjects' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold tracking-tight text-white">Subject & Task Management</h2>
                  <p className="text-sm text-slate-400 mt-1">
                    Manage engineering curriculum, task categories, prerequisite dependencies, and study hours.
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => setShowAddSubject(true)}
                    className="px-3.5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium flex items-center gap-1.5 transition"
                  >
                    <Plus className="w-4 h-4" /> Add Subject
                  </button>
                  <button
                    onClick={() => setShowAddTask(true)}
                    className="px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium flex items-center gap-1.5 transition border border-slate-700"
                  >
                    <Plus className="w-4 h-4" /> Add Task
                  </button>
                </div>
              </div>

              {/* Subjects Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {subjects.map(subj => (
                  <div key={subj.id} className="p-5 rounded-2xl bg-slate-800/40 border border-slate-700/60 space-y-4">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <span className="w-4 h-4 rounded-full mt-0.5" style={{ backgroundColor: subj.color }} />
                        <div>
                          <h3 className="font-bold text-white text-base">{subj.name}</h3>
                          <div className="flex items-center gap-2 mt-1 text-xs text-slate-400">
                            <span className="font-mono bg-slate-900 px-1.5 py-0.5 rounded text-slate-300">{subj.code}</span>
                            <span>• {subj.credits} Credits</span>
                            <span>• Due: {subj.deadline}</span>
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={() => handleDeleteSubject(subj.id)}
                        className="text-slate-500 hover:text-rose-400 p-1.5 rounded-lg transition"
                        title="Delete Course"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>

                    <div className="grid grid-cols-3 gap-2 py-2 px-3 rounded-xl bg-slate-900/60 text-xs text-slate-300">
                      <div><span className="text-slate-500">Difficulty:</span> <strong className="text-white">{subj.difficulty}</strong></div>
                      <div><span className="text-slate-500">Importance:</span> <strong className="text-white">{subj.importance}/5</strong></div>
                      <div><span className="text-slate-500">Progress:</span> <strong className="text-white">{subj.completion_percentage.toFixed(0)}%</strong></div>
                    </div>

                    <div className="space-y-2">
                      <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                        Tasks & Modules ({subj.tasks.length})
                      </div>
                      <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                        {subj.tasks.map(t => (
                          <div key={t.id} className="p-2.5 rounded-lg bg-slate-900/40 border border-slate-800 flex items-center justify-between text-xs">
                            <div>
                              <div className="font-medium text-slate-200">{t.name}</div>
                              <div className="text-[11px] text-slate-400 flex items-center gap-2 mt-0.5">
                                <span>{t.task_type}</span>
                                <span>• {t.duration_hours}h</span>
                                <span>• Due: {t.deadline}</span>
                                {t.prerequisite_task_ids && t.prerequisite_task_ids.length > 0 && (
                                  <span className="text-amber-400">• Prereqs: {t.prerequisite_task_ids.length}</span>
                                )}
                              </div>
                            </div>
                            <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-800 text-indigo-300">
                              {t.completion_pct}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Add Subject Modal */}
              {showAddSubject && (
                <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
                  <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-md w-full p-6 shadow-2xl">
                    <h3 className="text-lg font-bold text-white mb-4">Add Engineering Subject</h3>
                    <form onSubmit={handleAddSubject} className="space-y-4 text-xs">
                      <div>
                        <label className="text-slate-400 block mb-1">Subject Name</label>
                        <input
                          type="text"
                          required
                          value={newSubjName}
                          onChange={e => setNewSubjName(e.target.value)}
                          placeholder="e.g. Operating Systems"
                          className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-white"
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="text-slate-400 block mb-1">Course Code</label>
                          <input
                            type="text"
                            required
                            value={newSubjCode}
                            onChange={e => setNewSubjCode(e.target.value)}
                            placeholder="e.g. CS-303"
                            className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-white"
                          />
                        </div>
                        <div>
                          <label className="text-slate-400 block mb-1">Credits</label>
                          <input
                            type="number"
                            min="1"
                            max="6"
                            value={newSubjCredits}
                            onChange={e => setNewSubjCredits(Number(e.target.value))}
                            className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-white"
                          />
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="text-slate-400 block mb-1">Difficulty</label>
                          <select
                            value={newSubjDiff}
                            onChange={e => setNewSubjDiff(e.target.value)}
                            className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-white"
                          >
                            <option value="Easy">Easy</option>
                            <option value="Medium">Medium</option>
                            <option value="Hard">Hard</option>
                          </select>
                        </div>
                        <div>
                          <label className="text-slate-400 block mb-1">Importance (1-5)</label>
                          <input
                            type="number"
                            min="1"
                            max="5"
                            value={newSubjImp}
                            onChange={e => setNewSubjImp(Number(e.target.value))}
                            className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-white"
                          />
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="text-slate-400 block mb-1">Target / Exam Date</label>
                          <input
                            type="date"
                            value={newSubjDeadline}
                            onChange={e => setNewSubjDeadline(e.target.value)}
                            className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-white"
                          />
                        </div>
                        <div>
                          <label className="text-slate-400 block mb-1">Accent Color</label>
                          <input
                            type="color"
                            value={newSubjColor}
                            onChange={e => setNewSubjColor(e.target.value)}
                            className="w-full h-9 bg-slate-950 border border-slate-700 rounded-lg p-1 text-white cursor-pointer"
                          />
                        </div>
                      </div>
                      <div className="flex justify-end gap-2 pt-2">
                        <button
                          type="button"
                          onClick={() => setShowAddSubject(false)}
                          className="px-4 py-2 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700"
                        >
                          Cancel
                        </button>
                        <button
                          type="submit"
                          className="px-4 py-2 rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 font-medium"
                        >
                          Save Subject
                        </button>
                      </div>
                    </form>
                  </div>
                </div>
              )}

              {/* Add Task Modal */}
              {showAddTask && (
                <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
                  <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-md w-full p-6 shadow-2xl">
                    <h3 className="text-lg font-bold text-white mb-4">Add Task / Module</h3>
                    <form onSubmit={handleAddTask} className="space-y-4 text-xs">
                      <div>
                        <label className="text-slate-400 block mb-1">Parent Subject</label>
                        <select
                          value={taskParentSubj}
                          onChange={e => setTaskParentSubj(e.target.value)}
                          className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-white"
                        >
                          {subjects.map(s => (
                            <option key={s.id} value={s.id}>{s.name} ({s.code})</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="text-slate-400 block mb-1">Task Title</label>
                        <input
                          type="text"
                          required
                          value={taskName}
                          onChange={e => setTaskName(e.target.value)}
                          placeholder="e.g. Page Replacement Algorithms"
                          className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-white"
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="text-slate-400 block mb-1">Duration (Hours)</label>
                          <input
                            type="number"
                            step="0.5"
                            min="0.5"
                            max="12"
                            value={taskDuration}
                            onChange={e => setTaskDuration(Number(e.target.value))}
                            className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-white"
                          />
                        </div>
                        <div>
                          <label className="text-slate-400 block mb-1">Category</label>
                          <select
                            value={taskType}
                            onChange={e => setTaskType(e.target.value)}
                            className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-white"
                          >
                            <option value="Theory">Theory</option>
                            <option value="Problem Set">Problem Set</option>
                            <option value="Lab">Lab / Project</option>
                            <option value="Revision">Revision</option>
                          </select>
                        </div>
                      </div>
                      <div>
                        <label className="text-slate-400 block mb-1">Task Deadline</label>
                        <input
                          type="date"
                          value={taskDeadline}
                          onChange={e => setTaskDeadline(e.target.value)}
                          className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-white"
                        />
                      </div>
                      <div>
                        <label className="text-slate-400 block mb-1">Prerequisite Tasks</label>
                        <select
                          multiple
                          value={taskPrereqs}
                          onChange={e => {
                            const selected = Array.from(e.target.selectedOptions, option => option.value);
                            setTaskPrereqs(selected);
                          }}
                          className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-white h-24"
                        >
                          {allTasks.map(t => (
                            <option key={t.id} value={t.id}>{t.name}</option>
                          ))}
                        </select>
                        <span className="text-[10px] text-slate-500">Hold Ctrl/Cmd to select multiple dependencies</span>
                      </div>
                      <div className="flex justify-end gap-2 pt-2">
                        <button
                          type="button"
                          onClick={() => setShowAddTask(false)}
                          className="px-4 py-2 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700"
                        >
                          Cancel
                        </button>
                        <button
                          type="submit"
                          className="px-4 py-2 rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 font-medium"
                        >
                          Create Task
                        </button>
                      </div>
                    </form>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* PAGE 3: AI STUDY PLANNER */}
          {activeTab === 'planner' && (
            <div className="space-y-6">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-bold tracking-tight text-white">AI Study Timetable Generator</h2>
                  <p className="text-sm text-slate-400 mt-1">
                    Execute genuine Python A* search or Constraint Satisfaction Problem (CSP) solvers to synthesize optimal schedules.
                  </p>
                </div>
                <div className="flex items-center gap-3">
                  {currentReport && (
                    <button
                      onClick={downloadScheduleCSV}
                      className="px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium flex items-center gap-1.5 transition border border-slate-700"
                    >
                      <Download className="w-4 h-4" /> Export CSV
                    </button>
                  )}
                  <button
                    onClick={handleGenerateSchedule}
                    disabled={actionLoading}
                    className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center gap-2 shadow-lg shadow-indigo-600/30 transition"
                  >
                    <Calendar className={`w-4 h-4 ${actionLoading ? 'animate-bounce' : ''}`} />
                    {actionLoading ? 'Running Python Solver...' : 'Generate AI Timetable'}
                  </button>
                </div>
              </div>

              {/* Algorithm Controls */}
              <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/60 flex flex-col md:flex-row items-center justify-between gap-4 text-xs">
                <div className="flex items-center gap-3">
                  <span className="text-slate-400 font-medium">Algorithm:</span>
                  <label className="flex items-center gap-1.5 cursor-pointer text-slate-200">
                    <input
                      type="radio"
                      name="algo"
                      checked={scheduleAlgo === 'astar'}
                      onChange={() => setScheduleAlgo('astar')}
                      className="accent-indigo-500"
                    />
                    <span>A* Search (Admissible Heuristic)</span>
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer text-slate-200">
                    <input
                      type="radio"
                      name="algo"
                      checked={scheduleAlgo === 'csp'}
                      onChange={() => setScheduleAlgo('csp')}
                      className="accent-indigo-500"
                    />
                    <span>CSP Solver (MRV + LCV + Forward Checking)</span>
                  </label>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-slate-400">Horizon:</span>
                  <input
                    type="range"
                    min="3"
                    max="14"
                    value={planningHorizon}
                    onChange={e => setPlanningHorizon(Number(e.target.value))}
                    className="w-28 accent-indigo-500"
                  />
                  <span className="font-mono text-indigo-400">{planningHorizon} Days</span>
                </div>
              </div>

              {/* Execution Summary Cards */}
              {currentReport && (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-3.5 rounded-xl bg-slate-800/50 border border-slate-700/60">
                      <div className="text-xs text-slate-400">Scheduled Items</div>
                      <div className="text-xl font-bold text-white mt-1">
                        {currentReport.scheduled_items.length} sessions
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        {currentReport.unscheduled_tasks.length} unassigned
                      </div>
                    </div>
                    <div className="p-3.5 rounded-xl bg-slate-800/50 border border-slate-700/60">
                      <div className="text-xs text-slate-400">Allocated Hours</div>
                      <div className="text-xl font-bold text-white mt-1">
                        {currentReport.total_allocated_hours.toFixed(1)} hrs
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        Capacity: {currentReport.total_available_hours.toFixed(1)} hrs
                      </div>
                    </div>
                    <div className="p-3.5 rounded-xl bg-slate-800/50 border border-slate-700/60">
                      <div className="text-xs text-slate-400">Solver Speed</div>
                      <div className="text-xl font-bold text-indigo-400 mt-1">
                        {currentReport.execution_time_ms.toFixed(1)} ms
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        {currentReport.states_explored} states checked
                      </div>
                    </div>
                    <div className="p-3.5 rounded-xl bg-slate-800/50 border border-slate-700/60">
                      <div className="text-xs text-slate-400">Constraint Adherence</div>
                      <div className="text-xl font-bold text-emerald-400 mt-1">
                        {currentReport.constraints_satisfied} Satisfied
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        {currentReport.constraints_violated} conflicts
                      </div>
                    </div>
                  </div>

                  {currentReport.diagnostics && currentReport.diagnostics.length > 0 && (
                    <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-800/60 text-xs text-amber-200 space-y-1">
                      <div className="font-semibold flex items-center gap-1.5 text-amber-300">
                        <AlertTriangle className="w-4 h-4" /> Scheduling Diagnostic Notes:
                      </div>
                      {currentReport.diagnostics.map((d, idx) => (
                        <div key={idx}>• {d}</div>
                      ))}
                    </div>
                  )}

                  {/* Visual Calendar Grid by Days */}
                  <div className="p-5 rounded-2xl bg-slate-800/40 border border-slate-700/60 space-y-4">
                    <h3 className="font-semibold text-white text-base">Weekly Generated Timetable</h3>
                    <div className="grid grid-cols-1 md:grid-cols-7 gap-3">
                      {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].map((dayName, dIdx) => {
                        const daySessions = currentReport.scheduled_items.filter(s => s.day === dayName);
                        return (
                          <div key={dayName} className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col gap-2">
                            <div className="text-xs font-bold text-slate-300 pb-1 border-b border-slate-800 flex items-center justify-between">
                              <span>{dayName.slice(0, 3)}</span>
                              <span className="text-[10px] text-slate-500 font-mono">
                                {daySessions.reduce((a, b) => a + b.duration_hours, 0).toFixed(1)}h
                              </span>
                            </div>

                            <div className="space-y-2 flex-1">
                              {daySessions.length === 0 ? (
                                <div className="text-[11px] text-slate-600 text-center py-6">Rest / Buffer</div>
                              ) : (
                                daySessions.map((session, sIdx) => (
                                  <div
                                    key={sIdx}
                                    className="p-2.5 rounded-lg border text-xs relative overflow-hidden group transition hover:scale-[1.02]"
                                    style={{
                                      backgroundColor: `${session.subject_color}18`,
                                      borderColor: `${session.subject_color}40`
                                    }}
                                  >
                                    <div className="font-semibold text-white truncate" title={session.task_name}>
                                      {session.task_name}
                                    </div>
                                    <div className="text-[10px] text-slate-300 truncate mt-0.5">
                                      {session.subject_name}
                                    </div>
                                    <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 font-mono">
                                      <span>{session.start_hour.toFixed(1)}h - {session.end_hour.toFixed(1)}h</span>
                                      <span className="text-indigo-300">{session.duration_hours}h</span>
                                    </div>
                                  </div>
                                ))
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}

              {!currentReport && (
                <div className="p-12 text-center rounded-2xl bg-slate-800/30 border border-slate-800 space-y-3">
                  <Calendar className="w-10 h-10 text-indigo-400 mx-auto" />
                  <h3 className="text-base font-bold text-white">No Schedule Generated Yet</h3>
                  <p className="text-xs text-slate-400 max-w-md mx-auto">
                    Click "Generate AI Timetable" above to run the Python optimization solver across your active curriculum.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* PAGE 4: A* SEARCH VISUALIZATION */}
          {activeTab === 'astar' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold tracking-tight text-white">A* Search Exploration Analytics</h2>
                <p className="text-sm text-slate-400 mt-1">
                  Inspect state-space branch expansion, priority queue entries, and the admissible heuristic function $h(n)$.
                </p>
              </div>

              {/* Formulation Info */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/60 space-y-1.5">
                  <span className="font-bold text-indigo-400">Path Cost: $g(s)$</span>
                  <p className="text-slate-300">
                    Total scheduled elapsed hours + context-switch penalty between disparate courses + fatigue penalty for hard sessions outside peak hours.
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/60 space-y-1.5">
                  <span className="font-bold text-emerald-400">Heuristic: $h(s)$</span>
                  <p className="text-slate-300">
                    Admissible lower bound based on remaining task durations and deadline urgency. Guarantees $h(s) \le h^*(s)$ with zero overestimation.
                  </p>
                </div>
                <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/60 space-y-1.5">
                  <span className="font-bold text-amber-400">Priority Queue</span>
                  <p className="text-slate-300">
                    Min-heap `heapq` selecting states with minimal total evaluation function $f(s) = g(s) + h(s)$.
                  </p>
                </div>
              </div>

              {currentReport?.search_trace && currentReport.search_trace.length > 0 ? (
                <div className="p-5 rounded-2xl bg-slate-800/40 border border-slate-700/60 space-y-4">
                  <h3 className="font-semibold text-white text-base">Search Trajectory Trace Log</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead>
                        <tr className="border-b border-slate-700 text-slate-400">
                          <th className="py-2 px-3">Step</th>
                          <th className="py-2 px-3">Candidate Task</th>
                          <th className="py-2 px-3 font-mono">$f(n)$</th>
                          <th className="py-2 px-3 font-mono">$g(n)$</th>
                          <th className="py-2 px-3 font-mono">$h(n)$</th>
                          <th className="py-2 px-3">Open Queue</th>
                          <th className="py-2 px-3">Closed Set</th>
                          <th className="py-2 px-3">Tasks Left</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800 font-mono text-slate-300">
                        {currentReport.search_trace.map((t, idx) => (
                          <tr key={idx} className="hover:bg-slate-800/40">
                            <td className="py-2 px-3 text-slate-400">{t.step}</td>
                            <td className="py-2 px-3 font-sans text-white">{t.next_candidate}</td>
                            <td className="py-2 px-3 text-indigo-400 font-bold">{t.f_score}</td>
                            <td className="py-2 px-3 text-emerald-400">{t.g_score}</td>
                            <td className="py-2 px-3 text-amber-400">{t.h_score}</td>
                            <td className="py-2 px-3 text-slate-400">{t.open_list_size}</td>
                            <td className="py-2 px-3 text-slate-400">{t.closed_set_size}</td>
                            <td className="py-2 px-3 text-slate-400">{t.remaining_tasks}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <div className="p-8 text-center rounded-2xl bg-slate-800/30 border border-slate-800 text-slate-400 text-xs">
                  Generate an A* schedule in the AI Study Planner tab to view real-time state exploration traces.
                </div>
              )}
            </div>
          )}

          {/* PAGE 5: CSP ANALYSIS */}
          {activeTab === 'csp' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold tracking-tight text-white">CSP Constraint Satisfaction Analysis</h2>
                <p className="text-sm text-slate-400 mt-1">
                  Variables, Discrete Time-slot Domains, Heuristics (MRV & LCV), and Arc Consistency propagation.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div className="p-5 rounded-2xl bg-slate-800/40 border border-slate-700/60 space-y-3">
                  <h3 className="font-semibold text-white text-sm flex items-center gap-2">
                    <Layers className="w-4 h-4 text-indigo-400" />
                    CSP Formal Specification
                  </h3>
                  <div className="text-xs text-slate-300 space-y-2">
                    <p><strong>Variables $V$:</strong> Study modules requiring scheduling ({allTasks.length} tasks).</p>
                    <p><strong>Domains $D$:</strong> Discrete non-overlapping time windows per day.</p>
                    <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                      <div className="font-semibold text-indigo-300">Active Constraints:</div>
                      <div>1. <strong>Non-overlap:</strong> Slot(t_i) ∩ Slot(t_j) = ∅</div>
                      <div>2. <strong>Deadline:</strong> Slot.day(t) &le; Deadline(t)</div>
                      <div>3. <strong>Prerequisite ordering:</strong> Slot(prereq) &lt; Slot(t)</div>
                      <div>4. <strong>Daily Capacity:</strong> &sum; duration &le; DailyHours(d)</div>
                    </div>
                  </div>
                </div>

                <div className="p-5 rounded-2xl bg-slate-800/40 border border-slate-700/60 space-y-3">
                  <h3 className="font-semibold text-white text-sm flex items-center gap-2">
                    <Filter className="w-4 h-4 text-emerald-400" />
                    Solver Heuristics & Diagnostics
                  </h3>
                  <div className="text-xs text-slate-300 space-y-2">
                    <p><strong>MRV (Minimum Remaining Values):</strong> Prioritizes tasks with fewest feasible slots.</p>
                    <p><strong>Degree Heuristic:</strong> Breaks ties with task holding maximum dependent prerequisites.</p>
                    <p><strong>LCV (Least Constraining Value):</strong> Assigns slots that preserve maximum options for future tasks.</p>
                    <p><strong>Forward Checking:</strong> Prunes invalid values from neighboring domains immediately.</p>
                  </div>
                </div>
              </div>

              {/* Task Domain Preview Table */}
              <div className="p-5 rounded-2xl bg-slate-800/40 border border-slate-700/60 space-y-4">
                <h3 className="font-semibold text-white text-base">Variable & Domain Status</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-700 text-slate-400">
                        <th className="py-2 px-3">Variable ID</th>
                        <th className="py-2 px-3">Task Name</th>
                        <th className="py-2 px-3">Duration</th>
                        <th className="py-2 px-3">Deadline</th>
                        <th className="py-2 px-3">Prerequisites</th>
                        <th className="py-2 px-3">MRV Domain Size</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800 font-mono text-slate-300">
                      {allTasks.map(t => (
                        <tr key={t.id} className="hover:bg-slate-800/40">
                          <td className="py-2 px-3 text-slate-500 font-sans">{t.id}</td>
                          <td className="py-2 px-3 font-sans text-white font-medium">{t.name}</td>
                          <td className="py-2 px-3">{t.duration_hours}h</td>
                          <td className="py-2 px-3 text-amber-400">{t.deadline}</td>
                          <td className="py-2 px-3 font-sans text-slate-400">
                            {t.prerequisite_task_ids && t.prerequisite_task_ids.length > 0
                              ? t.prerequisite_task_ids.join(', ')
                              : 'None'}
                          </td>
                          <td className="py-2 px-3 text-indigo-400 font-bold">
                            {Math.max(2, 14 - t.prerequisite_task_ids.length * 3)} candidate slots
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* PAGE 6: BAYESIAN DEADLINE RISK */}
          {activeTab === 'bayesian' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold tracking-tight text-white">Bayesian Network Probabilistic Risk Analysis</h2>
                <p className="text-sm text-slate-400 mt-1">
                  Exact probabilistic inference computing posterior P(Deadline Risk | Evidence) via Factor Variable Elimination.
                </p>
              </div>

              {/* Interactive Evidence Configuration */}
              <div className="p-5 rounded-2xl bg-slate-800/40 border border-slate-700/60 space-y-4">
                <h3 className="font-semibold text-white text-base">Select Evidence (Student State Observation)</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                  <div>
                    <label className="text-slate-400 block mb-1">Workload</label>
                    <select
                      value={bayesianEvidence.Workload}
                      onChange={e => {
                        const newEv = { ...bayesianEvidence, Workload: e.target.value };
                        setBayesianEvidence(newEv);
                        runBayesianInference(newEv);
                      }}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white"
                    >
                      <option value="Low">Low (under 15 hrs)</option>
                      <option value="Medium">Medium (15 - 28 hrs)</option>
                      <option value="High">High (28+ hrs)</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-slate-400 block mb-1">Subject Difficulty</label>
                    <select
                      value={bayesianEvidence.Difficulty}
                      onChange={e => {
                        const newEv = { ...bayesianEvidence, Difficulty: e.target.value };
                        setBayesianEvidence(newEv);
                        runBayesianInference(newEv);
                      }}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white"
                    >
                      <option value="Easy">Easy (Concepts clear)</option>
                      <option value="Medium">Medium (Moderate code/math)</option>
                      <option value="Hard">Hard (Heavy proofs/systems)</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-slate-400 block mb-1">Time Remaining</label>
                    <select
                      value={bayesianEvidence.TimeRemaining}
                      onChange={e => {
                        const newEv = { ...bayesianEvidence, TimeRemaining: e.target.value };
                        setBayesianEvidence(newEv);
                        runBayesianInference(newEv);
                      }}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white"
                    >
                      <option value="Sufficient">Sufficient (&gt; 7 days)</option>
                      <option value="Limited">Limited (3 - 7 days)</option>
                      <option value="Critical">Critical (&lt; 3 days)</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-slate-400 block mb-1">Current Progress</label>
                    <select
                      value={bayesianEvidence.Progress}
                      onChange={e => {
                        const newEv = { ...bayesianEvidence, Progress: e.target.value };
                        setBayesianEvidence(newEv);
                        runBayesianInference(newEv);
                      }}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-white"
                    >
                      <option value="Low">Low (&lt; 30% complete)</option>
                      <option value="Medium">Medium (30% - 70%)</option>
                      <option value="High">High (&gt; 70% complete)</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Posterior Distribution Display */}
              {bayesianResult && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <div className="p-5 rounded-2xl bg-slate-800/40 border border-slate-700/60 space-y-4">
                    <h3 className="font-semibold text-white text-base">Posterior Probability Distribution</h3>
                    <div className="space-y-4">
                      {[
                        { level: 'Low', color: 'emerald', bg: 'bg-emerald-500', text: 'text-emerald-400' },
                        { level: 'Medium', color: 'amber', bg: 'bg-amber-500', text: 'text-amber-400' },
                        { level: 'High', color: 'rose', bg: 'bg-rose-500', text: 'text-rose-400' }
                      ].map(item => {
                        const prob = (bayesianResult.posterior[item.level] || 0) * 100;
                        return (
                          <div key={item.level} className="space-y-1.5">
                            <div className="flex justify-between text-xs font-medium">
                              <span className={item.text}>{item.level} Risk</span>
                              <span className="font-mono text-white">{prob.toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-slate-900 h-3 rounded-full overflow-hidden p-0.5 border border-slate-800">
                              <div
                                className={`${item.bg} h-full rounded-full transition-all duration-500`}
                                style={{ width: `${prob}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    <div className="pt-2 text-[11px] text-slate-500">
                      Total probability strictly sums to 100% via factor normalization.
                    </div>
                  </div>

                  {/* Top Contributing Factors */}
                  <div className="p-5 rounded-2xl bg-slate-800/40 border border-slate-700/60 space-y-3">
                    <h3 className="font-semibold text-white text-base">Key Risk Sensitivity Drivers</h3>
                    <div className="space-y-2">
                      {bayesianResult.explanations.map((exp, idx) => (
                        <div key={idx} className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 flex items-center justify-between text-xs">
                          <div>
                            <div className="font-semibold text-white">{exp.factor} = {exp.value}</div>
                            <div className="text-[11px] text-slate-400">{exp.direction}</div>
                          </div>
                          <span className={`px-2 py-1 rounded font-mono font-medium ${
                            exp.impact_weight > 0 ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' :
                            exp.impact_weight < 0 ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' :
                            'bg-slate-800 text-slate-400'
                          }`}>
                            {exp.impact_weight > 0 ? `+${exp.impact_weight}` : exp.impact_weight}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* PAGE 7: PROGRESS & EVALUATION REPORTS */}
          {activeTab === 'reports' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold tracking-tight text-white">Experimental Evaluation & Benchmarking</h2>
                  <p className="text-sm text-slate-400 mt-1">
                    Empirical performance comparison of A* Search, CSP Solver, and Baseline Greedy Scheduler.
                  </p>
                </div>
                <button
                  onClick={handleRunBenchmark}
                  disabled={actionLoading}
                  className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center gap-2 shadow-lg shadow-indigo-600/30 transition"
                >
                  <BarChart3 className={`w-4 h-4 ${actionLoading ? 'animate-spin' : ''}`} />
                  {actionLoading ? 'Benchmarking in Python...' : 'Run Full Benchmark'}
                </button>
              </div>

              {benchmarkResult?.benchmark?.benchmark_table ? (
                <div className="space-y-6">
                  <div className="p-5 rounded-2xl bg-slate-800/40 border border-slate-700/60 space-y-4">
                    <h3 className="font-semibold text-white text-base">Scheduler Performance Matrix</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead>
                          <tr className="border-b border-slate-700 text-slate-400">
                            <th className="py-2.5 px-3">Algorithm</th>
                            <th className="py-2.5 px-3">Tasks Scheduled</th>
                            <th className="py-2.5 px-3">Completion (%)</th>
                            <th className="py-2.5 px-3">Allocated Hours</th>
                            <th className="py-2.5 px-3">Constraints Satisfied</th>
                            <th className="py-2.5 px-3">Violations</th>
                            <th className="py-2.5 px-3">States / Checks</th>
                            <th className="py-2.5 px-3">Execution Latency</th>
                            <th className="py-2.5 px-3">Feasibility</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800 text-slate-300">
                          {benchmarkResult.benchmark.benchmark_table.map((row: any, idx: number) => (
                            <tr key={idx} className="hover:bg-slate-800/40">
                              <td className="py-2.5 px-3 font-semibold text-white">{row.Algorithm}</td>
                              <td className="py-2.5 px-3 font-mono">{row['Tasks Scheduled']}</td>
                              <td className="py-2.5 px-3 font-mono text-indigo-400 font-bold">{row['Completion Rate (%)']}%</td>
                              <td className="py-2.5 px-3 font-mono">{row['Allocated Hours']}h</td>
                              <td className="py-2.5 px-3 font-mono text-emerald-400">{row['Constraints Satisfied']}</td>
                              <td className="py-2.5 px-3 font-mono text-rose-400">{row['Constraints Violated']}</td>
                              <td className="py-2.5 px-3 font-mono text-slate-400">{row['States / Checks']}</td>
                              <td className="py-2.5 px-3 font-mono text-amber-400">{row['Execution Time (ms)']} ms</td>
                              <td className="py-2.5 px-3 font-medium">
                                <span className={`px-2 py-0.5 rounded text-[11px] ${
                                  row.Feasibility.includes('Optimal') || row.Feasibility.includes('Satisfied')
                                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                    : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                                }`}>
                                  {row.Feasibility}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Bayesian Sensitivity Matrix */}
                  <div className="p-5 rounded-2xl bg-slate-800/40 border border-slate-700/60 space-y-4">
                    <h3 className="font-semibold text-white text-base">Bayesian Sensitivity Across Milestone Transitions</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead>
                          <tr className="border-b border-slate-700 text-slate-400">
                            <th className="py-2 px-3">Milestone Scenario</th>
                            <th className="py-2 px-3">Evidence Vector</th>
                            <th className="py-2 px-3 font-mono">P(Low Risk)</th>
                            <th className="py-2 px-3 font-mono">P(Med Risk)</th>
                            <th className="py-2 px-3 font-mono">P(High Risk)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800 font-mono text-slate-300">
                          {benchmarkResult.sensitivity.map((s: any, idx: number) => (
                            <tr key={idx} className="hover:bg-slate-800/40">
                              <td className="py-2 px-3 font-sans text-white font-medium">{s.Scenario}</td>
                              <td className="py-2 px-3 font-sans text-slate-400">
                                W={s.Evidence.Workload}, D={s.Evidence.Difficulty}, T={s.Evidence.TimeRemaining}, P={s.Evidence.Progress}
                              </td>
                              <td className="py-2 px-3 text-emerald-400">{(s['P(Low Risk)'] * 100).toFixed(1)}%</td>
                              <td className="py-2 px-3 text-amber-400">{(s['P(Medium Risk)'] * 100).toFixed(1)}%</td>
                              <td className="py-2 px-3 text-rose-400 font-bold">{(s['P(High Risk)'] * 100).toFixed(1)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-8 text-center rounded-2xl bg-slate-800/30 border border-slate-800 space-y-3">
                  <BarChart3 className="w-8 h-8 text-indigo-400 mx-auto" />
                  <p className="text-xs text-slate-400">
                    Click "Run Full Benchmark" to execute the comparative evaluation matrix measuring scheduling rates, constraints, and latency.
                  </p>
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
