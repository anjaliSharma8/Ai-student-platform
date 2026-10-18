"""
AI-Powered Student Performance Optimizer
Streamlit User Interface
Dashboard for Engineering Students managing academic workload via:
- A* Search Timetable Scheduling
- Constraint Satisfaction Problem (CSP) Solver
- Topological AI Planning & Task Decomposition
- Bayesian Network Deadline Risk Inference
- SQLite persistence & Experimental Evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import io

from database import Database
from models import Subject, Task, StudentProfile, Difficulty, Importance, DeadlineRiskLevel
from planning import AIPlanner
from astar_scheduler import AStarScheduler
from csp_solver import CSPSolver
from bayesian_network import StudentRiskBayesianNetwork
from evaluation import Evaluator

# Configure Streamlit page
st.set_page_config(
    page_title="AI Engineering Student Performance Optimizer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for academic engineering dashboard aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #4F46E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
@st.cache_resource
def get_db():
    return Database()

db = get_db()

# Navigation
st.sidebar.title("🎓 StudyOptimizer AI")
st.sidebar.caption("Advanced AI Search, Planning & Bayesian Reasoning")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Overview Dashboard",
        "2. Subject & Task Management",
        "3. AI Study Planner",
        "4. A* Search Visualization",
        "5. CSP Constraint Analysis",
        "6. Bayesian Deadline Risk",
        "7. Progress & Evaluation Reports"
    ]
)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset to Sample Engineering Dataset"):
    db.load_sample_dataset()
    st.sidebar.success("Demonstration dataset reloaded successfully!")
    st.rerun()

# -------------------------------------------------------------
# PAGE 1: OVERVIEW DASHBOARD
# -------------------------------------------------------------
if page == "1. Overview Dashboard":
    st.markdown('<div class="main-header">Engineering Academic Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Holistic engineering student workload overview, task completion status, and urgent deadlines.</div>', unsafe_allow_html=True)

    subjects = db.get_all_subjects()
    profile = db.get_student_profile()

    total_subjects = len(subjects)
    all_tasks = [t for s in subjects for t in s.tasks]
    total_tasks = len(all_tasks)
    pending_tasks = [t for t in all_tasks if t.completion_pct < 100.0]
    total_remaining_hours = sum(t.remaining_hours for t in all_tasks)
    total_weekly_avail_hours = sum(profile.daily_available_hours.values())
    avg_completion = np.mean([s.completion_percentage for s in subjects]) if subjects else 0.0

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Enrolled Subjects", total_subjects, f"{sum(s.credits for s in subjects)} Credits")
    c2.metric("Total Tasks", total_tasks, f"{len(pending_tasks)} Pending")
    c3.metric("Est. Remaining Work", f"{total_remaining_hours:.1f} hrs", f"Weekly Cap: {total_weekly_avail_hours:.0f}h")
    c4.metric("Avg Course Progress", f"{avg_completion:.1f}%")

    # Bayesian Risk Quick Metric
    bn = StudentRiskBayesianNetwork()
    quick_ev = {
        "Workload": "High" if total_remaining_hours > 25 else "Medium",
        "Difficulty": "Hard",
        "TimeRemaining": "Limited",
        "Progress": "Medium" if avg_completion > 50 else "Low"
    }
    quick_risk = bn.infer_risk_posterior(quick_ev)
    c5.metric("Systemic Deadline Risk", f"{quick_risk.get('High', 0.0)*100:.0f}% High", f"{quick_risk.get('Low', 0.0)*100:.0f}% Low")

    st.markdown("### Subject Workload Breakdown")
    col_chart, col_urgent = st.columns([3, 2])

    with col_chart:
        sub_df = pd.DataFrame([
            {
                "Subject": s.name,
                "Code": s.code,
                "Difficulty": s.difficulty,
                "Remaining Hours": sum(t.remaining_hours for t in s.tasks),
                "Completion (%)": s.completion_percentage,
                "Deadline": s.deadline
            } for s in subjects
        ])
        fig = px.bar(
            sub_df,
            x="Subject",
            y="Remaining Hours",
            color="Difficulty",
            color_discrete_map={"Easy": "#10B981", "Medium": "#F59E0B", "Hard": "#EF4444"},
            text="Remaining Hours",
            title="Estimated Remaining Study Hours per Subject"
        )
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_urgent:
        st.markdown("#### ⏳ Impending Deadlines")
        sorted_tasks = sorted(all_tasks, key=lambda t: t.deadline)
        today = date.today()
        for t in sorted_tasks[:4]:
            subj = next((s for s in subjects if s.id == t.subject_id), None)
            subj_name = subj.name if subj else "Course"
            try:
                days_left = (datetime.strptime(t.deadline, "%Y-%m-%d").date() - today).days
                badge = f"🔴 {days_left}d left" if days_left <= 3 else f"🟡 {days_left}d left"
            except Exception:
                badge = "📅 Upcoming"

            with st.container():
                st.markdown(f"**{t.name}** ({subj_name}) — `{badge}`")
                st.caption(f"Type: {t.task_type} | Duration: {t.duration_hours}h | Progress: {t.completion_pct:.0f}%")
                st.progress(int(t.completion_pct) / 100.0)

    st.markdown("### Quick Progress Logger & Dynamic Replan")
    st.info("Log your daily study progress below. Updating a task will automatically invoke the AI Planning module to recalculate critical paths and remaining workload.")
    q_col1, q_col2, q_col3 = st.columns([3, 2, 1])

    task_options = {f"{t.name} ({next((s.code for s in subjects if s.id == t.subject_id), '')})": t.id for t in all_tasks}
    selected_task_label = q_col1.selectbox("Select Task to Update", list(task_options.keys()))
    new_pct = q_col2.slider("Set New Completion Percentage", 0, 100, 50, step=5)

    if q_col3.button("Update & Replan", type="primary"):
        tid = task_options[selected_task_label]
        AIPlanner.replan_on_progress_update(subjects, tid, float(new_pct))
        for s in subjects:
            db.save_subject(s)
        st.success(f"Updated '{selected_task_label}' to {new_pct}%. Dynamic AI replanning completed!")
        st.rerun()

# -------------------------------------------------------------
# PAGE 2: SUBJECT & TASK MANAGEMENT
# -------------------------------------------------------------
elif page == "2. Subject & Task Management":
    st.markdown('<div class="main-header">Coursework & Task Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Add, modify, and configure engineering courses, tasks, prerequisites, and deadlines.</div>', unsafe_allow_html=True)

    subjects = db.get_all_subjects()

    tab1, tab2, tab3 = st.tabs(["📚 Existing Subjects & Tasks", "➕ Add New Subject", "➕ Add New Task"])

    with tab1:
        for s in subjects:
            with st.expander(f"📖 {s.name} ({s.code}) — {s.credits} Credits | {s.difficulty} | Due: {s.deadline}", expanded=True):
                c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                c1.markdown(f"**Prerequisite Subjects:** {', '.join(s.prerequisite_subject_ids) if s.prerequisite_subject_ids else 'None'}")
                c2.markdown(f"**Importance:** {s.importance}/5")
                c3.markdown(f"**Progress:** {s.completion_percentage:.1f}%")
                if c4.button(f"🗑️ Delete Subject", key=f"del_subj_{s.id}"):
                    db.delete_subject(s.id)
                    st.success(f"Deleted {s.name}")
                    st.rerun()

                st.markdown("##### Sub-tasks:")
                if not s.tasks:
                    st.caption("No tasks defined yet.")
                else:
                    task_data = []
                    for t in s.tasks:
                        task_data.append({
                            "ID": t.id,
                            "Task Name": t.name,
                            "Type": t.task_type,
                            "Hours": t.duration_hours,
                            "Remaining (h)": t.remaining_hours,
                            "Deadline": t.deadline,
                            "Progress (%)": t.completion_pct,
                            "Prerequisites": ", ".join(t.prerequisite_task_ids) if t.prerequisite_task_ids else "None"
                        })
                    st.dataframe(pd.DataFrame(task_data), use_container_width=True)

    with tab2:
        st.markdown("#### Add Engineering Course")
        with st.form("add_subject_form"):
            name = st.text_input("Course Name", placeholder="e.g. Operating Systems")
            code = st.text_input("Course Code", placeholder="e.g. CS-303")
            col_a, col_b, col_c = st.columns(3)
            credits_val = col_a.number_input("Credits", min_value=1, max_value=6, value=4)
            diff = col_b.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
            imp = col_c.slider("Importance (1-5)", 1, 5, 4)
            deadline_val = st.date_input("Course Target / Exam Date", value=date.today() + timedelta(days=14))
            color_val = st.color_picker("Display Color", "#3B82F6")

            submitted = st.form_submit_button("Save Subject", type="primary")
            if submitted:
                if not name or not code:
                    st.error("Please fill in both course name and course code.")
                else:
                    subj_id = f"subj_{code.lower().replace(' ', '_').replace('-', '_')}"
                    new_subj = Subject(
                        id=subj_id,
                        name=name,
                        code=code,
                        credits=int(credits_val),
                        difficulty=diff,
                        importance=int(imp),
                        deadline=deadline_val.isoformat(),
                        color=color_val
                    )
                    db.save_subject(new_subj)
                    st.success(f"Subject '{name}' created successfully!")
                    st.rerun()

    with tab3:
        st.markdown("#### Add Sub-task / Module")
        if not subjects:
            st.warning("Please create a subject first.")
        else:
            with st.form("add_task_form"):
                subj_choice = st.selectbox("Parent Subject", options=[s.id for s in subjects], format_func=lambda sid: next(s.name for s in subjects if s.id == sid))
                task_name = st.text_input("Task Title", placeholder="e.g. Virtual Memory Paging Problem Set")
                tc1, tc2, tc3 = st.columns(3)
                ttype = tc1.selectbox("Task Category", ["Theory", "Problem Set", "Lab", "Revision"])
                dur = tc2.number_input("Duration (hours)", min_value=0.5, max_value=12.0, value=3.0, step=0.5)
                t_deadline = tc3.date_input("Task Deadline", value=date.today() + timedelta(days=7))

                all_existing_task_ids = [t.id for s in subjects for t in s.tasks]
                selected_prereqs = st.multiselect("Prerequisite Tasks (must be completed first)", all_existing_task_ids)

                task_submitted = st.form_submit_button("Create Task", type="primary")
                if task_submitted:
                    if not task_name:
                        st.error("Please enter a task title.")
                    else:
                        parent = next(s for s in subjects if s.id == subj_choice)
                        new_t = Task(
                            id=f"task_{len(all_existing_task_ids) + 1}_{int(datetime.now().timestamp())}",
                            subject_id=subj_choice,
                            name=task_name,
                            duration_hours=float(dur),
                            deadline=t_deadline.isoformat(),
                            task_type=ttype,
                            difficulty=parent.difficulty,
                            importance=parent.importance,
                            prerequisite_task_ids=selected_prereqs
                        )
                        parent.tasks.append(new_t)
                        db.save_subject(parent)
                        st.success(f"Task '{task_name}' added to {parent.name}!")
                        st.rerun()

# -------------------------------------------------------------
# PAGE 3: AI STUDY PLANNER
# -------------------------------------------------------------
elif page == "3. AI Study Planner":
    st.markdown('<div class="main-header">AI Study Planner & Timetable Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Generates optimal study timetables using genuine Python A* search or Constraint Satisfaction Problem (CSP) solvers.</div>', unsafe_allow_html=True)

    subjects = db.get_all_subjects()
    profile = db.get_student_profile()

    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    selected_algorithm = ctrl1.radio(
        "Choose AI Scheduling Algorithm",
        ["A* Search Algorithm (State-space with Admissible Heuristic)", "Constraint Satisfaction Problem (CSP Solver with MRV & LCV)"],
        horizontal=True
    )
    horizon_days = ctrl2.slider("Planning Horizon (Days)", 3, 14, 7)
    generate_btn = ctrl3.button("🚀 Generate Schedule", type="primary")

    if generate_btn or "last_schedule_report" in st.session_state:
        if generate_btn:
            with st.spinner("Executing Python AI scheduling algorithm..."):
                if "A*" in selected_algorithm:
                    scheduler = AStarScheduler(max_expansions=3000)
                    report = scheduler.schedule(subjects, profile, horizon_days=horizon_days)
                else:
                    solver = CSPSolver(max_backtracks=3000)
                    report = solver.solve(subjects, profile, horizon_days=horizon_days)

                st.session_state["last_schedule_report"] = report
                st.session_state["last_algo_name"] = selected_algorithm
        else:
            report = st.session_state["last_schedule_report"]

        # Results summary cards
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Tasks Scheduled", len(report.scheduled_items), f"{len(report.unscheduled_tasks)} Unscheduled")
        rc2.metric("Allocated Hours", f"{report.total_allocated_hours:.1f} hrs", f"Avail: {report.total_available_hours:.1f}h")
        rc3.metric("Execution Latency", f"{report.execution_time_ms:.1f} ms")
        rc4.metric("Constraints Satisfied", report.constraints_satisfied, f"{report.constraints_violated} Conflicts")

        if report.diagnostics:
            for diag in report.diagnostics:
                st.warning(f"⚠️ {diag}")

        st.markdown("### 📅 Weekly Study Timetable")

        if not report.scheduled_items:
            st.info("No study sessions could be scheduled. Check daily hour capacity or extend deadline.")
        else:
            sched_df = pd.DataFrame([item.to_dict() for item in report.scheduled_items])
            # Display visually
            fig = px.timeline(
                sched_df,
                x_start=[datetime.strptime(f"2026-10-19 {int(r['start_hour']):02d}:{int((r['start_hour']%1)*60):02d}", "%Y-%m-%d %H:%M") for _, r in sched_df.iterrows()],
                x_end=[datetime.strptime(f"2026-10-19 {int(r['end_hour']):02d}:{int((r['end_hour']%1)*60):02d}", "%Y-%m-%d %H:%M") for _, r in sched_df.iterrows()],
                y="day",
                color="subject_name",
                hover_data=["task_name", "duration_hours", "period", "deadline"],
                title="Generated Daily Time Allocation Grid"
            )
            fig.update_yaxes(categoryorder="array", categoryarray=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            st.markdown("#### Chronological Schedule Details")
            display_table = sched_df[["day", "start_hour", "end_hour", "duration_hours", "subject_name", "task_name", "period", "deadline"]].copy()
            display_table.columns = ["Day", "Start (h)", "End (h)", "Duration (h)", "Subject", "Task", "Period", "Deadline"]
            st.dataframe(display_table, use_container_width=True)

            # Download CSV
            csv_buf = io.StringIO()
            display_table.to_csv(csv_buf, index=False)
            st.download_button(
                "📥 Download Schedule as CSV",
                data=csv_buf.getvalue(),
                file_name="engineering_study_schedule.csv",
                mime="text/csv"
            )

# -------------------------------------------------------------
# PAGE 4: A* SEARCH VISUALIZATION
# -------------------------------------------------------------
elif page == "4. A* Search Visualization":
    st.markdown('<div class="main-header">A* Search Exploration Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detailed visualization of A* state-space search, priority queue expansions, and heuristic evaluation.</div>', unsafe_allow_html=True)

    st.markdown("""
    #### Mathematical Formulation of A* Search
    - **State Representation $s$**: Partial schedule tuple `(slot_index, assigned_sessions, unscheduled_tasks, last_subject)`.
    - **Path Cost $g(s)$**: Accumulated elapsed study hours + cognitive context-switch penalty + fatigue penalty for hard subjects outside peak hours.
    - **Heuristic Function $h(s)$**: Admissible lower bound:
      $$h(s) = \\sum_{t \\in \\text{unscheduled}} \\text{remaining\\_hours}(t) + \\text{urgency\\_penalty}(t)$$
      Because every remaining task must take at least its remaining duration, $h(s) \\le h^*(s)$, guaranteeing admissibility and optimal path discovery without overestimating cost.
    - **Evaluation Function**: $f(s) = g(s) + h(s)$
    """)

    subjects = db.get_all_subjects()
    profile = db.get_student_profile()

    if st.button("Run A* Search with Trace Sampling", type="primary"):
        scheduler = AStarScheduler(max_expansions=3000)
        report = scheduler.schedule(subjects, profile, horizon_days=7)
        st.session_state["astar_trace_report"] = report

    if "astar_trace_report" in st.session_state:
        rep = st.session_state["astar_trace_report"]
        trace = rep.search_trace

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Total States Expanded", rep.states_explored)
        sc2.metric("Closed Set Size (Pruned)", rep.backtracking_steps)
        sc3.metric("Search Time", f"{rep.execution_time_ms:.1f} ms")

        if trace:
            trace_df = pd.DataFrame(trace)
            st.markdown("### Search Trajectory: $f(n)$, $g(n)$, and $h(n)$ over Iterations")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trace_df["step"], y=trace_df["f_score"], mode="lines+markers", name="Total f(n) = g + h", line=dict(color="#4F46E5", width=2)))
            fig.add_trace(go.Scatter(x=trace_df["step"], y=trace_df["g_score"], mode="lines", name="Accumulated Path Cost g(n)", line=dict(color="#10B981", dash="dash")))
            fig.add_trace(go.Scatter(x=trace_df["step"], y=trace_df["h_score"], mode="lines", name="Heuristic Estimate h(n)", line=dict(color="#F59E0B", dash="dot")))
            fig.update_layout(xaxis_title="Expansion Step", yaxis_title="Cost Value", height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Exploration Trace Log")
            st.dataframe(trace_df, use_container_width=True)

# -------------------------------------------------------------
# PAGE 5: CSP CONSTRAINT ANALYSIS
# -------------------------------------------------------------
elif page == "5. CSP Constraint Analysis":
    st.markdown('<div class="main-header">CSP Formulation & Constraint Verification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspection of Variables, Discrete Domains, Heuristics (MRV & LCV), and Constraint Propagation.</div>', unsafe_allow_html=True)

    subjects = db.get_all_subjects()
    profile = db.get_student_profile()
    all_tasks = [t for s in subjects for t in s.tasks if t.remaining_hours > 0]

    st.markdown("### 1. Variables & Discrete Domains")
    var_col, dom_col = st.columns(2)

    with var_col:
        st.markdown(f"**Variables (X):** {len(all_tasks)} Pending Task Sessions")
        var_data = [{"Task ID": t.id, "Title": t.name, "Hours": t.remaining_hours, "Deadline": t.deadline, "Prerequisites": len(t.prerequisite_task_ids)} for t in all_tasks]
        st.dataframe(pd.DataFrame(var_data), use_container_width=True)

    with dom_col:
        scheduler_helper = AStarScheduler()
        slots = scheduler_helper.generate_discrete_slots(profile, horizon_days=7)
        st.markdown(f"**Discrete Domains (D):** {len(slots)} Available Time Slots Across 7 Days")
        dom_data = [{"Slot ID": s.slot_id, "Day": s.day, "Window": f"{s.start_hour:04.1f} - {s.end_hour:04.1f}", "Period": s.period} for s in slots[:10]]
        st.dataframe(pd.DataFrame(dom_data), use_container_width=True)
        st.caption(f"... showing 10 of {len(slots)} total candidate slots.")

    st.markdown("### 2. Constraints Formulation")
    c_box = st.container()
    c_box.markdown("""
    - **Unary Constraint $C_{\\text{deadline}}$**: $\\forall t \\in X, \\quad \\text{Slot.day\\_index}(t) \\le \\text{Deadline.day\\_index}(t)$
    - **Binary Constraint $C_{\\text{prereq}}$**: If $p$ is a prerequisite of $t$, then $\\text{Slot}(p) < \\text{Slot}(t)$ (strictly prior in calendar).
    - **All-Different Overlap Constraint $C_{\\text{non-overlap}}$**: $\\forall t_i, t_j \\in X \\; (i \\ne j), \\quad \\text{Slot}(t_i) \\cap \\text{Slot}(t_j) = \\emptyset$
    - **Capacity Constraint $C_{\\text{capacity}}$**: $\\sum_{t \\text{ in day } d} \\text{duration}(t) \\le \\text{AvailableHours}(d)$
    """)

    st.markdown("### 3. Solver Execution & Diagnostics")
    if st.button("Execute CSP Solver with Backtracking & Forward Checking", type="primary"):
        solver = CSPSolver(max_backtracks=3000)
        report = solver.solve(subjects, profile, horizon_days=7)
        st.session_state["csp_report"] = report

    if "csp_report" in st.session_state:
        rep = st.session_state["csp_report"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Solver Result", "SAT (Feasible)" if rep.success else "UNSAT (Infeasible)")
        c2.metric("Constraint Checks", rep.states_explored)
        c3.metric("Backtracking Steps", rep.backtracking_steps)
        pruning_events = rep.search_trace[0].get("pruning_events", 0) if rep.search_trace else 0
        c4.metric("Forward Check Prunings", pruning_events)

        if not rep.success and rep.diagnostics:
            st.error("Constraint Conflict Analysis:")
            for d in rep.diagnostics:
                st.write(f"- {d}")

# -------------------------------------------------------------
# PAGE 6: BAYESIAN DEADLINE RISK
# -------------------------------------------------------------
elif page == "6. Bayesian Deadline Risk":
    st.markdown('<div class="main-header">Bayesian Network Probabilistic Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Quantifies the probability distribution of missing academic deadlines using exact probabilistic inference.</div>', unsafe_allow_html=True)

    bn = StudentRiskBayesianNetwork()

    st.markdown("### 1. Bayesian Directed Acyclic Graph (DAG) Structure")
    st.info("""
    **Causal Architecture:**
    - `Workload` {Low, Medium, High} ───►
    - `Difficulty` {Easy, Medium, Hard} ──► **[ Deadline Risk ]** {Low, Medium, High}
    - `Time Remaining` {Sufficient, Limited, Critical} ──►
    - `Progress` {Low, Medium, High} ──►
    """)

    st.markdown("### 2. Enter Evidence / Observed Student State")
    e1, e2, e3, e4 = st.columns(4)
    w_input = e1.selectbox("Current Workload", ["Low", "Medium", "High"], index=2)
    d_input = e2.selectbox("Subject Difficulty", ["Easy", "Medium", "Hard"], index=1)
    t_input = e3.selectbox("Time Remaining", ["Sufficient", "Limited", "Critical"], index=1)
    p_input = e4.selectbox("Current Progress", ["Low", "Medium", "High"], index=0)

    evidence = {
        "Workload": w_input,
        "Difficulty": d_input,
        "TimeRemaining": t_input,
        "Progress": p_input
    }

    posterior = bn.infer_risk_posterior(evidence)
    explanations = bn.explain_contributions(evidence, posterior)

    st.markdown("### 3. Posterior Probability Distribution: $P(\\text{Deadline Risk} \\mid \\text{Evidence})$")
    col_bars, col_details = st.columns([3, 2])

    with col_bars:
        post_df = pd.DataFrame([
            {"Risk Level": "Low Risk", "Probability (%)": posterior.get("Low", 0.0) * 100, "Color": "#10B981"},
            {"Risk Level": "Medium Risk", "Probability (%)": posterior.get("Medium", 0.0) * 100, "Color": "#F59E0B"},
            {"Risk Level": "High Risk", "Probability (%)": posterior.get("High", 0.0) * 100, "Color": "#EF4444"}
        ])
        fig = px.bar(
            post_df,
            x="Risk Level",
            y="Probability (%)",
            color="Risk Level",
            color_discrete_map={"Low Risk": "#10B981", "Medium Risk": "#F59E0B", "High Risk": "#EF4444"},
            text="Probability (%)",
            title="Posterior Distribution P(Deadline Risk | Evidence)"
        )
        fig.update_layout(yaxis=dict(range=[0, 100]), height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_details:
        st.markdown("#### Primary Contributing Drivers")
        for exp in explanations:
            badge_icon = "🔴" if exp["tone"] == "negative" else ("🟢" if exp["tone"] == "positive" else "⚪")
            st.markdown(f"{badge_icon} **{exp['factor']} = {exp['value']}**")
            st.caption(f"Effect: {exp['direction']} (Sensitivity Weight: {exp['impact_weight']:+.2f})")

        st.caption("Illustrative probabilities based on engineering coursework workload models.")

# -------------------------------------------------------------
# PAGE 7: PROGRESS & EVALUATION REPORTS
# -------------------------------------------------------------
elif page == "7. Progress & Evaluation Reports":
    st.markdown('<div class="main-header">Experimental Evaluation & Benchmarking</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical performance comparison of A* Search, CSP Solver, and Baseline Scheduler.</div>', unsafe_allow_html=True)

    subjects = db.get_all_subjects()
    profile = db.get_student_profile()

    st.markdown("### 1. Comparative Algorithm Benchmark")
    if st.button("Run Full Experimental Benchmark", type="primary"):
        with st.spinner("Evaluating A*, CSP, and Baseline greedy scheduler..."):
            benchmark_data = Evaluator.run_comparative_benchmark(subjects, profile, horizon_days=7)
            st.session_state["benchmark_data"] = benchmark_data

    if "benchmark_data" in st.session_state:
        b_df = pd.DataFrame(st.session_state["benchmark_data"]["benchmark_table"])
        st.dataframe(b_df, use_container_width=True)

        # Plotly comparison
        fig = px.bar(
            b_df,
            x="Algorithm",
            y="Execution Time (ms)",
            color="Feasibility",
            title="Execution Latency Comparison across Schedulers"
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 2. Bayesian Sensitivity Across Study Milestones")
    st.write("Demonstrates how deadline-miss risk adapts dynamically as the student logs study progress:")
    sensitivity_results = Evaluator.evaluate_bayesian_sensitivity()
    sens_df = pd.DataFrame(sensitivity_results)
    st.dataframe(sens_df, use_container_width=True)
