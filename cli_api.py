"""
Python CLI Bridge for AI Algorithms.
Allows external callers (Express, scripts, CLI) to invoke genuine Python AI modules
and receive validated JSON responses.
"""

import sys
import json
from models import Subject, Task, StudentProfile, Difficulty
from database import Database
from astar_scheduler import AStarScheduler
from csp_solver import CSPSolver
from planning import AIPlanner
from bayesian_network import StudentRiskBayesianNetwork
from evaluation import Evaluator

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No command provided"}))
        sys.exit(1)

    cmd = sys.argv[1]
    db = Database()

    try:
        if cmd == "get_data":
            subjects = [s.to_dict() for s in db.get_all_subjects()]
            profile = db.get_student_profile().to_dict()
            print(json.dumps({"subjects": subjects, "profile": profile}))

        elif cmd == "reset_sample":
            db.load_sample_dataset()
            subjects = [s.to_dict() for s in db.get_all_subjects()]
            profile = db.get_student_profile().to_dict()
            print(json.dumps({"status": "ok", "subjects": subjects, "profile": profile}))

        elif cmd == "update_progress":
            task_id = sys.argv[2]
            pct = float(sys.argv[3])
            subjects = db.get_all_subjects()
            plan_res = AIPlanner.replan_on_progress_update(subjects, task_id, pct)
            for s in subjects:
                db.save_subject(s)
            print(json.dumps({"status": "ok", "replan": plan_res, "subjects": [s.to_dict() for s in subjects]}))

        elif cmd == "run_astar":
            horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            subjects = db.get_all_subjects()
            profile = db.get_student_profile()
            scheduler = AStarScheduler(max_expansions=3000)
            report = scheduler.schedule(subjects, profile, horizon_days=horizon)
            print(json.dumps(report.to_dict()))

        elif cmd == "run_csp":
            horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            subjects = db.get_all_subjects()
            profile = db.get_student_profile()
            solver = CSPSolver(max_backtracks=3000)
            report = solver.solve(subjects, profile, horizon_days=horizon)
            print(json.dumps(report.to_dict()))

        elif cmd == "run_planning":
            subjects = db.get_all_subjects()
            plan_res = AIPlanner.plan_curriculum(subjects)
            print(json.dumps(plan_res))

        elif cmd == "run_bayesian":
            evidence_json = sys.argv[2] if len(sys.argv) > 2 else "{}"
            evidence = json.loads(evidence_json)
            bn = StudentRiskBayesianNetwork()
            posterior = bn.infer_risk_posterior(evidence)
            explanations = bn.explain_contributions(evidence, posterior)
            print(json.dumps({"posterior": posterior, "explanations": explanations}))

        elif cmd == "run_benchmark":
            horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            subjects = db.get_all_subjects()
            profile = db.get_student_profile()
            res = Evaluator.run_comparative_benchmark(subjects, profile, horizon_days=horizon)
            sens = Evaluator.evaluate_bayesian_sensitivity()
            print(json.dumps({"benchmark": res, "sensitivity": sens}))

        elif cmd == "save_subject":
            subj_data = json.loads(sys.argv[2])
            subj = Subject.from_dict(subj_data)
            db.save_subject(subj)
            print(json.dumps({"status": "ok"}))

        elif cmd == "delete_subject":
            subj_id = sys.argv[2]
            db.delete_subject(subj_id)
            print(json.dumps({"status": "ok"}))

        else:
            print(json.dumps({"error": f"Unknown command {cmd}"}))
            sys.exit(1)

    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))
        sys.exit(1)

if __name__ == "__main__":
    main()
