use nc_optimize::{
    distributed_benders::*, BendersProblem, BendersScenario, Bound, LpSubproblemID, Problem,
};
use nc_sparse::CooMatrix;
use std::{
    io::Write,
    process::{Command, Stdio},
};

fn block(cost: f64, demand: Option<f64>, upper: f64) -> Problem {
    let mut constraints = CooMatrix::new(usize::from(demand.is_some()), 1);
    if demand.is_some() {
        constraints.push(0, 0, 1.0).unwrap();
    }
    Problem {
        objective: vec![cost],
        constraints: constraints.to_csr().unwrap(),
        row_bounds: demand
            .into_iter()
            .map(|lower| Bound {
                lower: Some(lower),
                upper: None,
            })
            .collect(),
        var_bounds: vec![Bound {
            lower: Some(0.0),
            upper: Some(upper),
        }],
        is_integer: vec![false],
    }
}

fn scenario(name: &str, probability: f64, demand: f64, upper: f64) -> BendersScenario {
    let mut coupling = CooMatrix::new(1, 1);
    coupling.push(0, 0, 1.0).unwrap();
    BendersScenario {
        id: LpSubproblemID {
            name: name.into(),
            scenario: None,
        },
        probability,
        recourse: block(4.0, Some(demand), upper),
        coupling: coupling.to_csr().unwrap(),
    }
}

fn problem() -> BendersProblem {
    BendersProblem {
        first_stage: block(1.5, None, 10.0),
        scenarios: vec![
            scenario("normal", 0.8, 2.0, 10.0),
            scenario("peak", 0.2, 6.0, 10.0),
        ],
    }
}

fn worker(job: &RecourseJob) -> RecourseReply {
    let mut process = Command::new(env!("CARGO_BIN_EXE_nc-benders-worker"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(
        process.stdin.take().unwrap(),
        "{}",
        serde_json::to_string(job).unwrap()
    )
    .unwrap();
    let output = process.wait_with_output().unwrap();
    assert!(output.status.success());
    serde_json::from_slice(&output.stdout).unwrap()
}

fn finish(coordinator: &mut DistributedBendersCoordinator) {
    while coordinator.status() == DistributedBendersStatus::AwaitingReplies {
        for job in coordinator.pending_jobs().unwrap().into_iter().rev() {
            assert_eq!(
                coordinator.submit(worker(&job)).unwrap(),
                ReplyDisposition::Accepted
            );
        }
    }
}

#[test]
fn workers_resume_partial_batch_and_replay_without_solver() {
    let problem = problem();
    let options = DistributedBendersOptions::default();
    let mut coordinator =
        DistributedBendersCoordinator::start(problem.clone(), "round-trip".into(), options)
            .unwrap();
    let first_jobs = coordinator.pending_jobs().unwrap();
    assert_eq!(first_jobs.len(), 2);
    let early = audit_benders_proof(&problem, &coordinator.proof(), options).unwrap();
    assert_eq!(early.claim, DistributedBendersClaim::Bounds);
    assert!(early.upper_bound.is_none());
    let reply = worker(&first_jobs[1]);
    let mut unresolved = reply.clone();
    unresolved.evidence = LpWitness::Unresolved;
    assert_eq!(
        coordinator.submit(unresolved).unwrap(),
        ReplyDisposition::Retry
    );
    let mut future = reply.clone();
    future.key.round += 1;
    assert!(coordinator.submit(future).is_err());
    assert_eq!(coordinator.pending_jobs().unwrap().len(), 2);
    assert_eq!(
        coordinator.submit(reply.clone()).unwrap(),
        ReplyDisposition::Accepted
    );
    let checkpoint = serde_json::to_string(&coordinator.checkpoint()).unwrap();
    let mut corrupt: serde_json::Value = serde_json::from_str(&checkpoint).unwrap();
    corrupt["pending"]["replies"][1]["evidence"]["point"][0] = 1000.into();
    assert!(DistributedBendersCoordinator::resume(
        problem.clone(),
        serde_json::from_value(corrupt).unwrap(),
        options
    )
    .is_err());
    let mut resumed = DistributedBendersCoordinator::resume(
        problem.clone(),
        serde_json::from_str(&checkpoint).unwrap(),
        options,
    )
    .unwrap();
    assert_eq!(resumed.pending_jobs().unwrap().len(), 1);
    assert_eq!(
        resumed.submit(reply.clone()).unwrap(),
        ReplyDisposition::Duplicate
    );
    finish(&mut resumed);
    assert_eq!(resumed.status(), DistributedBendersStatus::Converged);
    assert_eq!(
        resumed.submit(reply.clone()).unwrap(),
        ReplyDisposition::Duplicate
    );
    let mut conflicting = reply;
    conflicting.evidence = LpWitness::Unresolved;
    assert!(resumed.submit(conflicting).is_err());
    let proof = serde_json::to_string(&resumed.proof()).unwrap();
    let audited = audit_benders_proof(
        &problem,
        &serde_json::from_str(&proof).unwrap(),
        DistributedBendersOptions {
            lp_iteration_limit: 1,
            ..options
        },
    )
    .unwrap();
    assert_eq!(&audited, resumed.audit());
    assert_eq!(audited.claim, DistributedBendersClaim::Optimality);
    assert!((audited.lower_bound.unwrap() - 6.2).abs() < 1e-8);
    assert_eq!(audited.incumbent.as_ref().unwrap().first_stage, [2.0]);
    let mut forged: serde_json::Value = serde_json::from_str(&proof).unwrap();
    forged["rounds"][0]["replies"][0]["evidence"]["point"][0] = 1000.into();
    assert!(
        audit_benders_proof(&problem, &serde_json::from_value(forged).unwrap(), options).is_err()
    );
    let mut wrong_model = problem;
    wrong_model.scenarios[0].recourse.row_bounds[0].lower = Some(100.0);
    assert!(audit_benders_proof(&wrong_model, &resumed.proof(), options).is_err());
}

#[test]
fn projected_farkas_and_master_infeasibility_are_audited() {
    let problem = BendersProblem {
        first_stage: block(1.5, None, 0.0),
        scenarios: vec![scenario("impossible", 1.0, 2.0, 1.0)],
    };
    let options = DistributedBendersOptions::default();
    let mut coordinator =
        DistributedBendersCoordinator::start(problem.clone(), "infeasible".into(), options)
            .unwrap();
    let job = coordinator.pending_jobs().unwrap().pop().unwrap();
    let mut invalid = worker(&job);
    assert!(matches!(invalid.evidence, LpWitness::Infeasible { .. }));
    invalid.evidence = LpWitness::Infeasible {
        row_multipliers: vec![0.0],
    };
    assert!(coordinator.submit(invalid).is_err());
    assert_eq!(coordinator.pending_jobs().unwrap().len(), 1);
    finish(&mut coordinator);
    assert_eq!(coordinator.status(), DistributedBendersStatus::Infeasible);
    let verified = audit_benders_proof(&problem, &coordinator.proof(), options).unwrap();
    assert_eq!(verified.claim, DistributedBendersClaim::Infeasibility);
    assert_eq!(verified.cuts, 1);
    assert!(verified.incumbent.is_none());
    let mut invalid = serde_json::to_value(coordinator.proof()).unwrap();
    invalid["closing_master"]["row_multipliers"][0] = 0.into();
    assert!(
        audit_benders_proof(&problem, &serde_json::from_value(invalid).unwrap(), options).is_err()
    );
}

#[test]
fn verified_round_limit_can_resume_with_more_budget() {
    let problem = problem();
    let mut coordinator = DistributedBendersCoordinator::start(
        problem.clone(),
        "limit".into(),
        DistributedBendersOptions {
            max_rounds: 1,
            ..Default::default()
        },
    )
    .unwrap();
    finish(&mut coordinator);
    assert_eq!(coordinator.status(), DistributedBendersStatus::RoundLimit);
    assert_eq!(coordinator.audit().claim, DistributedBendersClaim::Bounds);
    let mut resumed = DistributedBendersCoordinator::resume(
        problem,
        coordinator.checkpoint(),
        Default::default(),
    )
    .unwrap();
    finish(&mut resumed);
    assert_eq!(resumed.status(), DistributedBendersStatus::Converged);
}
