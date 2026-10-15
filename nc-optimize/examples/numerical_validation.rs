use nc_optimize::{run_reference_validation, ValidationTier};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tier = match std::env::args().nth(1).as_deref() {
        None | Some("standard") => ValidationTier::Standard,
        Some("smoke") => ValidationTier::Smoke,
        Some("extended") => ValidationTier::Extended,
        Some(value) => return Err(format!("unknown validation tier: {value}").into()),
    };
    let report = run_reference_validation()?;
    let snapshot = report.snapshot("rust-numericcore", tier);
    let manifest_issues = snapshot.manifest_issues();
    print!("{}", snapshot.to_json_pretty());
    if !manifest_issues.is_empty() {
        return Err(manifest_issues.join("; ").into());
    }
    if snapshot.cases.is_empty() || snapshot.cases.iter().any(|case| !case.passed()) {
        std::process::exit(1);
    }
    Ok(())
}
