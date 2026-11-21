use std::io::{self, BufRead, Write};

use nc_optimize::distributed_benders::{solve_recourse_job, RecourseJob};

fn main() -> io::Result<()> {
    let mut output = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let line = line?;
        let response = serde_json::from_str::<RecourseJob>(&line)
            .map_err(|error| error.to_string())
            .and_then(|job| solve_recourse_job(job).map_err(|error| error.to_string()))
            .and_then(|reply| serde_json::to_string(&reply).map_err(|error| error.to_string()));
        match response {
            Ok(reply) => writeln!(output, "{reply}")?,
            Err(error) => writeln!(output, "{}", serde_json::json!({ "error": error }))?,
        }
        output.flush()?;
    }
    Ok(())
}
