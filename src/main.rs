mod clip;
mod matrix_multiplication;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    clip::second_test()?;

    Ok(())
}
