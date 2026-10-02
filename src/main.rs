mod clip;
mod matrix_multiplication;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    clip::load_image_test()?;

    Ok(())
}
