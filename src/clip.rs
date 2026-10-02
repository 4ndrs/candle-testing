use std::env;

use candle_core::{DType, Device, Tensor};
use candle_transformers::models::clip;

pub fn load_image_test() -> Result<(), Box<dyn std::error::Error>> {
    let path = &env::args().collect::<Vec<String>>()[1];
    let device = Device::Cpu;
    let config = clip::ClipConfig::vit_base_patch32();

    let img = image::ImageReader::open(path)?.decode()?;

    let (width, height) = (config.image_size, config.image_size);

    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );

    let img = img.to_rgb8();
    let img = img.into_raw();

    let img = Tensor::from_vec(img, (height, width, 3), &device)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?;

    let img = (&img / 255.0)?;

    let first_channel = img.get(0)?;

    let values = first_channel.flatten_all()?.to_vec1::<f32>()?;

    dbg!(img, values);

    Ok(())
}
