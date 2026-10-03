use std::env;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip;
use tokenizers::Tokenizer;

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

    let std = Tensor::new(&[0.26862954f32, 0.26130258, 0.27577711], &device)?.reshape((3, 1, 1))?;
    let mean = Tensor::new(&[0.48145466f32, 0.4578275, 0.40821073], &device)?.reshape((3, 1, 1))?;

    let img = img.broadcast_div(&std)?;
    let img = img.broadcast_sub(&mean)?;

    let img = img.unsqueeze(0)?;

    let var_builder = unsafe {
        VarBuilder::from_mmaped_safetensors(&["model.safetensors"], DType::F32, &device)?
    };

    let model = clip::ClipModel::new(var_builder, &config)?;

    let tokenizer = Tokenizer::from_file("tokenizer.json").map_err(|error| format!("{error}"))?;

    let encoding = tokenizer
        .encode("anime girl smiling", true)
        .map_err(|error| format!("{error}"))?;

    let ids = encoding.get_ids();

    let input_ids = Tensor::new(ids, &device)?.unsqueeze(0)?;

    let (logits_per_text, logits_per_image) = model.forward(&img, &input_ids)?;

    dbg!(logits_per_text, logits_per_image);

    Ok(())
}
