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

    let img = img.broadcast_sub(&mean)?;
    let img = img.broadcast_div(&std)?;

    let img = img.unsqueeze(0)?;

    let var_builder = unsafe {
        VarBuilder::from_mmaped_safetensors(&["model.safetensors"], DType::F32, &device)?
    };

    let model = clip::ClipModel::new(var_builder, &config)?;

    let tokenizer = Tokenizer::from_file("tokenizer.json").map_err(|error| format!("{error}"))?;

    let sentences = vec![
        "anime girl smiling",
        "anime boy smiling",
        "anime girl sad",
        "dog barking",
    ];

    let mut ids: Vec<_> = tokenizer
        .encode_batch(sentences, true)
        .map_err(|error| format!("{error}"))?
        .iter()
        .map(|encoding| encoding.get_ids().to_owned())
        .collect();

    let vocab = tokenizer.get_vocab(true);
    let pad_id = vocab.get("<|endoftext|>").ok_or("No pad token")?;
    let max_len = ids.iter().map(|ids| ids.len()).max().unwrap_or(0);

    let ids: Vec<_> = ids
        .iter_mut()
        .map(|id_vec| {
            let len_diff = max_len - id_vec.len();

            if len_diff > 0 {
                id_vec.extend(vec![pad_id; len_diff]);
            }

            id_vec.to_owned()
        })
        .collect();

    let input_ids = Tensor::new(ids, &device)?;

    dbg!(&input_ids);

    let (logits_per_text, logits_per_image) = model.forward(&img, &input_ids)?;

    let score = logits_per_image.to_vec2::<f32>()?;

    dbg!(score);

    Ok(())
}
