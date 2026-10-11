use std::env;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::{self, ClipConfig};
use tokenizers::Tokenizer;

#[allow(dead_code)]
pub fn first_test() -> Result<(), Box<dyn std::error::Error>> {
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

    let sentences = [
        "anime girl smiling",
        "anime boy smiling",
        "anime girl sad",
        "dog barking",
    ];

    let mut ids: Vec<_> = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .map_err(|error| format!("{error}"))?
        .into_iter()
        .map(|encoding| encoding.get_ids().to_vec())
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

    let (_logits_per_text, logits_per_image) = model.forward(&img, &input_ids)?;

    let score = logits_per_image.squeeze(0)?.to_vec1::<f32>()?;

    let mut scored_sentences: Vec<_> = score.into_iter().zip(sentences).collect();

    scored_sentences.sort_by(|(score_a, _), (score_b, _)| score_b.total_cmp(score_a));

    for (score, sentence) in scored_sentences {
        println!("{score} {sentence}");
    }

    Ok(())
}

fn preprocess_image(
    path: &std::path::Path,
    config: &ClipConfig,
    device: &Device,
) -> Result<Tensor, Box<dyn std::error::Error>> {
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

    Ok(img)
}

#[allow(dead_code)]
pub fn second_test() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let config = clip::ClipConfig::vit_base_patch32();
    let sentence = &env::args().collect::<Vec<_>>()[1];

    let tokenizer = Tokenizer::from_file("tokenizer.json").map_err(|error| format!("{error}"))?;

    let var_builder = unsafe {
        VarBuilder::from_mmaped_safetensors(&["model.safetensors"], DType::F32, &device)?
    };

    let model = clip::ClipModel::new(var_builder, &config)?;

    let encoding = tokenizer
        .encode(sentence.to_string(), true)
        .map_err(|error| format!("{error}"))?;

    let input_ids = Tensor::new(encoding.get_ids(), &device)?.unsqueeze(0)?;

    let paths = std::fs::read_dir("./images")?
        .map(|entry| entry.map(|entry| entry.path()))
        .filter(|result| match result {
            Ok(path) => path.is_file(),
            Err(_) => true,
        })
        .collect::<Result<Vec<_>, _>>()?;

    let images = paths
        .iter()
        .map(|path| preprocess_image(path, &config, &device))
        .collect::<Result<Vec<_>, _>>()?;

    let images = Tensor::stack(&images, 0)?;

    dbg!(&images, &input_ids);

    let (_, logits_per_image) = model.forward(&images, &input_ids)?;

    dbg!(&logits_per_image);

    let logits_per_image = logits_per_image.squeeze(1)?;

    let score = logits_per_image.to_vec1::<f32>()?;

    let mut scored_images = score.iter().zip(paths).collect::<Vec<_>>();

    scored_images.sort_by(|(score_a, _), (score_b, _)| score_b.total_cmp(score_a));

    for (score, image_path) in scored_images {
        println!("{score} {}", image_path.display())
    }

    Ok(())
}
