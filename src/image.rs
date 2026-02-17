//! 图片处理模块
//!
//! 提供图片 token 计算和缩放功能。
//!
//! # Token 计算公式（Anthropic 官方）
//! ```text
//! tokens = (width × height) / 750
//! ```
//!
//! # 缩放规则
//! 1. 长边超过 max_long_edge 时，等比缩放
//! 2. 总像素超过 max_pixels 时，等比缩放
//! 3. 多图模式（图片数 >= threshold）使用独立的像素限制配置

use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use image::AnimationDecoder;
use image::codecs::gif::GifDecoder;
use image::{DynamicImage, ImageFormat, ImageReader};
use std::io::{BufReader, Cursor};
use std::time::Duration;

use crate::model::config::CompressionConfig;

const GIF_MAX_OUTPUT_FRAMES: usize = 20;
const GIF_MAX_FPS: usize = 5;
const GIF_MIN_FRAME_DELAY: Duration = Duration::from_millis(10);
const GIF_FRAME_OUTPUT_FORMAT: &str = "jpeg";

#[derive(Debug)]
pub struct GifSamplingResult {
    pub frames: Vec<ImageProcessResult>,
    pub duration_ms: u64,
    pub source_frames: usize,
    pub sampling_interval_ms: u64,
    pub output_format: &'static str,
}

/// 图片处理结果
#[derive(Debug)]
pub struct ImageProcessResult {
    /// 处理后的 base64 数据
    pub data: String,
    /// 原始尺寸 (width, height)
    pub original_size: (u32, u32),
    /// 处理后尺寸 (width, height)
    pub final_size: (u32, u32),
    /// 估算的 token 数
    pub tokens: u64,
    /// 是否进行了缩放
    pub was_resized: bool,
    /// 是否进行了重新编码（即使无需缩放）
    ///
    /// 主要用于 GIF：即便尺寸已符合限制，也会重新编码为静态帧，
    /// 避免把“体积巨大但分辨率很小的动图”原样发送到上游导致请求体过大。
    pub was_reencoded: bool,
    /// 原始图片字节数（base64 解码后）
    pub original_bytes_len: usize,
    /// 处理后图片字节数（编码后、base64 前）
    pub final_bytes_len: usize,
}

/// 将 GIF 抽帧并重编码为多张静态图（用于降低请求体、提升“动图内容”识别效果）
///
/// 采样策略（符合你给的约束）：
/// - 总帧数不超过 `GIF_MAX_OUTPUT_FRAMES`
/// - 采样频率不超过 `GIF_MAX_FPS`（每秒最多 5 张）
/// - 当 GIF 过长导致超出总帧数时，按“秒级上限”下调采样频率（例如 8 秒 GIF → 每秒最多 2 张）
pub fn process_gif_frames(
    base64_data: &str,
    config: &CompressionConfig,
    image_count: usize,
) -> Result<GifSamplingResult, String> {
    let gif_bytes = BASE64
        .decode(base64_data)
        .map_err(|e| format!("base64 解码失败: {}", e))?;
    let original_bytes_len = gif_bytes.len();

    // Pass 1：计算时长（ms）与源帧数，用于确定采样间隔
    let (duration_ms, source_frames) = {
        let decoder = GifDecoder::new(BufReader::new(Cursor::new(&gif_bytes)))
            .map_err(|e| format!("GIF 解码失败: {}", e))?;
        let mut total = 0u64;
        let mut n = 0usize;
        for frame in decoder.into_frames() {
            let frame = frame.map_err(|e| format!("GIF 帧解码失败: {}", e))?;
            let delay = Duration::from(frame.delay()).max(GIF_MIN_FRAME_DELAY);
            total = total.saturating_add(delay.as_millis().min(u128::from(u64::MAX)) as u64);
            n += 1;
        }
        if n == 0 {
            return Err("GIF 不包含任何帧".to_string());
        }
        (total.max(1), n)
    };

    // 计算采样间隔：
    // - 优先按“每秒最多 N 张（N<=5）”控制（用户期望的直觉规则）
    // - 当 GIF 超长（duration_secs > max_frames）时，转为按 max_frames 均匀采样
    let duration_secs_ceil = ((duration_ms + 999) / 1000).max(1) as usize;
    let fps_by_total = GIF_MAX_OUTPUT_FRAMES / duration_secs_ceil; // integer fps-per-second cap
    let fps = fps_by_total.min(GIF_MAX_FPS);
    let sampling_interval_ms = if fps > 0 {
        (1000 / fps as u64).max(1000 / GIF_MAX_FPS as u64)
    } else {
        // duration_secs_ceil > GIF_MAX_OUTPUT_FRAMES：平均 < 1 fps，改为均匀抽取 max_frames 张
        ((duration_ms + GIF_MAX_OUTPUT_FRAMES as u64 - 1) / GIF_MAX_OUTPUT_FRAMES as u64).max(1)
    };

    // 根据图片数量选择像素限制（复用现有策略）
    let max_pixels = if image_count >= config.image_multi_threshold {
        config.image_max_pixels_multi
    } else {
        config.image_max_pixels_single
    };

    // Pass 2：按采样间隔选择帧并重编码
    let decoder = GifDecoder::new(BufReader::new(Cursor::new(&gif_bytes)))
        .map_err(|e| format!("GIF 解码失败: {}", e))?;

    let mut frames_out = Vec::new();
    let mut elapsed_ms = 0u64; // 当前帧起始时间
    let mut next_sample_ms = 0u64;

    for frame in decoder.into_frames() {
        if frames_out.len() >= GIF_MAX_OUTPUT_FRAMES {
            break;
        }

        let frame = frame.map_err(|e| format!("GIF 帧解码失败: {}", e))?;
        let delay = Duration::from(frame.delay()).max(GIF_MIN_FRAME_DELAY);
        let frame_start_ms = elapsed_ms;

        // 用帧起始时间做采样（避免同一帧被重复选中）
        if frame_start_ms >= next_sample_ms {
            let buffer = frame.into_buffer();
            let original_size = (buffer.width(), buffer.height());

            let (target_w, target_h) = apply_scaling_rules(
                original_size.0,
                original_size.1,
                config.image_max_long_edge,
                max_pixels,
            );
            let needs_resize = target_w != original_size.0 || target_h != original_size.1;

            let img = DynamicImage::ImageRgba8(buffer);
            let processed = if needs_resize {
                img.resize(target_w, target_h, image::imageops::FilterType::Lanczos3)
            } else {
                img
            };

            let final_size = (processed.width(), processed.height());
            let (data, final_bytes_len) = encode_image(&processed, GIF_FRAME_OUTPUT_FORMAT)?;

            frames_out.push(ImageProcessResult {
                data,
                original_size,
                final_size,
                tokens: calculate_tokens(final_size.0, final_size.1),
                was_resized: needs_resize,
                was_reencoded: true,
                original_bytes_len,
                final_bytes_len,
            });

            next_sample_ms = frame_start_ms.saturating_add(sampling_interval_ms);
        }

        elapsed_ms = elapsed_ms.saturating_add(delay.as_millis().min(u128::from(u64::MAX)) as u64);
    }

    if frames_out.is_empty() {
        return Err("GIF 抽帧结果为空".to_string());
    }

    Ok(GifSamplingResult {
        frames: frames_out,
        duration_ms,
        source_frames,
        sampling_interval_ms,
        output_format: GIF_FRAME_OUTPUT_FORMAT,
    })
}

/// 强制将任意图片重编码为指定格式（可选缩放）
///
/// 用于需要把输入格式（如 GIF）转换为上游更稳定支持的静态格式（如 JPEG）时。
pub fn process_image_to_format(
    base64_data: &str,
    output_format: &str,
    config: &CompressionConfig,
    image_count: usize,
) -> Result<ImageProcessResult, String> {
    let bytes = BASE64
        .decode(base64_data)
        .map_err(|e| format!("base64 解码失败: {}", e))?;
    let original_bytes_len = bytes.len();

    let reader = ImageReader::new(Cursor::new(&bytes))
        .with_guessed_format()
        .map_err(|e| format!("图片格式识别失败: {}", e))?;
    let original_size = reader
        .into_dimensions()
        .map_err(|e| format!("读取图片尺寸失败: {}", e))?;

    let max_pixels = if image_count >= config.image_multi_threshold {
        config.image_max_pixels_multi
    } else {
        config.image_max_pixels_single
    };

    let (target_w, target_h) = apply_scaling_rules(
        original_size.0,
        original_size.1,
        config.image_max_long_edge,
        max_pixels,
    );
    let needs_resize = target_w != original_size.0 || target_h != original_size.1;

    let img = image::load_from_memory(&bytes).map_err(|e| format!("图片加载失败: {}", e))?;
    let processed = if needs_resize {
        img.resize(target_w, target_h, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };

    let final_size = (processed.width(), processed.height());
    let (data, final_bytes_len) = encode_image(&processed, output_format)?;

    Ok(ImageProcessResult {
        data,
        original_size,
        final_size,
        tokens: calculate_tokens(final_size.0, final_size.1),
        was_resized: needs_resize,
        was_reencoded: true,
        original_bytes_len,
        final_bytes_len,
    })
}

/// 从 base64 数据计算图片 token（不缩放）
///
/// 返回 (tokens, width, height)，解析失败返回 None
pub fn estimate_image_tokens(base64_data: &str) -> Option<(u64, u32, u32)> {
    let bytes = BASE64.decode(base64_data).ok()?;
    let reader = ImageReader::new(Cursor::new(&bytes))
        .with_guessed_format()
        .ok()?;
    let (width, height) = reader.into_dimensions().ok()?;

    // 应用 Anthropic 缩放规则计算 token
    let (scaled_w, scaled_h) = apply_scaling_rules(width, height, 1568, 1_150_000);
    let tokens = calculate_tokens(scaled_w, scaled_h);

    Some((tokens, width, height))
}

/// 处理图片：根据配置缩放并返回处理结果
///
/// # 参数
/// - `base64_data`: 原始 base64 编码的图片数据
/// - `format`: 图片格式（"jpeg", "png", "gif", "webp"）
/// - `config`: 压缩配置
/// - `image_count`: 当前请求中的图片总数（用于判断是否启用多图模式）
pub fn process_image(
    base64_data: &str,
    format: &str,
    config: &CompressionConfig,
    image_count: usize,
) -> Result<ImageProcessResult, String> {
    // 解码 base64
    let bytes = BASE64
        .decode(base64_data)
        .map_err(|e| format!("base64 解码失败: {}", e))?;
    let original_bytes_len = bytes.len();

    // 先只读取图片头获取尺寸（避免不必要的全量解码）
    let reader = ImageReader::new(Cursor::new(&bytes))
        .with_guessed_format()
        .map_err(|e| format!("图片格式识别失败: {}", e))?;
    let original_size = reader
        .into_dimensions()
        .map_err(|e| format!("读取图片尺寸失败: {}", e))?;

    // 根据图片数量选择像素限制
    let max_pixels = if image_count >= config.image_multi_threshold {
        config.image_max_pixels_multi
    } else {
        config.image_max_pixels_single
    };

    // 计算目标尺寸
    let (target_w, target_h) = apply_scaling_rules(
        original_size.0,
        original_size.1,
        config.image_max_long_edge,
        max_pixels,
    );

    let needs_resize = target_w != original_size.0 || target_h != original_size.1;

    // GIF 特殊处理：即使不需要缩放，也强制重新编码为静态帧。
    // 原因：动图通常“像素不大但字节巨大”，直接透传 base64 会显著放大请求体，
    // 进而触发上游 400 Improperly formed request。
    let force_reencode = format.eq_ignore_ascii_case("gif");
    let should_decode_and_encode = needs_resize || force_reencode;

    // 仅在需要缩放或强制重编码时才全量解码图片
    let (output_data, final_size, final_bytes_len, was_reencoded) = if should_decode_and_encode {
        let img = image::load_from_memory(&bytes).map_err(|e| format!("图片加载失败: {}", e))?;
        let processed = if needs_resize {
            img.resize(target_w, target_h, image::imageops::FilterType::Lanczos3)
        } else {
            img
        };
        let size = (processed.width(), processed.height());
        let (data, bytes_len) = encode_image(&processed, format)?;
        (data, size, bytes_len, force_reencode && !needs_resize)
    } else {
        (base64_data.to_string(), original_size, original_bytes_len, false)
    };

    let tokens = calculate_tokens(final_size.0, final_size.1);

    Ok(ImageProcessResult {
        data: output_data,
        original_size,
        final_size,
        tokens,
        was_resized: needs_resize,
        was_reencoded,
        original_bytes_len,
        final_bytes_len,
    })
}

/// 应用 Anthropic 缩放规则
///
/// 1. 长边不超过 max_long_edge
/// 2. 总像素不超过 max_pixels
fn apply_scaling_rules(width: u32, height: u32, max_long_edge: u32, max_pixels: u32) -> (u32, u32) {
    let mut w = width as f64;
    let mut h = height as f64;

    // 规则 1: 长边限制
    let long_edge = w.max(h);
    if long_edge > max_long_edge as f64 {
        let scale = max_long_edge as f64 / long_edge;
        w *= scale;
        h *= scale;
    }

    // 规则 2: 总像素限制
    let pixels = w * h;
    if pixels > max_pixels as f64 {
        let scale = (max_pixels as f64 / pixels).sqrt();
        w *= scale;
        h *= scale;
    }

    (w.floor().max(1.0) as u32, h.floor().max(1.0) as u32)
}

/// 计算 token 数
#[inline]
fn calculate_tokens(width: u32, height: u32) -> u64 {
    ((width as u64 * height as u64) + 375) / 750 // 四舍五入
}

/// 将图片编码为 base64
fn encode_image(img: &DynamicImage, format: &str) -> Result<(String, usize), String> {
    let mut buffer = Cursor::new(Vec::new());

    let image_format = match format {
        "jpeg" | "jpg" => ImageFormat::Jpeg,
        "png" => ImageFormat::Png,
        "gif" => ImageFormat::Gif,
        "webp" => ImageFormat::WebP,
        _ => return Err(format!("不支持的图片格式: {}", format)),
    };

    img.write_to(&mut buffer, image_format)
        .map_err(|e| format!("图片编码失败: {}", e))?;

    let encoded = buffer.into_inner();
    let bytes_len = encoded.len();
    Ok((BASE64.encode(encoded), bytes_len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_rules() {
        // 测试长边限制
        assert_eq!(
            apply_scaling_rules(2000, 1000, 1568, 10_000_000),
            (1568, 784)
        );

        // 测试像素限制
        assert_eq!(
            apply_scaling_rules(1200, 1200, 1568, 1_000_000),
            (1000, 1000)
        );

        // 测试无需缩放
        assert_eq!(apply_scaling_rules(800, 600, 1568, 1_150_000), (800, 600));
    }

    #[test]
    fn test_calculate_tokens() {
        assert_eq!(calculate_tokens(1092, 1092), 1590); // 1:1 标准
        assert_eq!(calculate_tokens(200, 200), 53); // 小图
    }

    #[test]
    fn test_gif_is_reencoded_even_without_resize() {
        use image::codecs::gif::{GifEncoder, Repeat};
        use image::{Delay, Frame, Rgba, RgbaImage};

        // 构造一个多帧 GIF（像素小但包含多帧），用于验证“强制重编码为静态帧”的行为。
        let mut frames = Vec::new();
        for i in 0..10u8 {
            let mut img = RgbaImage::new(32, 32);
            for p in img.pixels_mut() {
                *p = Rgba([i, 255u8.saturating_sub(i), 0, 255]);
            }
            frames.push(Frame::from_parts(
                img,
                0,
                0,
                Delay::from_numer_denom_ms(10, 1),
            ));
        }

        let mut buf = Vec::new();
        {
            let mut encoder = GifEncoder::new(&mut buf);
            encoder.set_repeat(Repeat::Infinite).unwrap();
            encoder.encode_frames(frames).unwrap();
        }

        let base64_data = BASE64.encode(&buf);
        let config = CompressionConfig::default();
        let result = process_image(&base64_data, "gif", &config, 1).unwrap();

        assert!(!result.was_resized);
        assert!(result.was_reencoded);
        assert_eq!(result.original_size, result.final_size);
        assert!(result.final_bytes_len < result.original_bytes_len);
    }

    #[test]
    fn test_process_gif_frames_sampling_8s_caps_to_2fps() {
        use image::codecs::gif::{GifEncoder, Repeat};
        use image::{Delay, Frame, Rgba, RgbaImage};

        // 8 秒 GIF：按 max 20 帧限制，每秒最多 2 帧（20/8=2）
        let frame_delay = Delay::from_numer_denom_ms(100, 1); // 0.1s
        let mut frames = Vec::new();
        for i in 0..80u8 {
            let mut img = RgbaImage::new(64, 64);
            for p in img.pixels_mut() {
                *p = Rgba([i, 0, 0, 255]);
            }
            frames.push(Frame::from_parts(img, 0, 0, frame_delay));
        }

        let mut buf = Vec::new();
        {
            let mut encoder = GifEncoder::new(&mut buf);
            encoder.set_repeat(Repeat::Infinite).unwrap();
            encoder.encode_frames(frames).unwrap();
        }

        let base64_data = BASE64.encode(&buf);
        let config = CompressionConfig::default();
        let res = process_gif_frames(&base64_data, &config, 1).unwrap();

        assert_eq!(res.duration_ms, 8000);
        assert_eq!(res.sampling_interval_ms, 500);
        assert_eq!(res.frames.len(), 16);
        assert!(res.frames.len() <= GIF_MAX_OUTPUT_FRAMES);
        assert_eq!(res.output_format, GIF_FRAME_OUTPUT_FORMAT);
    }

    #[test]
    fn test_process_gif_frames_sampling_4s_hits_5fps_and_20_frames() {
        use image::codecs::gif::{GifEncoder, Repeat};
        use image::{Delay, Frame, Rgba, RgbaImage};

        // 4 秒 GIF：每秒最多 5 帧（20/4=5），应能采满 20 帧
        let frame_delay = Delay::from_numer_denom_ms(100, 1); // 0.1s
        let mut frames = Vec::new();
        for i in 0..40u8 {
            let mut img = RgbaImage::new(64, 64);
            for p in img.pixels_mut() {
                *p = Rgba([0, i, 0, 255]);
            }
            frames.push(Frame::from_parts(img, 0, 0, frame_delay));
        }

        let mut buf = Vec::new();
        {
            let mut encoder = GifEncoder::new(&mut buf);
            encoder.set_repeat(Repeat::Infinite).unwrap();
            encoder.encode_frames(frames).unwrap();
        }

        let base64_data = BASE64.encode(&buf);
        let config = CompressionConfig::default();
        let res = process_gif_frames(&base64_data, &config, 1).unwrap();

        assert_eq!(res.duration_ms, 4000);
        assert_eq!(res.sampling_interval_ms, 200);
        assert_eq!(res.frames.len(), 20);
        assert!(res.frames.len() <= GIF_MAX_OUTPUT_FRAMES);
        assert_eq!(res.output_format, GIF_FRAME_OUTPUT_FORMAT);
    }
}
