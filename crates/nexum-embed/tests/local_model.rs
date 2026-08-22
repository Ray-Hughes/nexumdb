//! Exercises the real ONNX model.
//!
//! Ignored by default: the first run downloads ~90 MB of weights. Run with
//! `cargo test -p nexum-embed --features local -- --ignored --nocapture`.

#![cfg(feature = "local")]

use nexum_embed::Embedder;
use nexum_embed::local::LocalEmbedder;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads model weights on first run"]
async fn the_local_model_produces_sensible_embeddings() {
    let embedder = LocalEmbedder::load("all-MiniLM-L6-v2")
        .await
        .expect("model should load");

    assert_eq!(embedder.dim(), 384);

    let texts: Vec<String> = [
        "The cat sat on the mat.",
        "A feline rested upon the rug.",
        "Quarterly revenue grew twelve percent.",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    let batch = embedder.embed(&texts).await.expect("embedding should work");
    assert_eq!(batch.len(), 3);
    for vector in &batch.vectors {
        assert_eq!(vector.len(), 384);
        assert!(vector.iter().all(|v| v.is_finite()), "no NaNs");
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "vectors should be unit length, got {norm}"
        );
    }

    // The paraphrase must be closer than the unrelated sentence. This is the
    // whole point of using a real model rather than the hash embedder.
    let paraphrase = cosine(&batch.vectors[0], &batch.vectors[1]);
    let unrelated = cosine(&batch.vectors[0], &batch.vectors[2]);
    println!("paraphrase={paraphrase:.4} unrelated={unrelated:.4}");
    assert!(
        paraphrase > unrelated + 0.2,
        "paraphrase ({paraphrase:.3}) should clearly beat unrelated ({unrelated:.3})"
    );
    assert!(
        paraphrase > 0.5,
        "paraphrase similarity was only {paraphrase:.3}"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads model weights on first run"]
async fn embeddings_are_stable_across_calls_and_batch_shapes() {
    let embedder = LocalEmbedder::load("all-MiniLM-L6-v2").await.unwrap();
    let text = "Provenance is a first-class concern.".to_string();

    let alone = embedder.embed(std::slice::from_ref(&text)).await.unwrap();
    // Batched alongside a much longer text, which changes the padded length.
    let padded = embedder
        .embed(&[text.clone(), "a much longer sentence ".repeat(20)])
        .await
        .unwrap();

    let drift = cosine(&alone.vectors[0], &padded.vectors[0]);
    assert!(
        drift > 0.999,
        "padding must not change a vector; similarity was {drift:.5}"
    );
}
