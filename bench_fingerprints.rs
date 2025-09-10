#![doc = r#"# StringWa.rs: Text Fingerprinting Benchmarks

This file benchmarks different libraries implementing text fingerprinting and feature
extraction, including SimHash for perceptual document similarity and TF-IDF for
statistical text analysis and feature weighting.

The input file is tokenized into lines or words and each token is processed for
fingerprint generation. SimHash generates compact 64-bit fingerprints for near-duplicate
detection, while TF-IDF produces weighted feature vectors for statistical analysis.

## Usage Examples

The benchmarks use environment variables to control the input dataset and mode:

- `STRINGWARS_DATASET`: Path to the input dataset file.
- `STRINGWARS_TOKENS`: Specifies how to interpret the input. Allowed values:
  - `lines`: Process the dataset line by line.
  - `words`: Process the dataset word by word.
- `STRINGWARS_BATCH`: Number of documents to process in each batch (default: 1024).

```sh
RUSTFLAGS="-C target-cpu=native" \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_BATCH=1024 \
    STRINGWARS_TOKENS=lines \
    cargo criterion --features bench_fingerprints bench_fingerprints --jobs 8
```
"#]
use std::env;
use std::fs;

use criterion::{Criterion, Throughput};
use fork_union::count_logical_cores;

use count_min_sketch::CountMinSketch8;
use probabilistic_collections::count_min_sketch::CountMinSketch as ProbCountMinSketch;
use probabilistic_collections::similarity::MinHash;
use stringzilla::szs::{DeviceScope, Fingerprints};

fn configure_bench() -> Criterion {
    // Use environment variable to choose between quick and thorough benchmarking
    let quick_mode = std::env::var("STRINGWARS_QUICK").is_ok();

    if quick_mode {
        Criterion::default()
            .sample_size(10)
            .warm_up_time(std::time::Duration::from_secs(1))
            .measurement_time(std::time::Duration::from_secs(5))
    } else {
        Criterion::default()
            .sample_size(1000)
            .warm_up_time(std::time::Duration::from_secs(10))
            .measurement_time(std::time::Duration::from_secs(120))
    }
}

fn bench_fingerprints(c: &mut Criterion) {
    let dataset_path =
        env::var("STRINGWARS_DATASET").expect("STRINGWARS_DATASET environment variable not set");
    let mode = env::var("STRINGWARS_TOKENS").unwrap_or_else(|_| "lines".to_string());
    let content = fs::read_to_string(&dataset_path).expect("Could not read dataset");

    let batch_size = env::var("STRINGWARS_BATCH")
        .unwrap_or_else(|_| "1024".to_string())
        .parse::<usize>()
        .expect("STRINGWARS_BATCH must be a number");

    let max_docs = env::var("STRINGWARS_MAX_DOCS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(10000);

    let documents: Vec<&str> = match mode.as_str() {
        "words" => content.split_whitespace().collect(),
        "lines" => content.lines().collect(),
        other => panic!(
            "Unknown STRINGWARS_TOKENS: {}. Use 'lines' or 'words'.",
            other
        ),
    };

    if documents.is_empty() {
        panic!("Dataset must contain at least one document for fingerprinting.");
    }

    let documents = if documents.len() > max_docs {
        &documents[..max_docs]
    } else {
        &documents
    };

    // Calculate average document size for throughput reporting
    let avg_doc_size: u64 =
        documents.iter().map(|doc| doc.len() as u64).sum::<u64>() / documents.len() as u64;

    // Unified fingerprinting benchmarks
    let mut g = c.benchmark_group("fingerprinting");
    g.throughput(Throughput::Elements(batch_size as u64));
    perform_fingerprint_benchmarks(&mut g, documents, batch_size);
    g.finish();
}

fn perform_fingerprint_benchmarks(
    g: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    documents: &[&str],
    batch_size: usize,
) {
    // Create CPU and GPU device scopes like in bench_similarities.rs
    let num_cores = count_logical_cores();
    let cpu = DeviceScope::cpu_cores(num_cores).expect("Failed to create CPU device scope");
    let maybe_gpu = DeviceScope::gpu_device(0);

    let dimensions = 256;
    let cpu_engine = Fingerprints::builder()
        .ascii()
        .dimensions(dimensions)
        .build(&cpu)
        .expect("Failed to create CPU fingerprinting engine");

    // StringZilla MinHash generation (CPU)
    g.bench_function("stringzilla::Fingerprints::CPU", |b| {
        let mut start_index = 0;
        b.iter(|| {
            let batch_docs: Vec<&str> = (0..batch_size)
                .map(|_| {
                    let doc = documents[start_index % documents.len()];
                    start_index = (start_index + 1) % documents.len();
                    doc
                })
                .collect();

            let (min_hashes, _) = cpu_engine
                .compute(&cpu, &batch_docs, dimensions)
                .expect("Failed to compute fingerprints");
            min_hashes
        })
    });

    // StringZilla MinHash generation (GPU, if available)
    if let Ok(gpu) = maybe_gpu {
        if let Ok(gpu_engine) = Fingerprints::builder()
            .ascii()
            .dimensions(dimensions)
            .build(&gpu)
        {
            g.bench_function("stringzilla::Fingerprints::GPU", |b| {
                let mut start_index = 0;
                b.iter(|| {
                    let batch_docs: Vec<&str> = (0..batch_size)
                        .map(|_| {
                            let doc = documents[start_index % documents.len()];
                            start_index = (start_index + 1) % documents.len();
                            doc
                        })
                        .collect();

                    let (min_hashes, _) = gpu_engine
                        .compute(&gpu, &batch_docs, dimensions)
                        .expect("Failed to compute fingerprints");
                    min_hashes
                })
            });
        }
    }

    g.bench_function("probabilistic_collections::minhash", |b| {
        let mut start_index = 0;
        b.iter(|| {
            let mut fingerprints = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let doc = documents[start_index % documents.len()];
                let words: Vec<&str> = doc.split_whitespace().collect();
                let minhash = MinHash::new(128);
                let min_hashes = minhash.get_min_hashes(words.iter());
                fingerprints.push(min_hashes);
                start_index = (start_index + 1) % documents.len();
            }
            fingerprints
        })
    });

    // TODO: finch MinHash - temporarily disabled due to API complexity
    // {
    //     g.bench_function("finch::minhash", |b| {
    //         let mut start_index = 0;
    //         b.iter(|| {
    //             let mut fingerprints = Vec::with_capacity(batch_size);
    //             for _ in 0..batch_size {
    //                 let _doc = documents[start_index % documents.len()];
    //                 let sketch = Sketch::new_empty();
    //                 fingerprints.push(sketch);
    //                 start_index = (start_index + 1) % documents.len();
    //             }
    //             fingerprints
    //         })
    //     });
    // }

    // count-min-sketch crate
    {
        g.bench_function("count_min_sketch::generation", |b| {
            let mut start_index = 0;
            b.iter(|| {
                let mut cms: CountMinSketch8<&[u8]> =
                    CountMinSketch8::new(1000, 5.0, 0.01).expect("Failed to create CMS");
                for _ in 0..batch_size {
                    let doc = documents[start_index % documents.len()];
                    for word in doc.split_whitespace() {
                        cms.increment(word.as_bytes());
                    }
                    start_index = (start_index + 1) % documents.len();
                }
                cms
            })
        });
    }
}

fn main() {
    println!("Text Fingerprinting Benchmarks");
    println!("- StringZilla: Native MinHash and Count-Min-Sketch");
    println!("- probabilistic-collections: MinHash and Count-Min-Sketch implementations");
    println!("- finch: High-performance MinHash for genomic/text data");
    println!("- count-min-sketch: Dedicated Count-Min-Sketch implementation");

    let mut criterion = configure_bench();
    bench_fingerprints(&mut criterion);
    criterion.final_summary();
}
