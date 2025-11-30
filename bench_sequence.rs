#![doc = r#"# StringWars: String Sequence Operations Benchmarks

This file benchmarks various libraries for processing string-identifiable collections.
Including sorting arrays of strings:

- StringZilla's `sz::argsort_permutation`
- The standard library's `sort_unstable`
- Rayon's parallel `par_sort_unstable`

Intersecting string collections, similar to "STRICT INNER JOIN" in SQL databases.

## Usage Example

The benchmarks use two environment variables to control the input dataset and mode:

- `STRINGWARS_DATASET`: Path to the input dataset file.
- `STRINGWARS_TOKENS`: Specifies how to interpret the input. Allowed values:
  - `lines`: Process the dataset line by line.
  - `words`: Process the dataset word by word.

To run the benchmarks with the appropriate CPU features enabled, you can use the following commands:

```sh
RUSTFLAGS="-C target-cpu=native" \
    RAYON_NUM_THREADS=1 \
    POLARS_MAX_THREADS=1 \
    STRINGWARS_DATASET=README.md \
    STRINGWARS_TOKENS=lines \
    cargo criterion --features bench_sequence bench_sequence --jobs $(nproc)
```
"#]

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, SamplingMode};
use stringtape::CharsCowsAuto;

use arrow::array::{ArrayRef, LargeStringArray};
use arrow::compute::{lexsort_to_indices, SortColumn};
use polars::prelude::*;
use rayon::prelude::*;
use stringzilla::sz;

mod utils;
use utils::{
    install_panic_hook, load_dataset, reclaim_memory, should_run, ComparisonsWallTime, ResultExt,
};

fn log_stringzilla_metadata() {
    let v = sz::version();
    eprintln!(
        "StringZilla: v{}.{}.{}, dispatch: {}, capabilities: {}",
        v.major,
        v.minor,
        v.patch,
        if sz::dynamic_dispatch() {
            "dynamic"
        } else {
            "static"
        },
        sz::capabilities().as_str()
    );
}

fn configure_bench() -> Criterion<ComparisonsWallTime> {
    Criterion::default()
        .with_measurement(ComparisonsWallTime::default())
        .sample_size(10) // Each loop processes the whole dataset.
        .warm_up_time(std::time::Duration::from_secs(5)) // Let CPU frequencies settle.
        .measurement_time(std::time::Duration::from_secs(10)) // Actual measurement time.
}

fn bench_argsort(
    group: &mut criterion::BenchmarkGroup<'_, ComparisonsWallTime>,
    unsorted: &CharsCowsAuto<'static>,
) {
    // ? We have a very long benchmark, flat sampling is what we need.
    // ? https://bheisler.github.io/criterion.rs/book/user_guide/advanced_configuration.html#sampling-mode
    group.sampling_mode(SamplingMode::Flat);
    // ? For comparison-based sorting algorithms, we can report throughput in terms of comparisons,
    // ? which is proportional to the number of elements in the array multiplied by the logarithm of
    // ? the number of elements.
    let count = unsorted.len();
    let throughput = count as f64 * (count as f64).log2();
    group.throughput(criterion::Throughput::Elements(throughput as u64));

    // Pre-allocate reusable index vector to avoid allocation on hot path
    let mut reusable_indices = Vec::with_capacity(count);

    // Pre-allocate SortOptions to avoid .default() calls on hot path
    const POLARS_SORT_OPTIONS: SortOptions = SortOptions {
        descending: false,
        nulls_last: false,
        multithreaded: true,
        maintain_order: false,
        limit: None,
    };
    let polars_sort_multiple_options = SortMultipleOptions::default();

    // Pre-allocate static string for column name
    const COLUMN_NAME: &str = "strings";

    // Benchmark: StringZilla's argsort
    if should_run("argsort/stringzilla::argsort_permutation") {
        // Collect StringTape into Vec<&str> for StringZilla (zero-copy, just references)
        let unsorted_refs: Vec<&str> = unsorted.iter().collect();

        group.bench_function("stringzilla::argsort_permutation", |b| {
            b.iter(|| {
                reusable_indices.clear();
                reusable_indices.extend(0..count);
                sz::argsort_permutation(&unsorted_refs, &mut reusable_indices)
                    .expect("StringZilla argsort failed");
                black_box(&reusable_indices);
            })
        });

        drop(unsorted_refs);
        reclaim_memory();
    }

    // Benchmark: Apache Arrow's `lexsort_to_indices`
    // https://arrow.apache.org/rust/arrow/compute/fn.lexsort.html
    // https://arrow.apache.org/rust/arrow/compute/fn.lexsort_to_indices.html
    // ! We can't use the conventional `StringArray` in most of our workloads, as it will
    // ! overflow the 32-bit tape offset capacity and panic.
    if should_run("argsort/arrow::lexsort_to_indices") {
        // Lazy initialization: only create array when this benchmark runs
        // Use from_iter_values to directly iterate from StringTape
        let array = Arc::new(LargeStringArray::from_iter_values(unsorted.iter())) as ArrayRef;

        group.bench_function("arrow::lexsort_to_indices", |b| {
            b.iter(|| {
                let column_to_sort = SortColumn {
                    values: array.clone(),
                    options: Some(arrow::compute::SortOptions {
                        descending: false,
                        nulls_first: true,
                    }),
                };
                match lexsort_to_indices(&[column_to_sort], None) {
                    Ok(indices) => black_box(indices),
                    Err(e) => panic!("Arrow lexsort failed: {:?}", e),
                }
            })
        });

        // Explicitly drop and reclaim memory (~4.7 GB)
        drop(array);
        reclaim_memory();
    }

    // Benchmark: Standard library argsort using `sort_unstable_by_key`
    if should_run("argsort/std::sort_unstable_by_key") {
        // Collect StringTape into Vec<&str> for indexing
        let unsorted_refs: Vec<&str> = unsorted.iter().collect();

        group.bench_function("std::sort_unstable_by_key", |b| {
            b.iter(|| {
                reusable_indices.clear();
                reusable_indices.extend(0..count);
                reusable_indices.sort_unstable_by_key(|&i| unsorted_refs[i]);
                black_box(&reusable_indices);
            })
        });

        drop(unsorted_refs);
        reclaim_memory();
    }

    // Benchmark: Parallel argsort using Rayon
    if should_run("argsort/rayon::par_sort_unstable_by_key") {
        // Collect StringTape into Vec<&str> for indexing
        let unsorted_refs: Vec<&str> = unsorted.iter().collect();

        group.bench_function("rayon::par_sort_unstable_by_key", |b| {
            b.iter(|| {
                reusable_indices.clear();
                reusable_indices.extend(0..count);
                reusable_indices.par_sort_unstable_by_key(|&i| unsorted_refs[i]);
                black_box(&reusable_indices);
            })
        });

        drop(unsorted_refs);
        reclaim_memory();
    }

    // Benchmark: Polars Series sort
    if should_run("argsort/polars::Series::sort") {
        // Polars can create Series from an iterator of &str
        let polars_series = Series::new(COLUMN_NAME.into(), unsorted.iter().collect::<Vec<&str>>());
        group.bench_function("polars::Series::sort", |b| {
            b.iter(|| {
                let sorted = polars_series.sort(POLARS_SORT_OPTIONS).unwrap();
                let _ = black_box(sorted);
            })
        });
        drop(polars_series);
        reclaim_memory();
    }

    // Benchmark: Polars Series argsort (returning indices)
    if should_run("argsort/polars::Series::arg_sort") {
        let polars_series = Series::new(COLUMN_NAME.into(), unsorted.iter().collect::<Vec<&str>>());
        group.bench_function("polars::Series::arg_sort", |b| {
            b.iter(|| {
                let indices = polars_series.arg_sort(POLARS_SORT_OPTIONS);
                black_box(indices);
            })
        });
        drop(polars_series);
        reclaim_memory();
    }

    // Benchmark: Polars DataFrame sort
    if should_run("argsort/polars::DataFrame::sort") {
        // Lazy initialization: only create DataFrame when needed
        // No unnecessary clone - DataFrame takes ownership directly
        let polars_dataframe = DataFrame::new(vec![Series::new(
            COLUMN_NAME.into(),
            unsorted.iter().collect::<Vec<&str>>(),
        )
        .into()])
        .unwrap();

        group.bench_function("polars::DataFrame::sort", |b| {
            b.iter(|| {
                let sorted = polars_dataframe
                    .sort([COLUMN_NAME], polars_sort_multiple_options.clone())
                    .unwrap();
                black_box(sorted);
            })
        });

        // Explicitly drop and reclaim memory (~4.7 GB)
        drop(polars_dataframe);
        reclaim_memory();
    }
}

fn main() {
    install_panic_hook();
    log_stringzilla_metadata();

    // Load the dataset defined by the environment variables
    let tokens_bytes = load_dataset().unwrap_nice();
    // Leak BytesCowsAuto to get 'static lifetime, then cast to CharsCowsAuto for UTF-8 string benchmarks
    let tokens_bytes_static: &'static _ = Box::leak(Box::new(tokens_bytes));
    let tokens = tokens_bytes_static
        .as_chars()
        .expect("Dataset must be valid UTF-8");

    let mut criterion = configure_bench();

    let mut group = criterion.benchmark_group("argsort");
    bench_argsort(&mut group, &tokens);
    group.finish();

    criterion.final_summary();
}
