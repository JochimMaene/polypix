//! Worker pools shared by every parallel entry point.

use std::sync::{Arc, Mutex, OnceLock};

pub(super) type CachedPool = Mutex<Option<(usize, Arc<rayon::ThreadPool>)>>;

pub(super) fn explicit_pool(worker_count: usize) -> Result<Arc<rayon::ThreadPool>, String> {
    // Explicit thread counts normally stay stable across repeated calls. Keep
    // one pool to cover that primary workload without growing an unbounded
    // cache for unusual alternating requests.
    static POOL: OnceLock<CachedPool> = OnceLock::new();
    let cache = POOL.get_or_init(|| Mutex::new(None));
    let mut cached = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((cached_count, pool)) = cached.as_ref() {
        if *cached_count == worker_count {
            return Ok(Arc::clone(pool));
        }
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
            .map_err(|error| format!("Could not create the requested thread pool: {error}"))?,
    );
    *cached = Some((worker_count, Arc::clone(&pool)));
    Ok(pool)
}

pub(super) fn run_with_parallelism<T: Send>(
    item_count: usize,
    parallel_worthwhile: bool,
    threads: Option<usize>,
    operation: impl FnOnce(bool) -> T + Send,
) -> Result<T, String> {
    if item_count <= 1 || threads == Some(1) || !parallel_worthwhile {
        return Ok(operation(false));
    }
    let (worker_count, use_global_pool) = match threads {
        Some(requested) => {
            let available = std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1);
            let workers = requested.min(available);
            let use_global = requested >= available && rayon::current_num_threads() <= workers;
            (workers, use_global)
        }
        None => (rayon::current_num_threads().min(item_count), true),
    };
    if worker_count <= 1 {
        return Ok(operation(false));
    }
    if use_global_pool {
        Ok(operation(true))
    } else {
        Ok(explicit_pool(worker_count)?.install(|| operation(true)))
    }
}
