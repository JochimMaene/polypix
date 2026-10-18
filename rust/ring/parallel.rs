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

/// The pool behind automatic parallelism. Rayon's own global pool is built
/// once per process: if that build fails, for example while the process is
/// out of threads, every later call panics. This pool is built on first use
/// instead, and a failed build is retried on the next call while that call
/// runs serially.
fn automatic_pool() -> Option<Arc<rayon::ThreadPool>> {
    static POOL: Mutex<Option<Arc<rayon::ThreadPool>>> = Mutex::new(None);
    let mut pool = POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if pool.is_none() {
        *pool = rayon::ThreadPoolBuilder::new().build().ok().map(Arc::new);
    }
    pool.clone()
}

pub(super) fn run_with_parallelism<T: Send>(
    item_count: usize,
    parallel_worthwhile: bool,
    threads: Option<usize>,
    operation: impl FnOnce(bool) -> T + Send,
) -> Result<T, String> {
    if item_count <= 1 || threads == Some(1) || !parallel_worthwhile || crate::forked() {
        return Ok(operation(false));
    }
    let automatic = automatic_pool();
    let (worker_count, use_automatic_pool) = match threads {
        Some(requested) => {
            let available = std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1);
            let workers = requested.min(available);
            let use_automatic = requested >= available
                && automatic
                    .as_ref()
                    .is_some_and(|pool| pool.current_num_threads() <= workers);
            (workers, use_automatic)
        }
        None => match &automatic {
            Some(pool) => (pool.current_num_threads().min(item_count), true),
            None => return Ok(operation(false)),
        },
    };
    if worker_count <= 1 {
        return Ok(operation(false));
    }
    match automatic {
        Some(pool) if use_automatic_pool => Ok(pool.install(|| operation(true))),
        _ => Ok(explicit_pool(worker_count)?.install(|| operation(true))),
    }
}
