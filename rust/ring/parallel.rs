//! Worker pools shared by every parallel entry point.
//!
//! Parallelism never changes an answer, only how long it takes, so every
//! path here falls back to running serially rather than failing a call.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Set before a pool is first touched. A child forked after this point may
/// have inherited a pool whose threads no longer exist, or a pool lock held
/// mid-build, so it must not use either.
static POOLS_TOUCHED: AtomicBool = AtomicBool::new(false);

pub(crate) fn pools_touched() -> bool {
    POOLS_TOUCHED.load(Ordering::Relaxed)
}

/// The pool behind automatic parallelism. Rayon's own global pool is built
/// once per process: if that build fails, for example while the process is
/// out of threads, every later call panics. This pool is built on first use
/// instead, and a failed build is retried on the next call.
fn automatic_pool() -> Option<Arc<rayon::ThreadPool>> {
    static POOL: Mutex<Option<Arc<rayon::ThreadPool>>> = Mutex::new(None);
    POOLS_TOUCHED.store(true, Ordering::Relaxed);
    let mut pool = POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if pool.is_none() {
        *pool = rayon::ThreadPoolBuilder::new().build().ok().map(Arc::new);
    }
    pool.clone()
}

/// A pool for an explicit thread count. Explicit counts normally stay stable
/// across repeated calls, so keep one pool for that workload without growing
/// an unbounded cache for unusual alternating requests.
fn explicit_pool(worker_count: usize) -> Option<Arc<rayon::ThreadPool>> {
    static POOL: Mutex<Option<(usize, Arc<rayon::ThreadPool>)>> = Mutex::new(None);
    POOLS_TOUCHED.store(true, Ordering::Relaxed);
    let mut cached = POOL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((count, pool)) = cached.as_ref() {
        if *count == worker_count {
            return Some(Arc::clone(pool));
        }
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
            .ok()?,
    );
    *cached = Some((worker_count, Arc::clone(&pool)));
    Some(pool)
}

pub(super) fn run_with_parallelism<T: Send>(
    item_count: usize,
    parallel_worthwhile: bool,
    threads: Option<usize>,
    operation: impl FnOnce(bool) -> T + Send,
) -> T {
    if item_count <= 1 || threads == Some(1) || !parallel_worthwhile || crate::pools_unusable() {
        return operation(false);
    }
    let pool = match threads {
        None => automatic_pool(),
        Some(requested) => {
            let available = std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1);
            let workers = requested.min(available);
            if workers <= 1 {
                return operation(false);
            }
            // Reuse the automatic pool only when it has exactly the requested
            // size; otherwise build the explicit pool, never both.
            let automatic = (workers == available)
                .then(automatic_pool)
                .flatten()
                .filter(|pool| pool.current_num_threads() == workers);
            automatic.or_else(|| explicit_pool(workers))
        }
    };
    match pool {
        Some(pool) if pool.current_num_threads().min(item_count) > 1 => {
            pool.install(|| operation(true))
        }
        _ => operation(false),
    }
}
