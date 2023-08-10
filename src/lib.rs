//! A free list allocator

#![no_std]
#![deny(
    missing_docs,
    trivial_casts,
    trivial_numeric_casts,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]
#![allow(unstable_name_collisions)]

/// The main allocator
mod block;

pub use block::{AllocatorError, BlockAllocator};
