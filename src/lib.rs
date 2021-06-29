//! A segmented allocator

#![no_std]
#![deny(
    missing_docs,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]

extern crate alloc;

/// The main allocator
pub mod block;

#[cfg(test)]
mod test;
