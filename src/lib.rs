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
mod block;

use alloc::alloc::{GlobalAlloc, Layout};
use block::{BlockAllocator, Locked};
use core::ptr;

/// An allocator type, the main allocator behind a Mutex
pub type Allocator = Locked<BlockAllocator>;

unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.lock()
            .allocate_next_fit(layout)
            .ok()
            .map_or(ptr::null_mut(), |ptr| ptr)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        if self.lock().dealloc_immediate_coalesce(ptr).is_err() {
            panic!("Invalid pointer");
        }
    }
}

#[cfg(test)]
mod test;
