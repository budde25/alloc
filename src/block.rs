use alloc::alloc::GlobalAlloc;
use bitflags::bitflags;
use core::ptr;
use core::{alloc::Layout, debug_assert, fmt};
use spin::Mutex;

// static variable to panic if we try to init the allocator twice
#[doc(hidden)]
static IS_INIT: Mutex<bool> = Mutex::new(false);

/// The size of the block header
const HEADER_SIZE: u64 = 8;

/// An allocator type, the main allocator behind a Mutex
pub type Allocator = Locked<BlockAllocator>;

unsafe impl Send for BlockAllocator {}

#[derive(Debug, PartialEq)]
enum AllocateError {
    OutOfSpace,
    InvalidPointer,
}

/// A segmentation based allocator based off https://pages.cs.wisc.edu/~remzi/OSTEP/vm-freespace.pdf
#[derive(Copy, Clone)]
pub struct BlockAllocator {
    current: *mut BlockHeader,
    bottom: u64,
    previous_allocated: bool,
}

impl BlockAllocator {
    /// Create a new empty allocator, must init before using
    pub const fn new() -> Self {
        Self {
            current: ptr::null_mut(),
            bottom: 0,
            previous_allocated: true,
        }
    }

    /// Initialize the allocator by definition a start and and end.
    /// This function must ONLY BE CALLED once
    ///
    /// # Safety
    ///
    /// Must provide valid pointers the the start and the end of the heap
    /// This is not checked so it is on you to get it right.
    pub unsafe fn init(&mut self, heap_start: u64, heap_size: u64) {
        // Disallow double init
        {
            let mut is_init = IS_INIT.lock();
            if *is_init {
                #[cfg(not(test))]
                panic!("Allocator has already been initialized, can only initialize once")
            }
            *is_init = true;
        }
        // heap start is the bottom
        self.bottom = heap_start;

        self.current = self.bottom as *mut BlockHeader;
        // subtract header size and the end header size
        self.current.write_volatile(BlockHeader::new(
            heap_size - HEADER_SIZE - HEADER_SIZE,
            false, // No allocation to start
            true,  // previous is not in the heap so yes it is allocated as far as we are concerned
        ));

        // get a pointer to the end of the heap. subtract the header size to stay in bounds
        let ptr = ((self.bottom + heap_size) as *mut BlockHeader).sub(1);
        ptr.write_volatile(BlockHeader::HEAP_END);
    }

    unsafe fn next(&mut self) {
        let next = self.current.read_volatile().next(self.current as u64);
        if next.read_volatile().end() {
            self.current = self.bottom as *mut BlockHeader;
            self.previous_allocated = true;
        } else {
            let allocated = self.current.read_volatile().allocated();
            self.current = next;
            self.previous_allocated = allocated;
        }
    }

    /// Get the next free block, that can fit an allocation of size (not including the block header)
    unsafe fn next_free(&mut self, size: u64) -> Result<(), AllocateError> {
        let start_ptr = self.current;
        let current = self.current.read_volatile();
        while current.allocated() || current.size() < size {
            self.next();
            if self.current == start_ptr {
                return Err(AllocateError::OutOfSpace);
            }
        }
        Ok(())
    }

    unsafe fn allocate_next_fit(&mut self, layout: Layout) -> Result<*mut u8, AllocateError> {
        let size = round_up_eight(layout.size() as u64); // Must be a multiple of 8

        self.next_free(size)?; // get us to a free block, that can fit our allocation

        let saved_size = self.current.read_volatile().size();
        // we found a spot
        self.current
            .write_volatile(BlockHeader::new(size, true, self.previous_allocated));

        // ptr we are going to return
        let data_start = self.current.add(1);

        let next = ((self.current as u64 + size) as *mut BlockHeader).add(1);
        if !next.read_volatile().end() {
            next.write_volatile(BlockHeader::new(
                saved_size - (size + HEADER_SIZE),
                false,
                true,
            ));
            self.current = next;
        }

        Ok((data_start) as *mut u8)
    }

    unsafe fn dealloc_immediate_coalesce(&mut self, ptr: *mut u8) -> Result<(), AllocateError> {
        // Ensure the pointer is valid to avoid undefined behavior
        if ptr as u64 % 8 != 0 {
            return Err(AllocateError::InvalidPointer);
        }

        let mut header = (ptr as *mut BlockHeader).sub(1); // This ptr points to the start of the data, get us to the block header

        let mut total_size = header.read_volatile().size(); // get the size of the new free area

        // immediate coalescing
        // front
        // make sure next header doesn't say the previous is allocated
        let next = header.read_volatile().next(header as u64);
        if next as u64 != self.bottom {
            let mut value = next.read_volatile();
            if !next.read_volatile().allocated() {
                total_size += value.size() + HEADER_SIZE;
            }
            value.set(BlockHeader::PREVIOUS_BLOCK_ALLOCATED, false);
            next.write_volatile(value);
        }

        // back
        if !header.read_volatile().previous_allocated() {
            let footer = header.sub(1);
            let block_size = footer.read_volatile().size();

            total_size += block_size + HEADER_SIZE;
            header = (header as u64 - (block_size + HEADER_SIZE)) as *mut BlockHeader;
        }

        // update the header and footer
        header.write_volatile(BlockHeader::new(total_size, false, true));
        let footer = header.read_volatile().footer(header as u64);
        footer.write_volatile(BlockHeader::new(total_size, false, false));

        // make sure that we don't save an invalid spot thanks to the coalescing
        self.current = header;

        Ok(())
    }
}

impl Default for BlockAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for BlockAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // we want a better debug for this type
        #[derive(Debug)]
        struct Block {
            heap_ptr: Hex,
            size: u64,
            allocated: bool,
            previous_allocated: bool,
        }

        struct Hex(u64);

        impl fmt::Debug for Hex {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{:#X}", self.0)
            }
        }

        let mut list = f.debug_list();

        // Safety: TODO
        unsafe {
            let mut current_ptr = self.bottom as *mut BlockHeader;
            let mut current = current_ptr.read_volatile();
            loop {
                let entry = Block {
                    heap_ptr: Hex(current_ptr as u64),
                    size: current.size(),
                    allocated: current.allocated(),
                    previous_allocated: current.previous_allocated(),
                };

                list.entry(&entry);

                if current.end() {
                    break;
                }

                current_ptr = current.next(current_ptr as u64);
                current = current_ptr.read_volatile();
            }
        }

        list.finish()
    }
}

/// Allows putting a type behind a Mutex
#[derive(Debug)]
pub struct Locked<T> {
    inner: Mutex<T>,
}

impl<T> Locked<T> {
    /// Create a new Mutex locked type
    pub const fn new(inner: T) -> Self {
        Self {
            inner: Mutex::new(inner),
        }
    }

    /// Get the interior
    pub fn lock(&self) -> spin::MutexGuard<T> {
        self.inner.lock()
    }
}

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

bitflags! {
    /// This represents the header of each block of memory that we allocate, it will also be the footer
    /// to free block. The 3 least significant bits are unused for the size since it will be always be a multiple of 8,
    /// Instead we will store metadata about the block there.
    #[repr(C)]
    struct BlockHeader: u64 {
        /// The current block is allocated
        const CURRENT_BLOCK_ALLOCATED = 0b1;
        /// The previous block is allocated
        const PREVIOUS_BLOCK_ALLOCATED = 0b10;
        /// Represents the end of available memory
        const HEAP_END = 0b100;
    }
}

impl BlockHeader {
    /// Create a new Block header
    pub fn new(size: u64, current: bool, previous: bool) -> Self {
        // SAFETY the other bits represent the size to they are a valid representation
        let mut size_status = unsafe { Self::from_bits_unchecked(size) };
        size_status.set(Self::CURRENT_BLOCK_ALLOCATED, current); // current bit
        size_status.set(Self::PREVIOUS_BLOCK_ALLOCATED, previous); // previous bit
        size_status
    }

    pub fn next(&self, current_position: u64) -> *mut Self {
        unsafe { ((current_position + self.size()) as *mut BlockHeader).add(1) }
    }

    pub fn footer(&self, current_position: u64) -> *mut Self {
        // debug_assert!(self.previous_allocated());
        (current_position + self.size()) as *mut BlockHeader
    }

    /// Returns the size of the allocated data (without taking into account the header), will always be a multiple of 8
    pub fn size(&self) -> u64 {
        let size = self.bits & (u64::MAX - 7); // Ex 0b1111_1111_1111_1000 for 32 bit, 2 more bytes of leading 1s for 64bit
        debug_assert!(size % 8 == 0);
        size
    }

    /// Returns true if we have reached the end of the heap, false otherwise
    pub fn end(&self) -> bool {
        self.bits == 0b100
    }

    pub fn allocated(&self) -> bool {
        self.contains(Self::CURRENT_BLOCK_ALLOCATED)
    }

    pub fn previous_allocated(&self) -> bool {
        self.contains(Self::PREVIOUS_BLOCK_ALLOCATED)
    }
}

/// panics if it would rounds above u64::max or round_to is 0
fn round_to(value: u64, round_to: u64) -> u64 {
    value + (round_to - 1) & !(round_to - 1)
}

/// Convenience method that rounds up to the nearest multiple of 8
fn round_up_eight(value: u64) -> u64 {
    round_to(value, 8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use core::alloc::Layout;
    use core::mem::{align_of, size_of};
    use std::{dbg, println};
    extern crate std;
    use std::vec;

    /// Create a new heap, aka a box
    fn new_heap() -> BlockAllocator {
        const HEAP_SIZE: usize = 1000;
        let heap_space = Box::into_raw(Box::new([0u8; HEAP_SIZE]));

        let mut heap = BlockAllocator::new();
        unsafe { heap.init(heap_space as u64, HEAP_SIZE as u64) };
        assert!(heap.bottom == heap_space as u64);
        heap
    }

    /// Create a new layout with size and align of a BlockHeader (u64)
    fn new_layout() -> Layout {
        Layout::from_size_align(size_of::<BlockHeader>(), align_of::<BlockHeader>()).unwrap()
    }

    /// Test that we correctly allocate one item
    #[test]
    fn alloc_one() {
        let mut heap = new_heap();
        let layout = new_layout();
        let res = unsafe { heap.allocate_next_fit(layout) };
        assert!(res.is_ok());
        assert_eq!(
            res.unwrap() as u64,
            heap.bottom + size_of::<BlockHeader>() as u64
        );
    }

    /// Test that we correctly allocate one item, incorrectly aligned
    #[test]
    fn alloc_bad_align() {
        let mut heap = new_heap();
        let layout = Layout::from_size_align(size_of::<u64>(), 1).unwrap();
        let res = unsafe { heap.allocate_next_fit(layout) };
        assert!(res.is_ok());
        assert_eq!(
            res.unwrap() as u64,
            heap.bottom + size_of::<BlockHeader>() as u64
        );
    }

    /// Test that we correctly allocate two items
    #[test]
    fn alloc_two() {
        let mut heap = new_heap();
        let layout = new_layout();
        unsafe { heap.allocate_next_fit(layout.clone()).unwrap() };
        let res = unsafe { heap.allocate_next_fit(layout.clone()) };
        assert!(res.is_ok());
        assert_eq!(
            res.unwrap() as u64,
            heap.bottom + (size_of::<BlockHeader>() * 3) as u64
        );
    }

    /// Test that if we max out the heap the next allocation will fail
    #[test]
    fn max_out() {
        let mut heap = new_heap();
        let layout = new_layout();
        let heap_size = 1000 - size_of::<BlockHeader>();
        let num_loops = heap_size / (size_of::<BlockHeader>() * 2);
        for _ in 0..num_loops {
            let res = unsafe { heap.allocate_next_fit(layout.clone()) };
            assert!(res.is_ok());
        }

        let res = unsafe { heap.allocate_next_fit(layout.clone()) };
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), AllocateError::OutOfSpace);
    }

    /// Test that we allocate the entire heap with one alloc
    #[test]
    fn alloc_entire() {
        let mut heap = new_heap();
        let heap_size = 1000 - (size_of::<BlockHeader>() * 2);
        let layout = Layout::from_size_align(heap_size, align_of::<BlockHeader>()).unwrap();
        let res = unsafe { heap.allocate_next_fit(layout) };
        assert!(res.is_ok());
        dbg!(heap);
        assert_eq!(
            res.unwrap() as u64,
            heap.bottom + size_of::<BlockHeader>() as u64
        );
    }

    /// Test that we correctly deallocate one item
    #[test]
    fn dealloc_one() {
        let mut heap = new_heap();
        let layout = new_layout();
        let res = unsafe { heap.allocate_next_fit(layout) };
        assert!(res.is_ok());
        let res = unsafe { heap.dealloc_immediate_coalesce(res.unwrap()) };
        assert!(res.is_ok());
    }

    /// Test that we correctly deallocate one item with two allocations then reuse the space
    #[test]
    fn dealloc_first() {
        let mut heap = new_heap();
        let layout = new_layout();
        let res = unsafe { heap.allocate_next_fit(layout.clone()) };
        unsafe { heap.allocate_next_fit(layout.clone()).unwrap() };
        assert!(res.is_ok());
        let res = unsafe { heap.dealloc_immediate_coalesce(res.unwrap()) };
        assert!(res.is_ok());
        println!("{:#?}", heap);
    }

    /// Test that deallocating a bad ptr throws an error
    #[test]
    fn dealloc_bad_ptr() {
        let mut heap = new_heap();
        let layout = new_layout();
        let res = unsafe { heap.allocate_next_fit(layout) };
        assert!(res.is_ok());
        let res =
            unsafe { heap.dealloc_immediate_coalesce(((res.unwrap() as u64) + 1) as *mut u8) };
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), AllocateError::InvalidPointer);
    }

    /// Test that if we allocate every spot then deallocate them all we have no fragmentation
    #[test]
    fn coalescing_forward() {
        let mut heap = new_heap();
        let layout = new_layout();
        let heap_size = 1000 - (size_of::<BlockHeader>() * 2);
        let num_loops = heap_size / (size_of::<BlockHeader>() * 2);
        // save them
        let mut ptrs = vec![];
        for _ in 0..num_loops {
            let res = unsafe { heap.allocate_next_fit(layout.clone()) };
            assert!(res.is_ok());
            ptrs.push(res.unwrap());
        }
        // dealloc
        for ptr in ptrs {
            unsafe { assert!(heap.dealloc_immediate_coalesce(ptr).is_ok()) };
        }
        // try to allocate the entire heap, should work since there should be no fragmentation
        let layout = Layout::from_size_align(heap_size, align_of::<BlockHeader>()).unwrap();
        let res = unsafe { heap.allocate_next_fit(layout) };
        assert!(res.is_ok());
        assert_eq!(
            res.unwrap() as u64,
            heap.bottom + size_of::<BlockHeader>() as u64
        );
    }

    /// Test that if we allocate every spot then deallocate them all backwards we have no fragmentation
    #[test]
    fn coalescing_backwards() {
        let mut heap = new_heap();
        let layout = new_layout();
        let heap_size = 1000 - (size_of::<BlockHeader>() * 2);
        let num_loops = heap_size / (size_of::<BlockHeader>() * 2);
        // save them
        let mut ptrs = vec![];
        for _ in 0..num_loops {
            let res = unsafe { heap.allocate_next_fit(layout.clone()) };
            assert!(res.is_ok());
            ptrs.push(res.unwrap());
        }
        // dealloc backwards
        ptrs.reverse();
        for ptr in ptrs {
            unsafe { assert!(heap.dealloc_immediate_coalesce(ptr).is_ok()) };
        }
        // try to allocate the entire heap, should work since there should be no fragmentation
        let layout = Layout::from_size_align(heap_size, align_of::<BlockHeader>()).unwrap();
        let res = unsafe { heap.allocate_next_fit(layout) };
        assert!(res.is_ok());
        assert_eq!(
            res.unwrap() as u64,
            heap.bottom + size_of::<BlockHeader>() as u64
        );
    }

    /// Test that if we allocate every spot then deallocate them all backwards we have no fragmentation
    #[test]
    fn coalescing_both() {
        let mut heap = new_heap();
        let layout = new_layout();
        let heap_size = 1000 - (size_of::<BlockHeader>() * 2);

        // allocate 3
        let one = unsafe { heap.allocate_next_fit(layout.clone()) };
        assert!(one.is_ok());

        let two = unsafe { heap.allocate_next_fit(layout.clone()) };
        assert!(two.is_ok());

        let three = unsafe { heap.allocate_next_fit(layout.clone()) };
        assert!(three.is_ok());

        dbg!(heap);
        // dealloc 1st and 3rd
        unsafe { assert!(heap.dealloc_immediate_coalesce(one.unwrap()).is_ok()) };
        unsafe { assert!(heap.dealloc_immediate_coalesce(three.unwrap()).is_ok()) };

        dbg!(heap);
        // dealloc 2nd to force it to coalesce in both directions
        unsafe { assert!(heap.dealloc_immediate_coalesce(two.unwrap()).is_ok()) };

        // try to allocate the entire heap, should work since there should be no fragmentation
        let layout = Layout::from_size_align(heap_size, align_of::<BlockHeader>()).unwrap();
        let res = unsafe { heap.allocate_next_fit(layout) };
        assert!(res.is_ok());
        assert_eq!(
            res.unwrap() as u64,
            heap.bottom + size_of::<BlockHeader>() as u64
        );
    }

    /// Test that creating header returns the correct bitflags
    #[test]
    fn test_header() {
        let header = unsafe { BlockHeader::from_bits_unchecked(27) };
        assert!(header.allocated());
        assert!(header.previous_allocated());
        assert_eq!(header.size(), 24);
    }

    /// Test the we always round up to the nearest 8
    #[test]
    fn test_round_to() {
        assert_eq!(8, round_to(7, 8));
        assert_eq!(40, round_to(33, 8));
        assert_eq!(32, round_to(32, 8));
    }

    /// Test that a header size is correct
    #[test]
    fn header_size() {
        assert_eq!(HEADER_SIZE, size_of::<BlockHeader>() as u64)
    }
}
