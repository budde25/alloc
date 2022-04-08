use core::alloc::Layout;
use core::cmp::Eq;
use core::fmt::{self, Debug, Display};
use header::{Block, BlockHeader};
use spin::Mutex;

// Static variable to panic if we try to init the allocator twice
#[doc(hidden)]
static IS_INIT: Mutex<bool> = Mutex::new(false);

/// The size of the block header
const HEADER_SIZE: u64 = 8;

// Required since it lives under a spinlock
unsafe impl Send for BlockAllocator {}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum AllocatorError {
    OutOfSpace,
    InvalidPointer,
}

impl Display for AllocatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            AllocatorError::InvalidPointer => write!(f, "Pointer must be a multiple of 8"),
            AllocatorError::OutOfSpace => write!(f, "No room left in heap"),
        }
    }
}

/// A segmentation based allocator
///
/// https://pages.cs.wisc.edu/~remzi/OSTEP/vm-freespace.pdf
#[derive(Clone)]
pub struct BlockAllocator {
    current: Block,
    start: u64,
    previous_allocated: bool,
}

impl BlockAllocator {
    /// Create a new empty allocator
    /// MUST INIT BEFORE USING
    pub const fn new() -> Self {
        Self {
            current: unsafe { Block::from_header(0 as *mut BlockHeader) },
            start: 0,
            previous_allocated: true,
        }
    }

    /// Initialize the allocator by definition a start and and end.
    /// This function must ONLY BE CALLED once
    ///
    /// # Safety
    ///
    /// Must provide valid and aligned pointers the the start and the end of the heap
    /// This is not checked so it is on you to get it right.
    pub unsafe fn init(&mut self, heap_start: u64, heap_size: u64) {
        if heap_start % 8 != 0 || heap_size % 8 != 0 {
            panic!("must be aligned")
        }

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
        self.start = heap_start;

        // subtract header size and the end header size
        let first_size = heap_size - HEADER_SIZE - HEADER_SIZE;

        self.current = Block::init(heap_start as *mut BlockHeader, first_size, false, true);
    }

    /// Advance to the next block
    ///
    /// Unsafety
    ///
    /// Requires the current block header to be valid
    /// Requires the next block header to be valid
    fn next_block(&mut self) {
        let next = self.current.next();
        if next.is_end() {
            self.current = unsafe { Block::from_header(self.start as *mut BlockHeader) };
            self.previous_allocated = true;
        } else {
            let allocated = self.current.is_allocated();
            self.current = next;
            self.previous_allocated = allocated;
        }
    }

    /// Get the next free block, that can fit an allocation of size (not including the block header)
    ///
    /// Unsafety
    ///
    /// Requires the current block header to be valid
    /// Requires the next arbitrary number of block header to be valid
    fn next_free(&mut self, size: u64) -> Result<(), AllocatorError> {
        let start = self.current;
        while self.current.is_allocated() || self.current.data_size() < size {
            self.next_block();
            if self.current == start {
                return Err(AllocatorError::OutOfSpace);
            }
        }
        Ok(())
    }

    /// Allocate using the next fit strategy
    /// Next fit keeps track of position to allocate into the first space that
    /// it can fit
    pub fn allocate_next_fit(&mut self, layout: Layout) -> Result<*mut u8, AllocatorError> {
        let size = round_up_eight(layout.size() as u64); // Must be a multiple of 8

        self.next_free(size)?; // get us to a free block, that can fit our allocation

        let saved_size = self.current.data_size();
        let perfect_fit = self.current.data_size() == size;

        // we found a spot
        self.current.set(BlockHeader::CURR_ALLOC, true);
        self.current
            .set(BlockHeader::PREV_ALLOC, self.previous_allocated);

        if !perfect_fit {
            let mut next = self.current.next();
            if !next.is_end() {
                next.set(BlockHeader::PREV_ALLOC, false);
            }
        }

        self.current.set_size(size);

        // ptr we are going to return
        let data_start = self.current.data_start();

        if !perfect_fit {
            let mut next = self.current.next();
            let updated_size = saved_size - (size + HEADER_SIZE);
            next.set_size(updated_size);
            next.set(BlockHeader::CURR_ALLOC, false);
            next.set(BlockHeader::PREV_ALLOC, true);
        }

        Ok(data_start)
    }

    /// Deallocate a preform immediate coalescing
    /// Immediate coalescing will check to see if there is a free block in front or behind
    /// it and will merge them together into a larger free block
    ///
    /// This function will panic is the pointer is not byte aligned
    pub fn dealloc_immediate_coalesce(&mut self, ptr: *mut u8) -> Result<(), AllocatorError> {
        // Ensure the pointer is valid to avoid undefined behavior
        if ptr as u64 % 8 != 0 {
            return Err(AllocatorError::InvalidPointer);
        }
        let start_block = unsafe { Block::from_header(self.start as *mut BlockHeader) };

        let mut header = unsafe { Block::from_data(ptr) }; // This ptr points to the start of the data, get us to the block header
        let mut total_size = header.data_size(); // get the size of the new free area

        // immediate coalescing
        // front
        // make sure next header doesn't say the previous is allocated
        let mut next = header.next();
        if next != start_block {
            if !next.is_allocated() {
                total_size += next.total_size();
                // make sure the next one has a prev allocation
                let next_next = next.next();
                if next_next != start_block && next_next.is_allocated() {
                    next.set(BlockHeader::PREV_ALLOC, false);
                }
            }
            next.set(BlockHeader::PREV_ALLOC, false);
        }

        // back
        if !header.is_previous_allocated() {
            total_size += header.previous_total_size();
            // warning our header is not invalid, no more reading from it
            header = header.previous();
        }

        // update the header
        header.set_size(total_size);
        header.set(BlockHeader::CURR_ALLOC, false);
        header.set(BlockHeader::PREV_ALLOC, true);
        // and the footer
        header.footer().set_size(total_size);

        // make sure that we don't save a freed header due to backwards coalescing
        self.current = header;

        Ok(())
    }
}

impl Default for BlockAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for BlockAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for b in self.clone().into_iter() {
            list.entry(&b);
        }
        list.finish()
    }
}

impl IntoIterator for BlockAllocator {
    type Item = Block;

    type IntoIter = BlockAllocatorIterator;

    fn into_iter(self) -> Self::IntoIter {
        BlockAllocatorIterator {
            current_item: unsafe { Block::from_header(self.start as *mut BlockHeader) },
        }
    }
}

#[derive(Debug)]
pub struct BlockAllocatorIterator {
    current_item: Block,
}

impl Iterator for BlockAllocatorIterator {
    type Item = Block;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_item.is_end() {
            None
        } else {
            let curr = self.current_item;
            self.current_item = self.current_item.next();
            Some(curr)
        }
    }
}

/// Allows putting a type behind a Mutex
pub struct Locked {
    inner: Mutex<BlockAllocator>,
}

impl Locked {
    /// Create a new Mutex locked type
    pub const fn new() -> Self {
        Self {
            inner: Mutex::new(BlockAllocator::new()),
        }
    }

    pub unsafe fn init(&mut self, heap_start: u64, heap_size: u64) {
        self.inner.lock().init(heap_start, heap_size)
    }

    pub fn lock(&self) -> spin::MutexGuard<BlockAllocator> {
        self.inner.lock()
    }
}

/// panics if it would rounds above u64::max or round_to is 0
fn round_to(value: u64, round_to: u64) -> u64 {
    (value + (round_to - 1)) & !(round_to - 1)
}

/// Convenience method that rounds up to the nearest multiple of 8
fn round_up_eight(value: u64) -> u64 {
    round_to(value, 8)
}

mod header {
    use bitflags::bitflags;
    use core::fmt::{self, Debug};
    use core::ops::{Deref, DerefMut};

    /// A pointer to a block header and the block header itself
    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Block {
        ptr: *mut BlockHeader,
    }

    impl Deref for Block {
        type Target = BlockHeader;
        fn deref(&self) -> &Self::Target {
            unsafe { &*self.ptr }
        }
    }

    impl DerefMut for Block {
        fn deref_mut(self: &mut Block) -> &mut Self::Target {
            unsafe { &mut *self.ptr }
        }
    }

    impl Block {
        /// Create a new Block from a BlockHeader pointer
        ///
        /// # Unsafety
        ///
        /// Must be a valid pointer to a Block Header
        pub const unsafe fn from_header(ptr: *mut BlockHeader) -> Self {
            Self { ptr }
        }

        /// Create a new Block from the start of data
        /// Useful for dealloc
        ///
        /// # Unsafety
        ///
        /// Must be a valid pointer to the first byte after the Block header
        pub unsafe fn from_data(ptr: *mut u8) -> Self {
            let header_ptr = (ptr as *mut BlockHeader).sub(1);
            Self::from_header(header_ptr)
        }

        pub unsafe fn init(
            ptr: *mut BlockHeader,
            data_size: u64,
            current: bool,
            previous: bool,
        ) -> Self {
            let block_header = BlockHeader::new(data_size, current, previous);
            let mut block = Self { ptr };
            block.write(block_header);
            block.next().write(BlockHeader::HEAP_END);
            block
        }

        /// private helper method
        fn as_mut_u8(&self) -> *mut u8 {
            self.ptr as *mut u8
        }

        /// Get the footer block
        /// must not be called if allocated
        pub fn footer(&self) -> Self {
            assert!(!self.is_allocated());
            let ptr =
                unsafe { self.as_mut_u8().add(self.data_size() as usize) as *mut BlockHeader };
            Block { ptr }
        }

        /// Get the next block
        pub fn next(&self) -> Self {
            assert!(!self.is_end());
            unsafe {
                Self::from_header(
                    (self.as_mut_u8().add(self.total_size() as usize)) as *mut BlockHeader,
                )
            }
        }

        /// Get the previous block
        pub fn previous(&self) -> Self {
            // previous size will assert for us
            let previous_total_size = self.previous_total_size();
            let ptr = unsafe {
                (self.ptr as *mut u8).sub(previous_total_size as usize) as *mut BlockHeader
            };
            let prev = unsafe { Self::from_header(ptr) };
            prev
        }

        /// Get the previous block data size
        pub fn previous_data_size(&self) -> u64 {
            assert!(!self.is_previous_allocated());
            let previous_footer = unsafe { self.ptr.sub(1).read_volatile() };
            previous_footer.data_size()
        }

        /// Get the previous block data size
        pub fn previous_total_size(&self) -> u64 {
            assert!(!self.is_previous_allocated());
            let previous_footer = unsafe { self.ptr.sub(1).read_volatile() };
            previous_footer.total_size()
        }

        // Write a block header to the current pointer position
        fn write(&mut self, header: BlockHeader) {
            unsafe { self.ptr.write_volatile(header) };
        }

        /// Get a pointer to the data start
        pub fn data_start(&self) -> *mut u8 {
            unsafe { self.ptr.add(1) as *mut u8 }
        }
    }

    impl Debug for Block {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            struct Hex(u64);
            impl Debug for Hex {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "{:#X}", self.0)
                }
            }

            let mut item = f.debug_struct("Block");

            item.field("ptr", &Hex(self.ptr as u64));
            item.field("data_size", &self.data_size());
            item.field("allocated", &self.is_allocated());
            item.field("previous_allocated", &self.is_previous_allocated());

            item.finish()
        }
    }

    bitflags! {
        /// This represents the header of each block of memory that we allocate, it will also be the footer
        /// to free block. The 3 least significant bits are unused for the size since it will be always be a multiple of 8,
        /// Instead we will store metadata about the block there.
        #[repr(C)]
        pub struct BlockHeader: u64 {
            /// The current block is allocated
            const CURR_ALLOC = 0b1;
            /// The previous block is allocated
            const PREV_ALLOC = 0b10;
            /// Represents the end of available memory
            const HEAP_END = 0b100;
        }
    }

    impl BlockHeader {
        /// Create a new Block header
        pub fn new(size: u64, current: bool, previous: bool) -> Self {
            // SAFETY the other bits represent the size to they are a valid representation
            let mut size_status = unsafe { Self::from_bits_unchecked(size) };
            size_status.set(Self::CURR_ALLOC, current); // current bit
            size_status.set(Self::PREV_ALLOC, previous); // previous bit
            size_status
        }

        pub fn set_size(&mut self, size: u64) {
            assert!(size % 8 == 0);
            self.bits = (size & (u64::MAX - 7)) | (self.bits & 0b111)
        }

        /// Get the size of the data block (without taking into account the header), will always be a multiple of 8
        pub fn data_size(&self) -> u64 {
            let size = self.bits & (u64::MAX - 7); // Ex 0b1111_1111_1111_1000 for 32 bit, 2 more bytes of leading 1s for 64bit
            debug_assert!(size % 8 == 0);
            size
        }

        /// Get the size of header plus the data block
        pub fn total_size(&self) -> u64 {
            use core::mem::size_of;
            self.data_size() + size_of::<BlockHeader>() as u64
        }

        /// Get if this is the end of the heap
        pub fn is_end(&self) -> bool {
            self.contains(Self::HEAP_END)
        }

        /// Get if this header is allocated
        pub fn is_allocated(&self) -> bool {
            self.contains(Self::CURR_ALLOC)
        }

        /// Get if this is the end of the heap
        pub fn is_previous_allocated(&self) -> bool {
            self.contains(Self::PREV_ALLOC)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use core::alloc::Layout;
    use core::mem::{align_of, size_of};
    extern crate std;
    use std::dbg;
    use std::vec;

    /// Create a new heap, aka a box
    fn new_heap() -> BlockAllocator {
        const HEAP_SIZE: usize = 1000;
        let heap_space = Box::into_raw(Box::new([0u64; HEAP_SIZE / 8]));

        let mut heap = BlockAllocator::new();
        unsafe { heap.init(heap_space as u64, HEAP_SIZE as u64) };
        assert_eq!(heap.start, heap_space as u64);
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
        let res = heap.allocate_next_fit(layout);
        assert!(res.is_ok());
        assert_eq!(
            res.unwrap() as u64,
            heap.start + size_of::<BlockHeader>() as u64
        );
    }

    /// Test that we correctly allocate one item, incorrectly aligned
    #[test]
    fn alloc_bad_align() {
        let mut heap = new_heap();
        let layout = Layout::from_size_align(size_of::<u64>(), 1).unwrap();
        let res = heap.allocate_next_fit(layout);
        assert_eq!(
            res.unwrap() as u64,
            heap.start + size_of::<BlockHeader>() as u64
        );
    }

    /// Test that we correctly allocate two items
    #[test]
    fn alloc_two() {
        let mut heap = new_heap();
        let layout = new_layout();

        heap.allocate_next_fit(layout.clone()).unwrap();

        let ptr = heap.allocate_next_fit(layout.clone()).unwrap();
        assert_eq!(
            ptr as u64,
            heap.start + (size_of::<BlockHeader>() * 3) as u64
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
            let res = heap.allocate_next_fit(layout.clone());
            assert!(res.is_ok());
        }

        let res = heap.allocate_next_fit(layout.clone());
        assert_eq!(res.unwrap_err(), AllocatorError::OutOfSpace);
    }

    /// Test that we allocate the entire heap with one alloc
    #[test]
    fn alloc_entire() {
        let mut heap = new_heap();
        let heap_size = 1000 - (size_of::<BlockHeader>() * 2);
        let layout = Layout::from_size_align(heap_size, align_of::<BlockHeader>()).unwrap();
        let res = heap.allocate_next_fit(layout);
        assert_eq!(
            res.unwrap() as u64,
            heap.start + size_of::<BlockHeader>() as u64
        );
    }

    /// Test that we correctly deallocate one item
    #[test]
    fn dealloc_one() {
        let mut heap = new_heap();
        let layout = new_layout();
        let res = heap.allocate_next_fit(layout);
        let res = heap.dealloc_immediate_coalesce(res.unwrap());
        assert!(res.is_ok());
    }

    /// Test that we correctly deallocate one item with two allocations then reuse the space
    #[test]
    fn dealloc_first() {
        let mut heap = new_heap();
        let layout = new_layout();
        let res = heap.allocate_next_fit(layout.clone());
        heap.allocate_next_fit(layout.clone()).unwrap();
        let res = heap.dealloc_immediate_coalesce(res.unwrap());
        assert!(res.is_ok());
    }

    /// Test that deallocating a bad ptr throws an error
    #[test]
    fn dealloc_bad_ptr() {
        let mut heap = new_heap();
        let layout = new_layout();
        let res = heap.allocate_next_fit(layout);
        assert!(res.is_ok());
        let res = heap.dealloc_immediate_coalesce(((res.unwrap() as u64) + 1) as *mut u8);
        assert_eq!(res.unwrap_err(), AllocatorError::InvalidPointer);
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
            let res = heap.allocate_next_fit(layout.clone());
            assert!(res.is_ok());
            ptrs.push(res.unwrap());
        }
        // dealloc
        for ptr in ptrs {
            assert!(heap.dealloc_immediate_coalesce(ptr).is_ok());
        }
        // try to allocate the entire heap, should work since there should be no fragmentation
        let layout = Layout::from_size_align(heap_size, align_of::<BlockHeader>()).unwrap();
        let res = heap.allocate_next_fit(layout);
        assert!(res.is_ok());
        assert_eq!(
            res.unwrap() as u64,
            heap.start + size_of::<BlockHeader>() as u64
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
            let res = heap.allocate_next_fit(layout.clone());
            assert!(res.is_ok());
            ptrs.push(res.unwrap());
        }
        // dealloc backwards
        ptrs.reverse();
        for ptr in ptrs {
            assert!(heap.dealloc_immediate_coalesce(ptr).is_ok());
        }
        // try to allocate the entire heap, should work since there should be no fragmentation
        let layout = Layout::from_size_align(heap_size, align_of::<BlockHeader>()).unwrap();
        let res = heap.allocate_next_fit(layout);
        assert_eq!(
            res.unwrap() as u64,
            heap.start + size_of::<BlockHeader>() as u64
        );
    }

    /// Test that if we allocate every spot then deallocate them all backwards we have no fragmentation
    #[test]
    fn coalescing_both() {
        let mut heap = new_heap();
        let layout = new_layout();
        let heap_size = 1000 - (size_of::<BlockHeader>() * 2);

        // allocate 3
        let one = heap.allocate_next_fit(layout.clone());
        assert!(one.is_ok());

        let two = heap.allocate_next_fit(layout.clone());
        assert!(two.is_ok());

        let three = heap.allocate_next_fit(layout.clone());
        assert!(three.is_ok());

        // dealloc 1st and 3rd
        assert!(heap.dealloc_immediate_coalesce(one.unwrap()).is_ok());
        assert!(heap.dealloc_immediate_coalesce(three.unwrap()).is_ok());

        // dealloc 2nd to force it to coalesce in both directions
        assert!(heap.dealloc_immediate_coalesce(two.unwrap()).is_ok());

        // try to allocate the entire heap, should work since there should be no fragmentation
        let layout = Layout::from_size_align(heap_size, align_of::<BlockHeader>()).unwrap();
        let res = heap.allocate_next_fit(layout);
        assert_eq!(
            res.unwrap() as u64,
            heap.start + size_of::<BlockHeader>() as u64
        );
    }

    /// Test that we can fill up a lot of data, dealloc the first one and still fill data
    #[test]
    fn fill_and_free_first() {
        let mut heap = new_heap();
        let layout = new_layout();

        // allocate 30
        let first = heap.allocate_next_fit(layout.clone()).unwrap();
        for _ in 0..29 {
            heap.allocate_next_fit(layout.clone()).unwrap();
        }

        // free our first one
        assert!(heap.dealloc_immediate_coalesce(first).is_ok());

        // relocate, I would assume this take the place of the first
        heap.allocate_next_fit(layout.clone()).unwrap();
    }

    /// Test that we can reuse the heap by deallocating and reallocating
    #[test]
    fn fill_and_free_all() {
        let mut heap = new_heap();
        let layout = new_layout();

        // allocate 30
        for _ in 0..10 {
            let ptrs: Vec<_> = (0..30)
                .map(|_| heap.allocate_next_fit(layout.clone()).unwrap())
                .collect();
            dbg!(&ptrs);

            // free them all
            for ptr in ptrs {
                assert!(heap.dealloc_immediate_coalesce(ptr).is_ok());
            }
        }
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

    /// Test that the set and get size works correctly
    #[test]
    fn set_get_size() {
        let mut block = BlockHeader::new(16, true, false);
        assert_eq!(block.data_size(), 16);
        assert_eq!(block.is_allocated(), true);
        assert_eq!(block.is_previous_allocated(), false);
        block.set_size(24);
        assert_eq!(block.data_size(), 24);
        assert_eq!(block.is_allocated(), true);
        assert_eq!(block.is_previous_allocated(), false);
        block.set_size(8);
        assert_eq!(block.data_size(), 8);
        assert_eq!(block.total_size(), 16);
        assert_eq!(block.is_allocated(), true);
        assert_eq!(block.is_previous_allocated(), false);
        block.set_size(16);
        assert_eq!(block.data_size(), 16);
        assert_eq!(block.total_size(), 24);
        assert_eq!(block.is_allocated(), true);
        assert_eq!(block.is_previous_allocated(), false);
    }
}
