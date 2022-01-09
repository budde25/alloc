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
pub struct BlockAllocator {
    current: Block,
    bottom: u64,
    previous_allocated: bool,
}

impl BlockAllocator {
    /// Create a new empty allocator
    /// MUST INIT BEFORE USING
    pub const fn new() -> Self {
        Self {
            current: unsafe { Block::new(0 as *mut BlockHeader) },
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

        self.current = Block::from(heap_start);

        // subtract header size and the end header size
        let first_size = heap_size - HEADER_SIZE - HEADER_SIZE;
        self.current
            .write(BlockHeader::new(first_size, false, true));

        // get a pointer to the end of the heap. subtract the header size to stay in bounds
        self.current.next().write(BlockHeader::HEAP_END);
    }

    /// Advance to the next block
    ///
    /// Unsafety
    ///
    /// Requires the current block header to be valid
    /// Requires the next block header to be valid
    unsafe fn next(&mut self) {
        let next = self.current.next();
        if next.end() {
            self.current = Block::from(self.bottom);
            self.previous_allocated = true;
        } else {
            let allocated = self.current.allocated();
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
    unsafe fn next_free(&mut self, size: u64) -> Result<(), AllocatorError> {
        let start = self.current.ptr();
        while self.current.allocated() || self.current.size() < size {
            self.next();
            if self.current.ptr() == start {
                return Err(AllocatorError::OutOfSpace);
            }
        }
        Ok(())
    }

    /// Allocate using the next fit strategy
    /// Next fit keeps track of position to allocate into the first space that
    /// it can fit
    ///
    /// This function will panic if:
    /// * It cannot find a spot to fit the allocation
    /// * It is full
    ///
    /// # Safety
    ///
    /// Must pass a valid Layout
    pub unsafe fn allocate_next_fit(&mut self, layout: Layout) -> Result<*mut u8, AllocatorError> {
        let size = round_up_eight(layout.size() as u64); // Must be a multiple of 8

        self.next_free(size)?; // get us to a free block, that can fit our allocation

        let saved_size = self.current.size();
        // we found a spot
        self.current
            .write(BlockHeader::new(size, true, self.previous_allocated));

        // ptr we are going to return
        let data_start = self.current.data_start();

        let mut next = self.current.next();
        if !next.end() && !next.allocated() {
            let updated_size = saved_size - (size + HEADER_SIZE);
            next.write(BlockHeader::new(updated_size, false, true));
            self.current = next;
        } else if !next.end() && next.allocated() {
            next.set_previous_allocated(true);
        }

        Ok(data_start)
    }

    /// Deallocate a preform immediate coalescing
    /// Immediate coalescing will check to see if there is a free block in front or behind
    /// it and will merge them together into a larger free block
    ///
    /// This function will panic is the pointer is not byte aligned
    ///
    /// # Safety
    ///
    /// * Must pass a valid pointer to the data that was allocated.
    /// * The pointer must be byte aligned  
    /// * Must not double free
    pub unsafe fn dealloc_immediate_coalesce(
        &mut self,
        ptr: *mut u8,
    ) -> Result<(), AllocatorError> {
        // Ensure the pointer is valid to avoid undefined behavior
        if ptr as u64 % 8 != 0 {
            return Err(AllocatorError::InvalidPointer);
        }

        let mut header = Block::from_data_start(ptr); // This ptr points to the start of the data, get us to the block header

        let mut total_size = header.size(); // get the size of the new free area

        // immediate coalescing
        // front
        // make sure next header doesn't say the previous is allocated
        let mut next = header.next();
        if next.ptr() as u64 != self.bottom {
            if !next.allocated() {
                total_size += next.size() + HEADER_SIZE;
            }
            next.set_previous_allocated(false);
        }

        // back
        if !header.previous_allocated() {
            total_size += header.previous_size() + HEADER_SIZE;
            // warning our header is not invalid, no more reading from it
            header = header.previous();
        }

        // update the header and footer size
        header.write(BlockHeader::new(total_size, false, true));
        header.write_footer(BlockHeader::new(total_size, false, false));

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

        // let start_block = unsafe { Block::new(self.bottom as *mut BlockHeader) };
        // list.entries(start_block);

        list.finish()
    }
}

/// Allows putting a type behind a Mutex
#[derive(Debug)]
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
    use core::convert::From;
    use core::fmt::{self, Debug};
    use core::ops::{Deref, DerefMut};

    /// A pointer to a block header and the block header itself
    #[derive(PartialEq, Eq)]
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
        pub const fn new(ptr: *mut BlockHeader) -> Self {
            Self { ptr }
        }

        /// Create a new Block from the start of data
        /// Useful for dealloc
        ///
        /// # Unsafety
        ///
        /// Must be a valid pointer to 1 byte ahead of a Block Header
        pub unsafe fn from_data_start(ptr: *mut u8) -> Self {
            let header_ptr = (ptr as *mut BlockHeader).sub(1);
            Self::new(header_ptr)
        }

        /// Get the footer block
        ///
        /// # Unsafety
        ///
        /// Must contain valid pointer to a Block Header
        /// Must not be allocated
        #[allow(dead_code)]
        pub unsafe fn footer(&self) -> BlockHeader {
            assert!(!self.allocated());
            *self.deref().footer(self.ptr as u64)
        }

        /// Get the next block
        ///
        /// # Unsafety
        ///
        /// Requires a valid Block Header
        /// Requires the next Block Header to be valid
        pub unsafe fn next(&self) -> Self {
            Self::new(self.deref().next(self.ptr as u64))
        }

        /// Get the previous block
        ///
        /// # Safety
        ///
        /// Must not be called when previous block in not free
        ///
        /// # Unsafety
        ///
        /// Requires a previous block to be valid
        /// Requires a footer to have been set
        pub unsafe fn previous(&self) -> Self {
            // previous size will assert for us
            let previous_size = self.previous_size();
            let ptr = ((self.ptr as u64 - previous_size) as *mut BlockHeader).sub(1);
            Self::new(ptr)
        }

        /// Get the previous block size
        ///
        /// # Safety
        ///
        /// Must not be called when previous block in not free
        ///
        /// # Unsafety
        ///
        /// Requires a footer to have been set
        pub unsafe fn previous_size(&self) -> u64 {
            assert!(!self.previous_allocated());
            let previous_footer = self.ptr.sub(1).read_volatile();
            previous_footer.size()
        }

        /// Get the pointer
        ///
        /// # Unsafety
        ///
        /// Requires the block header to be valid
        pub unsafe fn ptr(&self) -> *const BlockHeader {
            self.ptr
        }

        /// Set if the current block is allocated
        ///
        /// # Unsafety
        ///
        /// Requires the block header to be valid
        #[allow(dead_code)]
        pub unsafe fn set_allocated(&mut self, allocated: bool) {
            if self.deref().allocated() != allocated {
                self.deref_mut()
                    .set(BlockHeader::CURRENT_BLOCK_ALLOCATED, allocated);
            }
        }

        /// Set if the previous block is allocated
        ///
        /// # Unsafety
        ///
        /// Requires the block header to be valid
        pub unsafe fn set_previous_allocated(&mut self, previous_allocated: bool) {
            if self.deref().previous_allocated() != previous_allocated {
                self.deref_mut()
                    .set(BlockHeader::PREVIOUS_BLOCK_ALLOCATED, previous_allocated);
            }
        }

        /// Write a block header to the current pointer position
        ///
        /// # Unsafety
        ///
        /// Requires the block header pointer to be valid
        pub unsafe fn write(&mut self, header: BlockHeader) {
            self.ptr.write_volatile(header);
        }

        /// Write a block header to the footer position
        ///
        /// Will panic if called while the block is allocated
        ///
        /// # Unsafety
        ///
        /// Requires the block header pointer to be valid
        pub unsafe fn write_footer(&mut self, footer: BlockHeader) {
            assert!(!self.allocated());
            (*self.ptr).footer(self.ptr as u64).write_volatile(footer);
        }

        /// Get a pointer to the data start
        ///
        /// # Unsafety
        ///
        /// Requires the block header to be valid
        pub unsafe fn data_start(&self) -> *mut u8 {
            self.ptr.add(1) as *mut u8
        }

        /// Get the size of the block (without he header)
        ///
        /// # Unsafety
        ///
        /// Requires the block header to be valid
        pub unsafe fn size(&self) -> u64 {
            self.deref().size()
        }

        /// Get if the current block is allocated
        ///
        /// # Unsafety
        ///
        /// Requires the block header to be valid
        pub unsafe fn allocated(&self) -> bool {
            self.deref().allocated()
        }

        /// Get if the previous block is allocated
        ///
        /// # Unsafety
        ///
        /// Requires the block header to be valid
        pub unsafe fn previous_allocated(&self) -> bool {
            self.deref().previous_allocated()
        }

        /// Get if this is the end block
        ///
        /// # Unsafety
        ///
        /// Requires the block header to be valid
        pub unsafe fn end(&self) -> bool {
            self.deref().end()
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
            unsafe {
                item.field("ptr", &Hex(self.ptr as u64));
                item.field("size", &self.size());
                item.field("allocated", &self.allocated());
                item.field("previous_allocated", &self.previous_allocated());
            }
            item.finish()
        }
    }

    impl From<*mut u8> for Block {
        fn from(ptr: *mut u8) -> Self {
            Self::new(ptr as *mut BlockHeader)
        }
    }

    impl From<u64> for Block {
        fn from(ptr: u64) -> Self {
            Self::new(ptr as *mut BlockHeader)
        }
    }

    bitflags! {
        /// This represents the header of each block of memory that we allocate, it will also be the footer
        /// to free block. The 3 least significant bits are unused for the size since it will be always be a multiple of 8,
        /// Instead we will store metadata about the block there.
        #[repr(C)]
        pub struct BlockHeader: u64 {
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

        /// Get a pointer to the next block header given its current header pointer
        fn next(&self, current_position: u64) -> *mut Self {
            unsafe { ((current_position + self.size()) as *mut BlockHeader).add(1) }
        }

        /// Get a pointer to the block footer given its current header pointer
        fn footer(&self, current_position: u64) -> *mut Self {
            // debug_assert!(self.previous_allocated());
            (current_position + self.size()) as *mut BlockHeader
        }

        /// Get the size of the block header (without taking into account the header), will always be a multiple of 8
        fn size(&self) -> u64 {
            let size = self.bits & (u64::MAX - 7); // Ex 0b1111_1111_1111_1000 for 32 bit, 2 more bytes of leading 1s for 64bit
            debug_assert!(size % 8 == 0);
            size
        }

        /// Get if this is the end of the heap
        fn end(&self) -> bool {
            self.contains(Self::HEAP_END)
        }

        /// Get if this header is allocated
        fn allocated(&self) -> bool {
            self.contains(Self::CURRENT_BLOCK_ALLOCATED)
        }

        /// Get if this is the end of the heap
        fn previous_allocated(&self) -> bool {
            self.contains(Self::PREVIOUS_BLOCK_ALLOCATED)
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
        assert_eq!(res.unwrap_err(), AllocatorError::OutOfSpace);
    }

    /// Test that we allocate the entire heap with one alloc
    #[test]
    fn alloc_entire() {
        let mut heap = new_heap();
        let heap_size = 1000 - (size_of::<BlockHeader>() * 2);
        let layout = Layout::from_size_align(heap_size, align_of::<BlockHeader>()).unwrap();
        let res = unsafe { heap.allocate_next_fit(layout) };
        assert!(res.is_ok());
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

        // dealloc 1st and 3rd
        unsafe { assert!(heap.dealloc_immediate_coalesce(one.unwrap()).is_ok()) };
        unsafe { assert!(heap.dealloc_immediate_coalesce(three.unwrap()).is_ok()) };

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

    #[test]
    fn fill_and_free_first() {
        let mut heap = new_heap();
        let layout = new_layout();

        // allocate 30
        let first = unsafe { heap.allocate_next_fit(layout.clone()).unwrap() };
        for _ in 0..29 {
            unsafe { heap.allocate_next_fit(layout.clone()).unwrap() };
        }

        // free our first one
        unsafe { assert!(heap.dealloc_immediate_coalesce(first).is_ok()) };

        // realocate, I would assume this take the place of the first
        unsafe { heap.allocate_next_fit(layout.clone()).unwrap() };
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
