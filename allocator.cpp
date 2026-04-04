#include "allocator.h"
#include <cstddef>
#include <stdexcept>
#include <mutex>

using namespace std;

ArenaAllocator::ArenaAllocator( size_t block_size )
{
    if ( !block_size )
        throw invalid_argument( "block size must be positive" );

    auto new_block = make_unique< ArenaAllocator::Block >( block_size );
    current_block_ptr.store( new_block.get(), std::memory_order_release );
    blocks.push_back( std::move( new_block ) );
}

void* ArenaAllocator::allocate( size_t size, size_t alignment )
{
    while ( true )
    {
        ArenaAllocator::Block* block = current_block_ptr.load( std::memory_order_acquire );
        size_t offset = current_offset.load( std::memory_order_acquire );
        
        size_t aligned_offset = ( offset + alignment - 1 ) & ~( alignment - 1 );
        size_t space_needed = aligned_offset - offset + size;

        const size_t block_limit = block->size();

        // ~(size_t)0 is a tombstone flag signaling a new block is being allocated.
        if ( offset == ~(size_t)0 || offset + space_needed > block_limit )
        {
            // Current block is full, or another thread is currently allocating a new block.
            // Use mutex to add a new one safely.
            std::lock_guard< std::mutex > lock( block_mutex );
            
            // Double check if another thread already advanced the block while we waited
            if ( current_block_ptr.load( std::memory_order_relaxed ) != block )
                continue;
                
            // Signal to other threads that we are transitioning blocks
            current_offset.store( ~(size_t)0, std::memory_order_release );
                
            auto new_block = make_unique< ArenaAllocator::Block >( block_limit );
            ArenaAllocator::Block* new_block_ptr = new_block.get();
            blocks.push_back( std::move( new_block ) );

            // Update state: store block pointer FIRST, then offset to prevent deadly ABA races
            current_block_ptr.store( new_block_ptr, std::memory_order_release );
            current_offset.store( size, std::memory_order_release );
            
            return new_block_ptr->data();
        }

        // Fast path: Allocation fits in current block and is 100% lock-free.
        if ( current_offset.compare_exchange_weak( offset, offset + space_needed, std::memory_order_release, std::memory_order_relaxed ) )
        {
            // ABA check: Verify the block pointer hasn't changed underneath us.
            // If it changed, our CAS miraculously applied to a NEW block by coincidence.
            // We intentionally leak the claimed space in the new block and retry.
            // (Leaking is harmless here since the entire Arena is wiped on reset() anyway).
            if ( current_block_ptr.load( std::memory_order_acquire ) != block )
                continue;
                
            return block->data() + aligned_offset;
        }
    }
}

void ArenaAllocator::reset() noexcept
{
    std::lock_guard< std::mutex > lock( block_mutex );
    current_block_ptr.store( blocks.front().get(), std::memory_order_release );
    current_offset.store( 0, std::memory_order_release );
}

GenerationalArenaAllocator::GenerationalArenaAllocator( size_t block_size )
    : block_size( block_size ), fst_arena_allocator( block_size ),
      snd_arena_allocator( block_size )
{
}

void GenerationalArenaAllocator::reset()
{
    std::swap( current_allocator, previous_allocator );
    current_allocator->reset();
}

size_t GenerationalArenaAllocator::allocated_size() const noexcept
{
    const size_t b =
        fst_arena_allocator.allocated_blocks() +
        snd_arena_allocator.allocated_blocks();
    const size_t o =
        fst_arena_allocator.get_current_offset() +
        snd_arena_allocator.get_current_offset();
    return b * block_size + o;
}