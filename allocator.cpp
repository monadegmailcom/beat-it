#include "allocator.h"
#include <iostream>
#include <stdexcept>
#include <source_location>

using namespace std; // NOSONAR

ArenaAllocator::ArenaAllocator( size_t block_size ) 
{
    if (!block_size)
        throw invalid_argument( "block size must be positive");

    auto new_block = std::make_unique<Block>(block_size);
    current_block_ptr.store(new_block.get());
    blocks.push_back(std::move(new_block));
}

void* ArenaAllocator::allocate( size_t size, size_t alignment ) // NOSONAR
{
    if (size + alignment > blocks.front()->size())
        throw invalid_argument( "allocation does not fit in block" );

    // while loop for retry on concurrent block adding.
    while (true)
    {
        Block* current_block = current_block_ptr.load(std::memory_order_acquire);

        size_t aligned_offset;
        size_t new_offset;
        size_t offset;
        do
        {
            offset = current_offset.load( std::memory_order_relaxed );
            aligned_offset = (offset + alignment - 1) & ~(alignment - 1);
            new_offset = aligned_offset + size;
        } while (!current_offset.compare_exchange_weak(
            offset, new_offset, std::memory_order_acq_rel, std::memory_order_relaxed ));

        if (new_offset <= current_block->size())
            // bump allocation successful.
            return current_block->data() + aligned_offset;
        else // add new block.
        {
            std::lock_guard _( block_mutex );
            // double check concurrent block adding.
            if (current_block_ptr.load() == current_block)
            {
                auto new_block = std::make_unique<Block>(current_block->size());
                Block* new_block_ptr = new_block.get();
                blocks.push_back(std::move(new_block));
                current_block_ptr.store(new_block_ptr, std::memory_order_release);
                current_offset.store( 0, std::memory_order_relaxed );
            }
        }
        // retry allocation from new block.
    }
}

void ArenaAllocator::reset()
{
    current_block_ptr.store( blocks.front().get(), std::memory_order_relaxed );
    current_offset.store( 0, std::memory_order_relaxed );
}

GenerationalArenaAllocator::GenerationalArenaAllocator( size_t block_size ) :
    fst_arena_allocator( block_size ),
    snd_arena_allocator( block_size ) {}

void GenerationalArenaAllocator::reset()
{
    std::swap( current_allocator, previous_allocator ); 
    current_allocator->reset();
}
