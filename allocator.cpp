#include "allocator.h"
#include <iostream>
#include <stdexcept>
#include <source_location>

using namespace std; // NOSONAR

ArenaAllocator::ArenaAllocator( size_t block_size ) 
{
    if (!block_size)
        throw source_location::current();

    blocks.emplace_back(std::make_unique<std::vector<char>>(block_size));
}

void* ArenaAllocator::allocate( size_t size, size_t alignment ) // NOSONAR
{
    if (size + alignment > blocks.front()->size())
        throw invalid_argument( "allocation does not fit in block" );

    // while loop for retry on concurrent block adding.
    while (true)
    {
        const size_t block_idx = current_block_idx.load( 
            std::memory_order_acquire );
        Block& current_block = *blocks[block_idx];

        size_t offset = current_offset.load( std::memory_order_relaxed );
        size_t aligned_offset;
        size_t new_offset;
        do
        {
            aligned_offset = (offset + alignment - 1) & ~(alignment - 1);
            new_offset = aligned_offset + size;
        } while (!current_offset.compare_exchange_weak(
            offset, new_offset, std::memory_order_relaxed ));

        if (new_offset <= current_block.size())
            // bump allocation successful.
            return current_block.data() + aligned_offset;
        else // add new block.
        {
            std::lock_guard _( block_mutex );
            // double check concurrent block adding.
            if (current_block_idx.load() == block_idx)
            {
                blocks.push_back( make_unique< vector< char >>( 
                    current_block.size()));
                current_block_idx.fetch_add( 1, std::memory_order_release ); 
                current_offset.store( 0, std::memory_order_relaxed );

                cout << "arena allocator runs out of space, adding block, new " 
                    << "number of blocks: " << current_block_idx.load() + 1 
                    << endl;
            }
        }
        // retry allocation from new block.
    }
}

void ArenaAllocator::reset()
{
    current_block_idx.store( 0, std::memory_order_relaxed );
    current_offset.store( 0, std::memory_order_relaxed );
}
