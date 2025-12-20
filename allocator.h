#include <vector>
#include <deque>
#include <atomic>
#include <memory>
#include <mutex>

class ArenaAllocator
{
public:
    using Block = std::vector< char >;
    explicit ArenaAllocator( size_t block_size );

    ArenaAllocator( ArenaAllocator const& ) = delete;
    ArenaAllocator& operator=( ArenaAllocator const& ) = delete;

    // thread-safe.
    template< typename T >
    void* allocate( size_t n = 1 ) // NOSONAR
    { return allocate( n * sizeof( T ), alignof( T )); }

    // thread-safe.
    void* allocate( size_t size, size_t alignment );

    void deallocator( void*, size_t ) const { /*do nothing*/ } // NOSONAR
                                                               
    // not thread safe.
    void reset() noexcept; 
   
    // not thread safe.
    size_t allocated_blocks() const noexcept
    { return blocks.size(); }
    
    // not thread safe.
    size_t get_current_offset() const noexcept 
    { return current_offset; }
private:
    std::deque< std::unique_ptr< Block >> blocks;
    std::mutex block_mutex;
    std::atomic< Block* > current_block_ptr {nullptr};
    std::atomic< size_t > current_offset {0};
};

class GenerationalArenaAllocator
{
public:
    explicit GenerationalArenaAllocator( size_t block_size );
    GenerationalArenaAllocator( GenerationalArenaAllocator const& ) = delete;
    GenerationalArenaAllocator& operator=( GenerationalArenaAllocator const& ) 
        = delete;

    template< typename T >
    void* allocate( size_t n = 1 ) // NOSONAR
    {
        return current_allocator->allocate< T >( n );
    }

    size_t allocated_size() const noexcept;

    void reset();
    
    ArenaAllocator const& get_fst_arena_allocator() const noexcept
    { return fst_arena_allocator; }
    ArenaAllocator const& get_snd_arena_allocator() const noexcept
    { return snd_arena_allocator; }
private:
    const size_t block_size;
    ArenaAllocator fst_arena_allocator;
    ArenaAllocator snd_arena_allocator;
    ArenaAllocator* current_allocator = &fst_arena_allocator;
    ArenaAllocator* previous_allocator = &snd_arena_allocator;
};
