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

    void* allocate( size_t size, size_t alignment );
    void deallocator( void*, size_t ) const { /*do nothing*/ } // NOSONAR
    // not thread safe.
    void reset();
private:
    std::deque< std::unique_ptr< Block >> blocks;
    std::mutex block_mutex;
    std::atomic< Block* > current_block_ptr {nullptr};
    std::atomic< size_t > current_offset {0};
};

template< typename T >
class TypedAllocator
{
public:
    explicit TypedAllocator( size_t objects_per_block ) 
    : arena_allocator( sizeof( T ) * objects_per_block ) {}

    T* allocate( size_t n = 1 )
    {
        void* memory = arena_allocator.allocate( 
            n * sizeof( T ), alignof( T ));
        return static_cast< T* >( memory );
    }

    void deallocate( T*, size_t ) noexcept { /*do nothing*/ } // NOSONAR
    void reset() { arena_allocator.reset(); }                                                            
private:
    ArenaAllocator arena_allocator;
};

