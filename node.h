#pragma once

#include "allocator.h"

#include <boost/intrusive/list.hpp>
#include <boost/intrusive/list_hook.hpp>

#include <iterator>
#include <memory>
#include <shared_mutex>

template< typename ValueT >
class Node;

template< typename ValueT >
class Node : public boost::intrusive::list_base_hook<>
{
public:
    using value_type = ValueT;
    using allocator_type = TypedAllocator< Node >;

    explicit Node( ValueT&& value ) : value( std::move( value )) {}

    Node( Node const& ) = delete;
    Node& operator=( Node const& ) = delete;

    ValueT& get_value() { return value; }
    ValueT const& get_value() const { return value; }

    boost::intrusive::list< Node >& get_children() { return children; }
    boost::intrusive::list< Node > const& get_children() const
    { return children; }
    std::shared_mutex& get_mutex() { return node_mutex; }

    std::unique_lock< std::mutex > lock() const
    { return std::unique_lock< std::mutex >( node_mutex ); }
private:
    ValueT value;
    boost::intrusive::list< Node > children;
    std::shared_mutex node_mutex;
};

template< typename ValueT >
using List = boost::intrusive::list< Node< ValueT >>;

template< typename ValueT >
class NodeAllocator
{
public:
    using value_type = ValueT;
    using node_type = Node< ValueT >;
    using allocator_type = TypedAllocator< node_type >;

    explicit NodeAllocator( size_t nodes_per_block )
    : fst_allocator( nodes_per_block ), snd_allocator( nodes_per_block ) {}

    node_type* get_root() noexcept { return root; }
    node_type const* get_root() const noexcept { return root; }
    
    node_type& allocate( value_type&& value )
    {
        return *(new (current_allocator->allocate()) 
            node_type( std::move( value )));
    }

    void rebase( node_type& node )
    {
        std::swap( current_allocator, previous_allocator ); 
        current_allocator->reset();
        root = &move( node ); 
    }
private:
    // recursively move the whole subtree.
    node_type& move( node_type& node)
    {
        node_type& new_node = allocate( std::move( node.get_value()));
        for (node_type& child : node.get_children())
            new_node.get_children().push_back( move( child ));
        return new_node;
    }

    allocator_type fst_allocator;
    allocator_type snd_allocator;
    allocator_type* current_allocator = &fst_allocator;
    allocator_type* previous_allocator = &snd_allocator;
    node_type* root = nullptr;
};

template< typename ValueT >
size_t node_count( Node< ValueT > const& node )
{
    size_t count = 1;
    for (Node< ValueT > const& child : node.get_children())
        count += node_count( child );
    return count;
}
