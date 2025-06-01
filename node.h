#pragma once

#include <boost/pool/pool_alloc.hpp>
#include <boost/intrusive/list.hpp>

#include <iterator>
#include <memory>

template< typename ValueT >
class Node;

template< typename ValueT >
using NodeAllocator = boost::fast_pool_allocator< 
    Node< ValueT >,
    boost::default_user_allocator_new_delete,
    boost::details::pool::null_mutex >;

template< typename ValueT >
class Node : public boost::intrusive::list_base_hook<>
{
public:
    Node( ValueT value, NodeAllocator< ValueT >& allocator ) 
    : value( std::move( value )), allocator( allocator ) {}

    Node( Node const& ) = delete;
    Node& operator=( Node const& ) = delete;

    ~Node() 
    {
        // destruct all children and deallocate their memory
        children.clear_and_dispose(
            [this]( Node<ValueT>* child ) 
            {
                child->~Node(); // Explicitly call destructor
                this->allocator.deallocate(child, 1); // Deallocate memory
            }
        );    
    }

    NodeAllocator< ValueT >& get_allocator() { return allocator; }
    
    ValueT& get_value() { return value; }
    ValueT const& get_value() const { return value; }

    boost::intrusive::list< Node >& get_children() { return children; }
    boost::intrusive::list< Node > const& get_children() const { return children; }
private:
    ValueT value;
    NodeAllocator< ValueT >& allocator;
    boost::intrusive::list< Node > children;
};

template< typename ValueT >
using NodePtr = std::unique_ptr< 
    Node< ValueT >, 
    std::function< void (Node< ValueT >*) > >; // deallocator

template< typename ValueT >                               
void deallocate( NodeAllocator< ValueT >& allocator, Node< ValueT >* ptr )
{
    if (ptr)
    {
        ptr->~Node< ValueT >(); // Call the destructor
        allocator.deallocate( ptr ); // Deallocate memory
    }
}

template< typename ValueT >
size_t node_count( Node< ValueT > const& node )
{
    size_t count = 1;
    for (Node< ValueT > const& child : node.get_children())
        count += node_count( child );
    return count;
}
