#pragma once

#include <boost/pool/pool_alloc.hpp>
#include <boost/intrusive/list.hpp>
#include <boost/intrusive/list_hook.hpp>

#include <iterator>
#include <memory>

template< typename ValueT >
class Node;

// using a null_mutex is not threadsafe, even if each pool_allocator is only
// used by a single thread
template< typename ValueT >
using NodeAllocator = boost::pool_allocator< Node< ValueT >>;
    //Node< ValueT >,
   // boost::default_user_allocator_new_delete,
   // boost::details::pool::null_mutex >;

template< typename ValueT >
class Node : public boost::intrusive::list_base_hook<>
{
public:
    Node( ValueT value, NodeAllocator< ValueT >& allocator ) 
    : value( std::move( value )), allocator( allocator ) {}

    Node( Node const& ) = delete;
    Node& operator=( Node const& ) = delete;

    // The destructor is responsible for cleaning up the entire subtree rooted at this node.
    ~Node()
    {
        // To safely destroy children while iterating, we first move them to a
        // temporary list. This unlinks them from `this->children`.
        boost::intrusive::list<Node<ValueT>> children_to_delete;
        children_to_delete.splice(children_to_delete.begin(), children);

        // Now that the original `children` list is empty, we can safely dispose
        // of the nodes in the temporary list. The disposer lambda will be called
        // for each child, which in turn calls its destructor, leading to safe recursion.
        children_to_delete.clear_and_dispose([this](auto child) {
            child->~Node();
            this->allocator.deallocate(child, 1);
        });
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
    std::function< void (Node< ValueT >*) > >; // Custom deleter

template< typename ValueT >
using List = boost::intrusive::list< Node< ValueT >>;

template< typename ValueT >
size_t node_count( Node< ValueT > const& node )
{
    size_t count = 1;
    for (Node< ValueT > const& child : node.get_children())
        count += node_count( child );
    return count;
}
