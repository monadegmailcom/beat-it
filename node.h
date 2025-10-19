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

    explicit Node( ValueT const& value ) : value( value ) {}

    Node( Node const& ) = delete;
    Node& operator=( Node const& ) = delete;

    ValueT& get_value() { return value; }
    ValueT const& get_value() const { return value; }

    boost::intrusive::list< Node >& get_children() { return children; }
    boost::intrusive::list< Node > const& get_children() const
    { return children; }
private:
    ValueT value;
    boost::intrusive::list< Node > children;
};

template< typename ValueT >
using List = boost::intrusive::list< Node< ValueT >>;

template< typename ValueT >
Node< ValueT >& copy_tree( 
    GenerationalArenaAllocator& allocator, Node< ValueT > const& node )
{
    using node_type = Node< ValueT >;
    auto* new_node = 
        new (allocator.allocate< node_type >()) node_type( node.get_value());
    for (node_type& child : node.get_children())
        new_node->get_children().push_back( copy_tree( allocator, child ));
    return *new_node;
}

template< typename ValueT >
size_t node_count( Node< ValueT > const& node )
{
    size_t count = 1;
    for (Node< ValueT > const& child : node.get_children())
        count += node_count( child );
    return count;
}
