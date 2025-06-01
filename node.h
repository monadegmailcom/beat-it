#pragma once

#include <boost/pool/pool_alloc.hpp>
#include <boost/iterator/iterator_facade.hpp>

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
class Node
{
public:
    Node( ValueT value, NodeAllocator< ValueT >& allocator ) 
    : value( std::move( value )), allocator( allocator ) {}

    Node( Node const& ) = delete;
    Node& operator=( Node const& ) = delete;

    ~Node() 
    {
        // destruct all children and deallocate their memory
        for (auto child = first_child; child;)
        {
            auto next = child->next_sibling;
            child->~Node();
            allocator.deallocate( child );
            child = next;
        }
    }

    ValueT& get_value() { return value; }
    ValueT const& get_value() const { return value; }

    class ChildItr : public boost::iterator_facade<
                            ChildItr, // Derived class
                            Node, // Value type
                            std::bidirectional_iterator_tag,
                            Node& // Reference type
                            >
    {
    public:
        explicit ChildItr( Node* node ) : node( node ) {}
        explicit ChildItr() : node( nullptr ) {}

        // Dereference operator
        Node& dereference() const { return *node; }

        // Pre-increment operator
        void increment() { node = node->next_sibling; }

        void decrement() { node = node->prev_sibling; }

        // Equality comparison
        bool equal(const ChildItr& other) const { return node == other.node; }
    private:
        friend class Node;
        friend class boost::iterator_core_access;
        Node* node;
    };

    void remove_child( ChildItr child )
    {
        if (child.node == nullptr)
            return;

        if (child.node->prev_sibling)
            child.node->prev_sibling->next_sibling = child.node->next_sibling;
        else
            first_child = child.node->next_sibling;

        if (child.node->next_sibling)
            child.node->next_sibling->prev_sibling = child.node->prev_sibling;

        // isolate removed child from children list
        child.node->next_sibling = nullptr;
        child.node->prev_sibling = nullptr;
    }

    Node* push_front_child( ValueT value )
    {
        Node* const child = 
            new (allocator.allocate()) 
            Node( std::move( value ), allocator );                            
        child->next_sibling = first_child;
        if (first_child)
            first_child->prev_sibling = child;
        first_child = child;
        return child;
    }

    ChildItr begin() const { return ChildItr( first_child ); }
    ChildItr end() const { return ChildItr( nullptr ); }

    void sort_prefix( 
        ChildItr suffix_itr, 
        std::vector< Node* >& stack,
        std::function< bool (ValueT const&, ValueT const&) > comp )
    {
        stack.clear();
        for (auto itr = begin(); itr != suffix_itr; ++itr) 
            stack.push_back( &*itr );

        // relink
        if (!stack.empty())
        {
            // use of stable sort is important because for moves with the same evaluation
            // following moves may be worse than the first due to alpha/beta pruning
            std::stable_sort( 
                stack.begin(), stack.end(),
                [comp](auto a, auto b)
                { return comp( a->get_value(), b->get_value()); });

            first_child = stack.front();
            first_child->prev_sibling = nullptr;

            for (size_t i = 0; i < stack.size() - 1; ++i) 
            {
                stack[i]->next_sibling = stack[i+1];
                stack[i+1]->prev_sibling = stack[i];
            }

            stack.back()->next_sibling = suffix_itr.node;
            if (suffix_itr.node) 
                suffix_itr.node->prev_sibling = stack.back();
        }
    }
private:
    ValueT value;
    NodeAllocator< ValueT >& allocator;

    Node* next_sibling = nullptr;
    Node* prev_sibling = nullptr;
    Node* first_child = nullptr;
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
size_t children_count( Node< ValueT > const& node ) 
{
    return std::distance( node.begin(), node.end());
}

template< typename ValueT >
size_t node_count( Node< ValueT > const& node )
{
    size_t count = 1;
    for (Node< ValueT > const& child : node)
        count += node_count( child );
    return count;
}
