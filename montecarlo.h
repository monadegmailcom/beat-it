#pragma once

#include "player.h"

#include <boost/pool/pool_alloc.hpp>
#include <cmath>
#include <iterator>

namespace montecarlo {

namespace detail {

template< typename MoveT, typename StateT >
struct Node;

} // namespace detail {

template< typename MoveT, typename StateT >
using NodeAllocator = boost::fast_pool_allocator< 
    detail::Node< MoveT, StateT >,
    boost::default_user_allocator_new_delete,
    boost::details::pool::null_mutex >;

template< typename MoveT, typename StateT >
struct Data
{
    Data( std::mt19937& g, NodeAllocator< MoveT, StateT >& allocator ) 
    : g( g ), allocator( allocator ) {}

    std::mt19937& g;
    std::vector< MoveT > move_stack;
    NodeAllocator< MoveT, StateT >& allocator;
    size_t playout_count = 0;
};

namespace detail {

template< typename MoveT, typename StateT >
GameResult playout( Game< MoveT, StateT > game, Data< MoveT, StateT >& data )
{
    GameResult result;
    for (result = GameResult::Undecided; result == GameResult::Undecided; 
         result = game.result())
    {
        // use get_valid_moves because it may be faster than the child iterator
        GameState< MoveT, StateT >::get_valid_moves( 
            data.move_stack, game.current_player_index(), game.get_state());
        
        if (data.move_stack.empty())
            throw std::runtime_error( "no valid moves to playout" );

        game = game.apply( data.move_stack[data.g() % data.move_stack.size()] );        
    }
    return result;
}

template< typename MoveT, typename StateT >
struct Node
{
    Node( Game< MoveT, StateT > const& game, MoveT const& move, NodeAllocator< MoveT, StateT >& allocator )
    : game( game ), next_move_itr( this->game.begin()), move( move ), 
      game_result( game.result()), allocator( allocator ) {}

    Node( Node const& ) = delete;
    Node& operator=( Node const& ) = delete;

    ~Node() 
    {
        for (auto child = first_child; child;)
        {
            auto next = child->next_sibling;
            child->~Node();
            allocator.deallocate( child );
            child = next;
        }
    }

    Game< MoveT, StateT > game;
    // next valid move iterator not already added as a child node
    typename Game< MoveT, StateT >::MoveItr next_move_itr; 
    const MoveT move; // the previous move resulting in this game
    const GameResult game_result; // the cached game result
    NodeAllocator< MoveT, StateT >& allocator;

    Node* next_sibling = nullptr;
    Node* prev_sibling = nullptr;
    Node* first_child = nullptr;

    double points = 0.0; // 1 for win, 0.5 for draw, 0 for loss
    size_t visits = 0;

    class ChildItr : public boost::iterator_facade<
                            ChildItr, // Derived class
                            Node, // Value type
                            std::bidirectional_iterator_tag,
                            Node& // Reference type
                        >
    {
    public:
        explicit ChildItr( Node* node ) : node( node ) {}

        Node& dereference() const { return *node; }

        // Pre-increment operator
        void increment() 
        {
            node = node->next_sibling;
        }

        // Post-increment operator
        void decrement() 
        {
            node = node->prev_sibling;
        }

        // Equality comparison
        bool equal(const ChildItr& other) const { return node == other.node; }
    private:
        friend struct Node;
        friend class boost::iterator_core_access;
        Node* node;
    };

    void remove_child( ChildItr child )
    {
        if (child.node == nullptr)
            return;

        auto prev = child.node->prev_sibling;
        auto next = child.node->next_sibling;
        if (prev)
            prev->next_sibling = next;
        else
            first_child = next;
        if (next)
            next->prev_sibling = prev;
        child.node->next_sibling = nullptr;
        child.node->prev_sibling = nullptr;
    }

    ChildItr begin() const { return ChildItr( first_child ); }
    ChildItr end() const { return ChildItr( nullptr ); }
};

template< typename MoveT, typename StateT >
size_t children_count( Node< MoveT, StateT > const& node ) 
{
    return std::distance( node.begin(), node.end());
}

template< typename MoveT, typename StateT >
size_t node_count( Node< MoveT, StateT > const& node )
{
    size_t count = 1;
    for (Node< MoveT, StateT > const& child : node)
        count += node_count( child );
    return count;
}

template< typename MoveT, typename StateT >
Node< MoveT, StateT >* push_front_child( 
    Node< MoveT, StateT >& node, MoveT const& child_move )
{
    Node< MoveT, StateT >* const child = 
        new (node.allocator.allocate()) Node< MoveT, StateT >( 
            node.game.apply( child_move ), child_move, node.allocator );                            
    child->next_sibling = node.first_child;
    node.first_child = child;
    return child;
}

template< typename MoveT, typename StateT >
double uct( Node< MoveT, StateT > const& node, size_t parent_visits, double exploration )
{
    return 
        1 - node.points / node.visits 
        + exploration * std::sqrt( std::log( parent_visits ) / node.visits );
}

template< typename MoveT, typename StateT >
Node< MoveT, StateT >* select( Node< MoveT, StateT >& node, double exploration )
{             
    typename Node< MoveT, StateT >::ChildItr itr = std::max_element( 
        node.begin(), node.end(), 
        [exploration, parent_visits = node.visits]
        (Node< MoveT, StateT > const& a, Node< MoveT, StateT > const& b)
        { return   uct( a, parent_visits, exploration ) 
                 < uct( b, parent_visits, exploration ); });

    return &*itr;
}

template< typename MoveT, typename StateT >
GameResult simulation( 
    Node< MoveT, StateT >& node, 
    Data< MoveT, StateT >& data, 
    double exploration)
{
    ++node.visits;

    GameResult backpropagation;

    if (node.game_result != GameResult::Undecided)
        backpropagation = node.game_result;
    else if (node.visits == 1) // PLAYOUT on first visit
    {
        backpropagation = playout< MoveT, StateT >( node.game, data );
        ++data.playout_count;
    }
    else // otherwise create or select child node
    {
        Node< MoveT, StateT >* selected_node = node.next_move_itr != node.game.end()
            // push front newly created node from next move if available
            ? push_front_child( node, *node.next_move_itr++ )
            // otherwise SELECT child node
            : select( node, exploration );

        if (!selected_node)
            throw std::runtime_error( "no node selected" );

        // recursively simulate the selected node
        backpropagation = simulation( *selected_node, data, exploration );
    }

    // update points
    const GameResult player_to_game_result[] = 
        { GameResult::Player1Win, GameResult::Player2Win };
    if (backpropagation == GameResult::Draw)
        node.points += 0.5;
    else if (backpropagation == player_to_game_result[node.game.current_player_index()])
        node.points += 1.0;

    return backpropagation;
}

template< typename MoveT, typename StateT >
using NodePtr = std::unique_ptr< Node< MoveT, StateT >, 
                                 std::function< void (Node< MoveT, StateT >*) > 
                               >;

template< typename MoveT, typename StateT >                               
void deallocate( NodeAllocator< MoveT, StateT >& allocator, Node< MoveT, StateT >* ptr )
{
    if (ptr)
    {
        ptr->~Node< MoveT, StateT >(); // Call the destructor
        allocator.deallocate( ptr ); // Deallocate memory
    }
}

} // namespace detail

template< typename MoveT, typename StateT >
class Player : public ::Player< MoveT >
{
public:
    Player( 
        Game< MoveT, StateT > const& game, 
        double exploration,
        size_t simulations,
        Data< MoveT, StateT >& data )
    : data( data ), 
      root( new (data.allocator.allocate()) 
            detail::Node< MoveT, StateT >( game, MoveT(), data.allocator ),
            [&data](detail::Node< MoveT, StateT >* ptr) { deallocate( data.allocator, ptr ); }
          ),
      exploration( exploration ), simulations( simulations ) 
    {}

    MoveT choose_move() override
    {
        for (size_t i = simulations; i != 0; --i)
            simulation( *root, data, exploration );

        // remove child with most visits
        typename detail::Node< MoveT, StateT >::ChildItr itr = 
            std::max_element( root->begin(), root->end(), 
                [](detail::Node< MoveT, StateT > const& a, detail::Node< MoveT, StateT > const& b)
                { return a.visits < b.visits; } );
        if (itr == root->end())
            throw std::runtime_error( "no move choosen" );

        root->remove_child( itr );
        root.reset( &*itr );

        return root->move;
    }

    void apply_opponent_move( MoveT const& move ) override
    {
        typename detail::Node< MoveT, StateT >::ChildItr itr = 
            std::find_if( root->begin(), root->end(), 
                [move](detail::Node< MoveT, StateT > const& node)
                { return node.move == move; } );
        auto node = itr == root->end() 
            ? new (this->data.allocator.allocate()) 
                   detail::Node< MoveT, StateT >( 
                        root->game.apply( move ), move, this->data.allocator ) 
            : &*itr;

        root->remove_child( itr );
        root.reset( node );
    }

    // debug interface ->
    detail::Node< MoveT, StateT > const& root_node() const { return *root; }
    // <- debug interface
private:
    Data< MoveT, StateT >& data;
    detail::NodePtr< MoveT, StateT > root;
    double exploration;
    size_t simulations;
};

template< typename MoveT, typename StateT >
using Buffer = char[sizeof( Player< MoveT, StateT > )];

template< typename MoveT, typename StateT >
PlayerFactory< MoveT > player_factory(
    Game< MoveT, StateT > const& game, 
    double exploration,
    size_t simulations,
    Data< MoveT, StateT >& data,
    Buffer< MoveT, StateT > raw_data )
{
    return [&game, &data, exploration, simulations, raw_data]()
    { 
        return std::unique_ptr< ::Player< MoveT >, void(*)( ::Player< MoveT >*) >( 
            new (raw_data) Player< MoveT, StateT >( game, exploration, simulations, data ), 
            [](::Player< MoveT >* p){p->~Player(); }); 
    };
}

} // namespace montecarlo
