#pragma once

#include "player.h"
#include "game.h"
#include "node.h"

#include <random>

namespace alphazero
{

namespace detail {

template< typename MoveT, typename StateT >
struct Value
{
    Value( Game< MoveT, StateT > const& game, MoveT const& move )
    : game( game ), move( move ), game_result( game.result()) {}

    Game< MoveT, StateT > game;
    MoveT move; // the previous move resulting in this game
    const GameResult game_result; // the cached game result

    size_t visits = 0;
    float nn_policy = 0.0; // probality of choosing this move
    // 1 for win, 0.5 for draw, 0 for loss
    float nn_value = 0.0; // value of the game state
    float nn_value_sum = 0.0;
};

} // namespace detail {

template< typename MoveT, typename StateT >
using NodeAllocator = ::NodeAllocator< detail::Value< MoveT, StateT > >;

// N is the maximal number of moves in any position for one-hot encoding in nn
template< typename MoveT, typename StateT >
struct Data
{
    Data( std::mt19937& g, NodeAllocator< MoveT, StateT >& allocator )
    : g( g ), allocator( allocator ) {}

    virtual ~Data() = default;

    // predict game state value and policy vector from nn
    // promise: policy_vector contain probability distribution of moves
    virtual float predict( Game< MoveT, StateT > const& ) = 0;

    // promise: return index of move in policy_vector
    virtual size_t move_to_policy_index( MoveT const& ) const = 0;

    std::mt19937& g;
    NodeAllocator< MoveT, StateT >& allocator;
    std::vector< float > policy_vector;
};

namespace detail {

template< typename MoveT, typename StateT >
float nn_eval( Node< Value< MoveT, StateT >>& node, Data< MoveT, StateT >& data )
{
    auto& value = node.get_value();
    value.nn_value = data.predict( value.game ); 
    for (auto& child : node.get_children())
        // todo: convert logits to probabilities with softmax?
        child.get_value().nn_policy = data.policy_vector[
            data.move_to_policy_index( child.get_value().move )]; 
    return value.nn_value;
}

float game_result_2_score( GameResult, PlayerIndex );

// upper confidence bound
template< typename MoveT, typename StateT >
float ucb( Value< MoveT, StateT > const& value, 
    size_t parent_visits, float c_base, float c_init )
{
    const float c = 
        std::log( (parent_visits + c_base + 1) / c_base) + c_init;
    float q = 0.0;
    if (value.visits != 0)
        q = value.nn_value_sum / value.visits;
    const float p = value.nn_policy;
        
    return q + c * p * std::sqrt( parent_visits / (value.visits + 1));
}

template< typename MoveT, typename StateT >
Node< Value< MoveT, StateT >>& select( 
    Node< Value< MoveT, StateT >>& node, 
    float c_base, float c_init )
{  
    return *std::ranges::max_element( 
        node.get_children(),
        [c_base, c_init, 
         parent_visits = node.get_value().visits]
        (auto const& a, auto const& b)
        { 
            return ucb( a.get_value(), parent_visits, c_base, c_init ) 
                 < ucb( b.get_value(), parent_visits, c_base, c_init ); 
        });
}

template< typename MoveT, typename StateT >
void add_children( Node< Value< MoveT, StateT >>& node )
{
    auto& value = node.get_value();
    for (MoveT const& move : value.game)
    {
        auto child = new 
            (node.get_allocator().allocate()) 
            Node( 
                Value( value.game.apply( move ), move ), 
                node.get_allocator());                            

        node.get_children().push_front( *child );
    }
}

template< typename MoveT, typename StateT >
float simulation( 
    Node< Value< MoveT, StateT >>& node, 
    Data< MoveT, StateT >& data, 
    float c_base, float c_init)
{
    auto& value = node.get_value();
    ++value.visits;

    float nn_value;

    if (value.game_result != GameResult::Undecided)
        nn_value = game_result_2_score( 
            value.game_result, value.game.current_player_index());
    else if (value.visits == 1) // nn eval on first visit
    {   
        add_children( node );
        nn_value = nn_eval< MoveT, StateT >( node, data );
    }
    else 
        // recursively simulate the selected child node
        // negate sign of return value because its from the opponent's perspective
        nn_value = -simulation( 
            select( node, c_base, c_init ), data, c_base, c_init );

    value.nn_value_sum += nn_value;

    return nn_value;
}

} // namespace detail {

template< typename MoveT, typename StateT >
class Player : public ::Player< MoveT >
{
public:
    Player( 
        Game< MoveT, StateT > const& game, 
        float c_base,
        float c_init,
        size_t simulations,
        size_t opening_moves,
        Data< MoveT, StateT >& data )
    : data( data ), 
      root( new (data.allocator.allocate()) 
            Node< detail::Value< MoveT, StateT >>( 
                detail::Value< MoveT, StateT >( game, MoveT()), data.allocator ),
            [&data](auto ptr) { deallocate( data.allocator, ptr ); }
          ),
      c_base( c_base ), c_init( c_init ), simulations( simulations ),
      opening_moves( opening_moves )
    {}

    MoveT choose_move() override
    {
        for (size_t i = simulations; i != 0; --i)
            simulation( *root, data, c_base, c_init );

        auto itr = root->get_children().begin();

        if (move_count < opening_moves)
        {
            // sample from children in opening phase by visit distribution
            // so we are more versatile in the opening
            size_t visits = 0;
            for (auto const& child : root->get_children())
                visits += child.get_value().visits;
            int r = data.g() % visits;
            for (;itr != root->get_children().end(); ++itr)
            {
                r -= itr->get_value().visits;
                if (r < 0)
                    break;
            }
        }
        else // choose the best after opening phase
        {
            // remove child with most visits
            itr = 
                std::ranges::max_element( root->get_children(),
                    [](auto const& a, auto const& b)
                    { return a.get_value().visits < b.get_value().visits; } );
            if (itr == root->get_children().end())
                throw std::runtime_error( "no move choosen" );
        }

        root->get_children().erase( itr );
        root.reset( &*itr );

        ++move_count;
        
        return root->get_value().move;
    }

    void apply_opponent_move( MoveT const& move ) override
    {
        auto itr = 
            std::ranges::find_if( 
                root->get_children(), 
                [move](auto const& node)
                { return node.get_value().move == move; } );
        Node< detail::Value< MoveT, StateT >>* new_root = nullptr;

        if (itr == root->get_children().end())
            new_root = new (this->data.allocator.allocate()) 
                   Node< detail::Value< MoveT, StateT >>( 
                        detail::Value< MoveT, StateT >(
                            root->get_value().game.apply( move ), move), 
                        this->data.allocator );
        else
        {
            root->get_children().erase( itr );
            new_root = &*itr;
        }

        root.reset( new_root );
    }
private:
    Data< MoveT, StateT >& data;
    NodePtr< detail::Value< MoveT, StateT > > root;
    float c_base;
    float c_init;
    size_t simulations;
    size_t opening_moves;
    size_t move_count = 0;
};



} // namespace alphazero