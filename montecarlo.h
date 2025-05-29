#pragma once

#include "player.h"
#include "node.h"

#include <cmath>

namespace montecarlo {

namespace detail {

template< typename MoveT, typename StateT >
struct Value
{
    Value( Game< MoveT, StateT > const& game, MoveT const& move )
    : game( game ), move( move ), game_result( game.result()), 
      next_move_itr(this->game.begin()) {}
    
    Value( Value&& other ) noexcept
        : game(std::move(other.game)), // Game is truly moved
          move(std::move(other.move)),
          game_result(other.game_result),
          next_move_itr(this->game.begin()), // Iterator bound to the newly moved-to this->game
          points(other.points),
          visits(other.visits) {}

    Game< MoveT, StateT > game;
    MoveT move; // the previous move resulting in this game
    const GameResult game_result; // the cached game result
    // iterator to next valid move not already added as a child node
    typename Game< MoveT, StateT >::MoveItr next_move_itr; 
    double points = 0.0; // 1 for win, 0.5 for draw, 0 for loss
    size_t visits = 0;
};

} // namespace detail {

template< typename MoveT, typename StateT >
using NodeAllocator = ::NodeAllocator< detail::Value< MoveT, StateT > >;

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

// require: game finally ends into result != Undecided
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
double uct( 
    Value< MoveT, StateT > const& value, 
    size_t parent_visits, double exploration )
{
    return 
        1 - value.points / value.visits 
        + exploration * std::sqrt( std::log( parent_visits ) / value.visits );
}

template< typename MoveT, typename StateT >
Node< Value< MoveT, StateT >>* select( 
    Node< Value< MoveT, StateT >>& node, double exploration )
{             
    auto itr = std::max_element( 
        node.begin(), node.end(), 
        [exploration, parent_visits = node.get_value().visits]
        (auto const& a, auto const& b)
        { return   uct( a.get_value(), parent_visits, exploration ) 
                 < uct( b.get_value(), parent_visits, exploration ); });

    return &*itr;
}

template< typename MoveT, typename StateT >
GameResult simulation( 
    Node< Value< MoveT, StateT >>& node, 
    Data< MoveT, StateT >& data, 
    double exploration)
{
    auto& value = node.get_value();
    ++value.visits;

    GameResult backpropagation;

    if (value.game_result != GameResult::Undecided)
        backpropagation = value.game_result;
    else if (value.visits == 1) // PLAYOUT on first visit
    {
        backpropagation = playout< MoveT, StateT >( value.game, data );
        ++data.playout_count;
    }
    else // otherwise create or select child node
    {
        Node< Value< MoveT, StateT >>* selected_node = nullptr;
        if (value.next_move_itr != value.game.end())
        {
            // push front newly created node from next move if available
            const MoveT move = *value.next_move_itr;
            ++value.next_move_itr;
            selected_node = node.push_front_child( 
                Value( value.game.apply( move ), move ));
        }
        else // otherwise SELECT child node
            selected_node = select( node, exploration );

        if (!selected_node)
            throw std::runtime_error( "no node selected" );

        // recursively simulate the selected node
        backpropagation = simulation( *selected_node, data, exploration );
    }

    // update points
    const GameResult player_to_game_result[] = 
        { GameResult::Player1Win, GameResult::Player2Win };
    if (backpropagation == GameResult::Draw)
        value.points += 0.5;
    else if (backpropagation == player_to_game_result[value.game.current_player_index()])
        value.points += 1.0;

    return backpropagation;
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
            Node< detail::Value< MoveT, StateT >>( 
                detail::Value< MoveT, StateT >( game, MoveT()), data.allocator ),
            [&data](auto ptr) { deallocate( data.allocator, ptr ); }
          ),
      exploration( exploration ), simulations( simulations ) 
    {}

    MoveT choose_move() override
    {
        for (size_t i = simulations; i != 0; --i)
            simulation( *root, data, exploration );

        // remove child with most visits
        auto itr = 
            std::max_element( root->begin(), root->end(), 
                [](auto const& a, auto const& b)
                { return a.get_value().visits < b.get_value().visits; } );
        if (itr == root->end())
            throw std::runtime_error( "no move choosen" );

        root->remove_child( itr );
        root.reset( &*itr );

        return root->get_value().move;
    }

    void apply_opponent_move( MoveT const& move ) override
    {
        auto itr = 
            std::find_if( root->begin(), root->end(), 
                [move](auto const& node)
                { return node.get_value().move == move; } );
        auto node = itr == root->end() 
            ? new (this->data.allocator.allocate()) 
                   Node< detail::Value< MoveT, StateT >>( 
                        detail::Value< MoveT, StateT >(
                            root->get_value().game.apply( move ), move), 
                        this->data.allocator ) 
            : &*itr;

        root->remove_child( itr );
        root.reset( node );
    }

    // debug interface ->
    Node< detail::Value< MoveT, StateT >> const& root_node() const { return *root; }
    // <- debug interface
private:
    Data< MoveT, StateT >& data;
    NodePtr< detail::Value< MoveT, StateT > > root;
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
