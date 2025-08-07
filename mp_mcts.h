#pragma once

#include "node.h"
#include "mp_game.h"

#include <random>
#include <array>
#include <functional>

namespace mp {
namespace mcts {

// payload of a node
template<
    typename MoveT, typename StateT, typename PlayerIndexT,
    typename AdditionalPayloadT >
struct Payload : public AdditionalPayloadT
{
    using move_type = MoveT;
    using state_type = StateT;
    using player_index_type = PlayerIndexT;
    using game_type = Game< MoveT, PlayerIndexT, StateT >;
    using game_result_type = GameResult< PlayerIndexT >;

    // notice the this pointer to access the game's begin iterator,
    //  it's important to use the iterator of the copied game
    Payload( game_type const& game, MoveT const& move )
    : game( game ), move( move ), game_result( game.result()),
      next_move_itr( this->game.begin()) {}

    game_type game;
    MoveT move; // the previous move resulting in this game
    const game_result_type game_result; // the cached game result
    // iterator to next valid move not already added as a child node
    typename game_type::MoveItr next_move_itr;
    size_t visits = 0;
};

template<
    typename MoveT, typename StateT, typename PlayerIndexT,
    typename AdditionalPayloadT >
class Player
{
public:
    using game_type = Game< MoveT, PlayerIndexT, StateT >;
    using payload_type =
        Payload< MoveT, StateT, PlayerIndexT, AdditionalPayloadT >;
    using node_type = Node< payload_type >;
    using allocator_type = NodeAllocator< payload_type >;

    Player(
        game_type const& game, size_t simulations, allocator_type& allocator )
    : root( new (allocator.allocate(1))
            node_type( payload_type( game, MoveT()), allocator )),
      simulations( simulations ), allocator( allocator )
    {
        if (!simulations)
            throw std::invalid_argument( "mp::mcts::Player::Player: "
                                         "simulations must be > 0" );
    }

    virtual ~Player()
    {
        root->~node_type();
        allocator.deallocate( root, 1 );
    }
private:
    node_type* root;
    const size_t simulations;
    allocator_type& allocator;
};

// pure montecarlo tree search with playout evaluation
namespace playout {

// additional payload for playout
struct AdditionalPayload
{
    double points = 0.0; // 1 for win, 0.5 for draw, 0 for loss
};

template< typename MoveT, typename StateT, typename PlayerIndexT >
using NodeAllocator = ::NodeAllocator<
    Payload< MoveT, StateT, PlayerIndexT, AdditionalPayload > >;

template< typename MoveT, typename StateT, typename PlayerIndexT >
class Player : public ::mp::mcts::Player<
    MoveT, StateT, PlayerIndexT, AdditionalPayload >
{
public:
    using game_type = Game< MoveT, PlayerIndexT, StateT >;

    Player(
        game_type const& game, size_t simulations,
        NodeAllocator< MoveT, StateT, PlayerIndexT >& allocator )
    : ::mp::mcts::Player< MoveT, StateT, PlayerIndexT, AdditionalPayload >(
            game, simulations, allocator ) {}
};

} // namespace playout {

// use neural network for game value and policies evaluation
namespace alphazero {

// additional payload for alphazero
struct AdditionalPayload
{
    // policy probality (prior) of the parent choosing this move
    float policy = 0.0;
    // value sum of the game outcomes from the next player's perspective:
    // -1 loss, 0 draw, 1 win
    float value_sum = 0.0;
};

} // namespace alphazero {

} // namespace mcts {
} // namespace mp {
