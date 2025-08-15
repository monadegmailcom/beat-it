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
    using rule_set_type = RuleSet< MoveT, PlayerIndexT, StateT >;
    using payload_type =
        Payload< MoveT, StateT, PlayerIndexT, AdditionalPayloadT >;
    using node_type = Node< payload_type >;
    using allocator_type = NodeAllocator< payload_type >;

    Player(
        game_type const& game, size_t simulations, allocator_type& allocator )
    : root( new (allocator.allocate(1))
            node_type( payload_type( game, MoveT()), allocator )),
      simulations( simulations )
    {
        if (!simulations)
            throw std::invalid_argument( "mp::mcts::Player::Player: "
                                         "simulations must be > 0" );
    }

    virtual float q_value( payload_type const&, size_t visits ) = 0;

    virtual ~Player()
    {
        auto& allocator = root->get_allocator();
        root->~node_type();
        allocator.deallocate( root, 1 );
    }
private:
    node_type* root;
    const size_t simulations;
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
    using base_type = ::mp::mcts::Player<
        MoveT, StateT, PlayerIndexT, AdditionalPayload >;
    using game_result_type = GameResult< PlayerIndexT >;

    Player(
        base_type::game_type const& game, size_t simulations,
        base_type::allocator_type& allocator,
        float exploration, unsigned seed )
    : base_type( game, simulations, allocator ), exploration( exploration ),
        g( seed ) {}
private:
    float q_value(
        base_type::payload_type const& payload, size_t parent_visits ) override
    {
        // ucb upper confidence bound
        return
            1 - payload.points / payload.visits
            + exploration * std::sqrt( std::log( parent_visits )
              / payload.visits );
    }

    // require: game finally ends into result != Undecided
    // promise: returned game result is Draw or Winner
    game_result_type playout( base_type::game_type const& game )
    {
        using GameResult = base_type::game_result_type;

        GameResult result;
        for (result = GameResult::Undecided; result == GameResult::Undecided;
             result = game.result())
        {
            // use get_valid_moves because it may be faster than the child
            //  iterator
            base_type::rule_set_type::get_valid_moves( moves, game.state());

            if (moves.empty())
                throw std::runtime_error( "no valid moves to playout" );

            game = game.apply( moves[g() % moves.size()] );
        }
        return result;
    }

    base_type::node_type& select( base_type::node_type& node )
    {
        auto& payload = node.get_value();
        // if another move is available push front newly created child node
        // and return
        if (payload.next_move_itr != payload.game.end())
        {
            const MoveT move = *payload.next_move_itr++;
            auto child = new (node.get_allocator().allocate(1))
                Node(
                    payload_type( payload.game.apply( move ), move ),
                    node.get_allocator());

            node.get_children().push_front( *child );
            return *child;
        }
        else // otherwise SELECT child node from children list
        {
            if (node.get_children().empty())
                throw std::runtime_error( "no children to select");

            return *std::ranges::max_element(
                node.get_children(),
                [this, parent_visits = node.get_value().visits]
                (auto const& a, auto const& b)
                {
                    return  ucb( a.get_value(), parent_visits )
                        < ucb( b.get_value(), parent_visits );
                });
        }
    }

    game_result_type simulation( base_type::node_type& node )
    {
        auto& payload = node.get_value();
        ++payload.visits;

        game_result_type backpropagation;

        if (payload.game_result != game_result_type::Undecided)
            backpropagation = payload.game_result;
        else if (payload.visits == 1) // PLAYOUT on first visit
            backpropagation = playout( payload.game );
        else // recursively simulate the selected child node
            backpropagation = simulation( select( node ));

        // update points
        if (backpropagation == game_result_type::Draw)
            payload.points += 0.5;
        else
        {
            auto* winner = std::get_if< Winner< PlayerIndexT > >(
                &backpropagation );
            if (!winner)
                throw std::runtime_error( "mp::mcts::Player::simulation: "
                                          "invalid game result" );
            auto [ player_index ] = *winner;
            if (player_index == payload.game.next_player_index())
                payload.points += 1.0;
        }

        return backpropagation;
    }

    const float exploration;
    std::vector< MoveT > moves;
    std::mt19937 g;
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
