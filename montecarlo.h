#pragma once

#include "player.h"
#include "node.h"
#include "exception.h"

#include <random>
#include <algorithm>
#include <cmath>

namespace montecarlo {

template< typename MoveT, typename StateT >
struct Payload 
{
    // iterator to next valid move not already added as a child node
    typename Game< MoveT, StateT >::MoveItr next_move_itr;
    float points = 0.0; // 1 for win, 0.5 for draw, 0 for loss
    size_t visits = 0;
};

template< typename MoveT, typename StateT >
class Player : public ::Player< MoveT >
{
public:
    using game_type = Game< MoveT, StateT >;
    using payload_type = Payload< MoveT, StateT >;
    using node_type = Node< MoveT, StateT, payload_type >;
    using pre_node_type = PreNode< MoveT, StateT, payload_type >;
    using allocator_type = GenerationalArenaAllocator;

    Player( 
        Game< MoveT, StateT > const& game, float exploration,
        size_t simulations, unsigned seed,
        allocator_type& allocator ) : 
            g( seed ), allocator( allocator), exploration( exploration ), 
            simulations( simulations ), 
            root( new (allocator.allocate< pre_node_type >()) 
                PreNode< MoveT, StateT, payload_type >( 
                    MoveT(), payload_type {.next_move_itr = game.begin()}, 
                    game )) {}

    double uct( payload_type const& payload, size_t parent_visits )
    {
        return
            1 - payload.points / payload.visits
            + exploration * std::sqrtf( 
                std::logf( parent_visits ) / payload.visits );
    }

    // require: game finally ends into result != Undecided
    GameResult playout( game_type game )
    {
        ++playout_count;

        GameResult result;
        for (result = GameResult::Undecided; result == GameResult::Undecided;
             result = game.result())
        {
            // use get_valid_moves because it may be faster than the child 
            // iterator
            GameState< MoveT, StateT >::get_valid_moves(
                move_stack, game.current_player_index(), game.get_state());

            if (move_stack.empty())
                throw beat_it::Exception( "no valid moves to playout" );

            game = game.apply( move_stack[g() % move_stack.size()] );
        }
        return result;
    }

    node_type& select( node_type& node )
    {
        auto& payload = node.get_payload();
        pre_node_type& pre_node = static_cast< pre_node_type& >( node );

        // if another move is available push front newly created child node
        if (payload.next_move_itr != pre_node.get_game().end())
        {
            const MoveT move = *payload.next_move_itr++;
            auto& child = *(new (allocator.allocate< pre_node_type >()) 
                pre_node_type( 
                    move, 
                    payload_type {.next_move_itr = pre_node.get_game().end()}, 
                    pre_node.get_game().apply( move )));

            node.get_children().push_front( child );
            return child;
        }
        else // otherwise SELECT child node from children list
        {
            if (node.get_children().empty())
                throw std::runtime_error( "no children to select");

            return *std::ranges::max_element(
                node.get_children(),
                [this, parent_visits = node.get_payload().visits]
                (auto const& a, auto const& b)
                {
                    return  uct( a.get_payload(), parent_visits )
                        < uct( b.get_payload(), parent_visits );
                });
        }
    }

    GameResult simulation( node_type& node )
    {
        auto& payload = node.get_payload();
        ++payload.visits;

        GameResult backpropagation;

        if (node.get_game_result() != GameResult::Undecided)
            backpropagation = node.get_game_result();
        else if (payload.visits == 1) // PLAYOUT on first visit
            backpropagation = playout( 
                static_cast< pre_node_type& >( node ).get_game());
        else // recursively simulate the selected child node
            backpropagation = simulation( select( node ));

        // update points
        const GameResult player_to_game_result[] =
            { GameResult::Player1Win, GameResult::Player2Win };
        if (backpropagation == GameResult::Draw)
            payload.points += 0.5;
        else if (backpropagation ==
                     player_to_game_result[node.get_current_player_index()])
            payload.points += 1.0;

        return backpropagation;
    }

    MoveT choose_move() override
    {
        for (size_t i = simulations; i != 0; --i)
            simulation( *root );

        // remove child with most visits
        auto itr =
            std::ranges::max_element( root->get_children(),
                [](auto const& a, auto const& b)
                { return a.get_payload().visits < b.get_payload().visits; } );
        if (itr == root->get_children().end())
            throw beat_it::Exception( "no move choosen" );

        allocator.reset(); 
        root = &itr->copy_tree( allocator );
        return root->get_move();
    }

    void apply_opponent_move( MoveT const& move ) override
    {
        auto itr = std::ranges::find_if(
            root->get_children(),
            [move](auto const& node)
            { return node.get_move() == move; } );
        if (itr == root->get_children().end())
            throw beat_it::Exception( "Invalid move.");
        node_type& new_root = *itr;

        allocator.reset();
        root = &new_root.copy_tree( allocator );
    }
    
    node_type& root_node() const { return *root; }
private:
    std::mt19937 g;
    std::vector< MoveT > move_stack;
    allocator_type& allocator;
    size_t playout_count = 0;

    float exploration;
    size_t simulations;
    node_type* root;
};

} // namespace montecarlo
