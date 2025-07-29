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
class Player : public ::Player< MoveT >
{
public:
    Player( Game< MoveT, StateT > const& game, double exploration,
        size_t simulations, unsigned seed,
        NodeAllocator< MoveT, StateT >& allocator )
    : g( seed ), allocator( allocator),
      root( new (allocator.allocate(1))
            Node< detail::Value< MoveT, StateT >>(
                detail::Value< MoveT, StateT >( game, MoveT()), allocator ),
            [&allocator = allocator](auto* ptr) {
                if (ptr) { ptr->~Node(); allocator.deallocate(ptr, 1); }
            }
          ),
      exploration( exploration ), simulations( simulations )
    {}

    double uct( detail::Value< MoveT, StateT > const& value, size_t parent_visits )
    {
        return
            1 - value.points / value.visits
            + exploration * std::sqrt( std::log( parent_visits ) / value.visits );
    }

    // require: game finally ends into result != Undecided
    GameResult playout( Game< MoveT, StateT > game )
    {
        ++playout_count;

        GameResult result;
        for (result = GameResult::Undecided; result == GameResult::Undecided;
            result = game.result())
        {
            // use get_valid_moves because it may be faster than the child iterator
            GameState< MoveT, StateT >::get_valid_moves(
                move_stack, game.current_player_index(), game.get_state());

            if (move_stack.empty())
                throw std::runtime_error( "no valid moves to playout" );

            game = game.apply( move_stack[g() % move_stack.size()] );
        }
        return result;
    }

    Node< detail::Value< MoveT, StateT >>& select(
        Node< detail::Value< MoveT, StateT >>& node )
    {
        auto& value = node.get_value();
        // if another move is available push front newly created child node
        if (value.next_move_itr != value.game.end())
        {
            const MoveT move = *value.next_move_itr++;
            auto child = new (node.get_allocator().allocate(1))
                Node(
                    detail::Value( value.game.apply( move ), move ),
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
                    return  uct( a.get_value(), parent_visits )
                        < uct( b.get_value(), parent_visits );
                });
        }
    }

    GameResult simulation( Node< detail::Value< MoveT, StateT >>& node )
    {
        auto& value = node.get_value();
        ++value.visits;

        GameResult backpropagation;

        if (value.game_result != GameResult::Undecided)
            backpropagation = value.game_result;
        else if (value.visits == 1) // PLAYOUT on first visit
            backpropagation = playout( value.game );
        else // recursively simulate the selected child node
            backpropagation = simulation( select( node ));

        // update points
        const GameResult player_to_game_result[] =
            { GameResult::Player1Win, GameResult::Player2Win };
        if (backpropagation == GameResult::Draw)
            value.points += 0.5;
        else if (backpropagation ==
                     player_to_game_result[value.game.current_player_index()])
            value.points += 1.0;

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
                { return a.get_value().visits < b.get_value().visits; } );
        if (itr == root->get_children().end())
            throw std::runtime_error( "no move choosen" );

        auto new_root = &*itr;
        root->get_children().erase( itr );
        root.reset( new_root );

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
            new_root = new (this->allocator.allocate(1))
                   Node< detail::Value< MoveT, StateT >>(
                        detail::Value< MoveT, StateT >(
                            root->get_value().game.apply( move ), move),
                        this->allocator );
        else
        {
            new_root = &*itr;
            root->get_children().erase( itr );
        }

        root.reset( new_root );
    }

    // debug interface ->
    Node< detail::Value< MoveT, StateT >> const& root_node() const { return *root; }
    // <- debug interface
private:
    std::mt19937 g;
    std::vector< MoveT > move_stack;
    NodeAllocator< MoveT, StateT >& allocator;
    size_t playout_count = 0;

    NodePtr< detail::Value< MoveT, StateT > > root;
    double exploration;
    size_t simulations;
};

} // namespace montecarlo
