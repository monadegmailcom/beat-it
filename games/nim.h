#include "../minimax.h"

#include <array>
#include <algorithm>

namespace nim {

struct Move
{
    size_t heap_index;
    size_t count;
};

const Move no_move = { 0, 0 };

bool operator==( Move const& lhs, Move const& rhs );

template< size_t N >
using State = std::array< size_t, N >;

template< size_t N >
using Game = ::Game< Move, State< N > >;

using Player = ::Player< Move >;

namespace minimax {

template< size_t N >
using Player = ::minimax::Player< Move, State< N > >;

template< size_t N >

using PlayerFactory = ::PlayerFactory< Move >;

} // namespace minimax {

namespace console {

template< size_t N >
class HumanPlayer : public Player
{
public:
    HumanPlayer( Game< N > const& game ) : game( game ) {}

    Move choose_move() override
    {
        std::vector< nim::Move > valid_moves;
        auto const& heaps = game.get_state();
        append_valid_moves( valid_moves, heaps );
        return choose(
            game.current_player_index(),
            std::vector< size_t >( heaps.begin(), heaps.end()), valid_moves );
    }
    void apply_opponent_move( Move const& move ) override
    {
        game = game.apply( move );
    }
private:
     Game< N > game;
};

Move choose( PlayerIndex, std::vector< size_t > const& heaps, std::vector< nim::Move > const& valid_moves );

} // namespace console {
} // namespace nim {

template< size_t N >
struct GameState< nim::Move, nim::State< N > >
{
    static void next_valid_move(
        std::optional< nim::Move >& move, PlayerIndex, nim::State< N > const& state )
    {
        if (!move)
            move = nim::Move {0, 1}; // first possibly valid move
        else
            ++move->count; // possible next move
        while (true)
        {
            if (move->heap_index >= state.size()) // no valid move possible anymore
            {
                move.reset();
                break;
            }
            else if (move->count <= state[move->heap_index]) // move is valid
                break;

            // try next move
            ++move->heap_index;
            move->count = 1;
        }
    }

   static void get_valid_moves(
        std::vector< nim::Move >& moves, PlayerIndex, nim::State< N > const& state )
    {
        moves.clear();
        for (size_t heap = 0; heap != state.size(); ++heap)
            for (size_t count = 1; count <= state[heap]; ++count)
                moves.push_back( nim::Move{ heap, count } );
    }

    static nim::State< N > apply( nim::Move const& move, PlayerIndex, nim::State< N > const& state )
    {
        if (move.heap_index >= state.size())
            throw std::invalid_argument(
                "invalid move heap index (" + std::to_string(move.heap_index) + ")" );
        if (move.count > state[move.heap_index])
            throw std::invalid_argument(
                "heap size (" + std::to_string(state[move.heap_index])
                + ") < move count (" + std::to_string(move.count) + ")" );

        nim::State< N > new_state( state );
        new_state[move.heap_index] -= move.count;

        return new_state;
    }

    static GameResult result( PlayerIndex player_index, nim::State< N > const& state )
    {
        if (std::all_of( state.begin(), state.end(), []( size_t heap_size ) { return heap_size == 0; } ))
        {
            if (player_index == Player1)
                return GameResult::Player1Win;
            else
                return GameResult::Player2Win;
        }
        else
            return GameResult::Undecided;
    }
};
