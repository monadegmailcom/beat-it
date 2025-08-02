#include "../mp_game.h"

#include <array>
#include <cassert>

namespace tron {

// deliberately restrict the number of players to type int8_t,
//  choose a bigger type if necessary
using PlayerIndex = int8_t;

const PlayerIndex no_player = -1;

enum class Move : uint8_t
{ Left, Right, Up, Down };

using Winner = multiplayer::Winner< PlayerIndex >;
using Draw = multiplayer::Draw;
using Undecided = multiplayer::Undecided;
using GameResult = multiplayer::GameResult< PlayerIndex >;

struct PlayerState
{
    int position = 0;
    bool alive = true;
};

// require: BoardSize > 0, NumberOfPlayers > 0
template< size_t BoardSize, size_t NumberOfPlayers >
struct State
{
    static constexpr size_t board_size = BoardSize;

    std::array< PlayerIndex, BoardSize * BoardSize > board = { no_player };
    std::array< PlayerState, NumberOfPlayers > player_states;
    PlayerIndex next_player_index = 0;
};

template< size_t BoardSize, size_t NumberOfPlayers >
using Game = multiplayer::Game<
    Move, PlayerIndex, State< BoardSize, NumberOfPlayers >>;

template< size_t BoardSize, size_t NumberOfPlayers >
State< BoardSize, NumberOfPlayers > initial_state()
{
    State< BoardSize, NumberOfPlayers > state;
    if constexpr (NumberOfPlayers == 1)
        state.player_states[0].position = BoardSize * BoardSize / 2;
    else if constexpr (NumberOfPlayers == 2)
    {
        state.player_states[0].position = BoardSize / 2;
        state.player_states[1].position = BoardSize * BoardSize - BoardSize / 2;
    }
    else
        static_assert(
            false,
            "tron::initial_state is only implemented for 1 or 2 players." );

    return state;
}

} // tron

namespace multiplayer {

// deliberately restrict the number of possible players to type uint8_t,
//  choose a bigger type if necessary
// require: BoardSize > 0, NumberOfPlayers > 0
template< size_t BoardSize, uint8_t NumberOfPlayers >
struct RuleSet<
    tron::Move, tron::PlayerIndex, tron::State< BoardSize, NumberOfPlayers >>
{
    using state_type = tron::State< BoardSize, NumberOfPlayers >;

    static tron::PlayerIndex next_player_index( state_type const& state )
    { return state.next_player_index; }

    static tron::GameResult result( state_type const& state )
    {
        int first_alive_index = -1;
        for (size_t i = 0; i < NumberOfPlayers; ++i)
            if (state.player_states[i].alive)
            {
                if (first_alive_index == -1)
                    first_alive_index = i;
                else
                    return tron::Undecided();
            }

        if (first_alive_index == -1)
            return tron::Draw(); // all dead
        else
            return tron::Winner( first_alive_index ); // only surviver
    }

    static void set_player(
        int idx, state_type& state,
        tron::PlayerIndex player_index)
    {
        if (idx < 0 || idx >= state.board.size() ||
            state.board[idx] != tron::no_player)
            state.player_states[player_index].alive = false;
        else
            state.board[idx] = player_index;
    }

    static state_type apply( tron::Move const& move, state_type const& state )
    {
        state_type new_state( state );

        tron::PlayerState const& player_state =
            state.player_states[state.next_player_index];
        const auto idx = player_state.position;
        const auto player_index = state.next_player_index;

        if (move == tron::Move::Left)
            set_player( idx - 1, new_state, player_index );
        else if (move == tron::Move::Right)
            set_player( idx + 1, new_state, player_index );
        else if (move == tron::Move::Up)
            set_player( idx - BoardSize, new_state, player_index );
        else if (move == tron::Move::Down)
            set_player( idx + BoardSize, new_state, player_index );

        return new_state;
    }

    // promise: return next valid move if available
    static void next_valid_move(
        std::optional< tron::Move >& move, state_type const& state )
    {
        if (!move)
            move = tron::Move::Left;
        else if (move == tron::Move::Left)
            move = tron::Move::Right;
        else if (move == tron::Move::Right)
            move = tron::Move::Up;
        else if (move == tron::Move::Up)
            move = tron::Move::Down;
        else
            move.reset();
    }
};

} // multiplayer