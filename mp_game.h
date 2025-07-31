#pragma once

#include <vector>
#include <optional>
#include <variant>
#include <iostream>

namespace multiplayer
{

using PlayerIndex = size_t;

struct Draw {};
struct Undecided {};
struct Winner { PlayerIndex player_index; };

using GameResult = std::variant< Draw, Undecided, Winner >;

std::ostream& operator<<( std::ostream&, GameResult );

// Helper base class to provide a default get_valid_moves implementation
template<typename DerivedGameState, typename MoveT, typename StateT>
struct GameStateBase
{
    static void get_valid_moves(
        std::vector< MoveT >& moves, PlayerIndex player_index,
        StateT const& state )
    {
        moves.clear();
        for (;;)
        {
            // Call next_valid_move on the actual specialized GameState
            std::optional< MoveT > move = DerivedGameState::next_valid_move(
                player_index, state );
            if (move)
                moves.push_back(*move);
            else
                break;
        }
    }
};

// each game is required to implement these
template< typename MoveT, typename StateT, size_t NumberOfPlayers >
struct GameState : public GameStateBase<
    GameState< MoveT, StateT, NumberOfPlayers >, MoveT, StateT >
{
    static std::optional< MoveT > next_valid_move(
        PlayerIndex, StateT const& );
    static void get_valid_moves(
        std::vector< MoveT >& moves, PlayerIndex, StateT const& );
    static StateT apply( MoveT const&, PlayerIndex, StateT const& );
    static GameResult result( PlayerIndex, StateT const& );
};

template< typename MoveT, typename StateT >
class Game
{
public:
    using move_type = MoveT;
    using state_type = StateT;

    Game( PlayerIndex player_index, StateT const& state )
        : player_index( player_index ), state( state ) {}
private:
    PlayerIndex player_index; // player to move next
    StateT state;
};

} // namespace multiplayer