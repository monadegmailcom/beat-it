#pragma once

#include <cstdint>
#include <vector>

enum PlayerIndex
{
    Player1 = 0,
    Player2
};

PlayerIndex toggle( PlayerIndex );

enum GameResult
{
    Draw = 0,
    Player1Win,
    Player2Win,
    Undecided
};

// for each game specialize game state
template< typename MoveT, typename StateT >
struct GameState
{
    static void append_valid_moves( std::vector< MoveT >& move_stack, PlayerIndex, StateT const& );
    static StateT apply( MoveT const&, PlayerIndex, StateT const& );
    static GameResult result( PlayerIndex, StateT const& state );
};

template< typename MoveT, typename StateT >
class Game
{
public:
    Game( PlayerIndex player_index, StateT const& state ) 
        : player_index( player_index ), state( state ) {}
    
    PlayerIndex current_player_index() const { return player_index; }
    GameResult result() const { return GameState< MoveT, StateT >::result( player_index, state ); }

    // promise: append valid moves to the move_stack
    void append_valid_moves( std::vector< MoveT >& move_stack ) const
    { GameState< MoveT, StateT >::append_valid_moves( move_stack, player_index, state ); }
    // require: move has to be a valid move
    Game apply( MoveT const& move ) const
    { return Game( toggle( player_index ), GameState< MoveT, StateT >::apply( move, player_index, state )); }
    StateT const& get_state() const { return state; }
private:
    PlayerIndex player_index;
    StateT state;
};
