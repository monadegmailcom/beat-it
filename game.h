#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <ranges>

enum PlayerIndex
{
    Player1 = 0,
    Player2
};

PlayerIndex toggle( PlayerIndex );

enum GameResult
{
    Player1Win,
    Player2Win,
    Draw,
    Undecided
};

template< typename MoveT >
class Game
{
public:
    Game( PlayerIndex player_index ) : player_index( player_index ) {}
    virtual ~Game() = default;
    
    PlayerIndex current_player_index() const { return player_index; }
    virtual GameResult result() const = 0;

    // promise: append valid moves to the move_stack
    virtual void append_valid_moves( std::vector< MoveT >& move_stack ) const = 0;
    // require: move has to be a valid move
    virtual std::unique_ptr< Game > apply( MoveT const& ) const = 0;
protected:
    PlayerIndex player_index;
};

template< typename MoveT >
class Player 
{
public:
    virtual ~Player() = default;
    // promise: return a valid move of game
    // require: game is not finished and has at least one valid move
    virtual MoveT choose( Game< MoveT > const& ) = 0;
};
