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

class Game
{
public:
    Game( PlayerIndex player_index ) : player_index( player_index ) {}
    virtual ~Game() {}
    
    PlayerIndex current_player_index() const { return player_index; }
protected:
    PlayerIndex player_index;
};

class DrawnGame : public Game
{
public:
    DrawnGame( PlayerIndex player_index ) : Game( player_index ) {}
};

class WonGame : public Game
{
public:
    WonGame( PlayerIndex winner ) : Game( winner ) {}
    PlayerIndex winner() const { return player_index; }
};

template< typename MoveT >
class UndecidedGame : public Game
{
public: 
    UndecidedGame( PlayerIndex player_index ) : Game( player_index ) {}
    virtual ~UndecidedGame() {}
    virtual std::ranges::subrange< typename std::vector< MoveT >::const_iterator > valid_moves() const = 0;

    // require: move index in valid_moves()
    virtual std::unique_ptr< Game > apply( size_t index ) const = 0;
};

template< typename MoveT >
class Player 
{
public:
    virtual ~Player() {}
    // promise: return index of move in game.valid_moves()
    virtual size_t choose( 
        UndecidedGame< MoveT > const& ) = 0;
};
