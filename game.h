#pragma once

#include <cstdint>
#include <memory>
#include <vector>

enum PlayerIndex
{
    Player1 = 0,
    Player2
};

PlayerIndex toggle( PlayerIndex );

template< typename MoveT >
class UndecidedGame;

template< typename MoveT >
class Player 
{
public:
    Player( PlayerIndex index ) : index( index ) {}
    virtual ~Player() {}
    PlayerIndex get_index() const { return index; }
    void set_index( PlayerIndex index ) { this->index = index; }

    // require: move iterator has to be in game.valid_moves()
    virtual std::vector< MoveT >::const_iterator choose( 
        UndecidedGame< MoveT > const& game ) = 0;
private:
    PlayerIndex index;
};

class Game
{
public:
    Game( PlayerIndex player_index ) : player_index( player_index ) {}
    virtual ~Game() {}
    
    PlayerIndex current_player_index() const { return player_index; }
protected:
    PlayerIndex player_index;
};

template< typename MoveT >
class UndecidedGame : public Game
{
public: 
    UndecidedGame( PlayerIndex player_index ) : Game( player_index ) {}
    virtual ~UndecidedGame() {}
    virtual std::vector< MoveT > const& valid_moves() const = 0;

    // require: move iterator has to be in valid_moves()
    virtual std::unique_ptr< Game > apply( std::vector< MoveT >::const_iterator) const = 0;
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
    PlayerIndex winner() const
    {
        return player_index;
    }
};
