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

template< typename MoveT >
using MoveRange = std::ranges::subrange< typename std::vector< MoveT >::const_iterator >;

template< typename MoveT >
class Game
{
public:
    Game( PlayerIndex player_index ) : player_index( player_index ) {}
    virtual ~Game() {}
    
    PlayerIndex current_player_index() const { return player_index; }
    virtual bool is_drawn() const = 0;
    virtual bool is_won() const = 0;
    virtual MoveRange< MoveT > valid_moves() const = 0;
    // require: move index in valid_moves()
    virtual std::unique_ptr< Game > apply( size_t index ) const = 0;
protected:
    PlayerIndex player_index;
};

template< typename MoveT >
class DrawnGame : public Game< MoveT >
{
public:
    DrawnGame( PlayerIndex player_index ) : Game< MoveT >( player_index ) {}
    bool is_drawn() const override { return true; }
    bool is_won() const override { return false; }
    MoveRange< MoveT > valid_moves() const override { return {}; }
    std::unique_ptr< Game< MoveT > > apply( size_t index ) const override
    {
        throw std::runtime_error( "invalid move" );
    }
};

template< typename MoveT >
class WonGame : public Game< MoveT >
{
public:
    WonGame( PlayerIndex winner ) : Game< MoveT >( winner ) {}
    PlayerIndex winner() const { return this->player_index; }
    bool is_drawn() const override { return false; }
    bool is_won() const override { return true; }
    MoveRange< MoveT > valid_moves() const override { return {}; }
    std::unique_ptr< Game< MoveT > > apply( size_t index ) const override
    {
        throw std::runtime_error( "invalid move" );
    }
};

template< typename MoveT >
class UndecidedGame : public Game< MoveT >
{
public: 
    UndecidedGame( PlayerIndex player_index ) : Game< MoveT >( player_index ) {}
    virtual ~UndecidedGame() {}
    
    bool is_drawn() const override { return false; }
    bool is_won() const override { return false; }
};

template< typename MoveT >
class Player 
{
public:
    virtual ~Player() {}
    // promise: return index of move in game.valid_moves()
    virtual size_t choose( Game< MoveT > const& ) = 0;
};
