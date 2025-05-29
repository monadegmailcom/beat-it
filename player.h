#pragma once
#include <memory>
#include <functional>

// require: choose_move and set_opponent_move are called alternatingly
template< typename MoveT >
class Player 
{
public:
    virtual ~Player() = default;

    // promise: return a valid move of game
    // require: game is not finished and has at least one valid move
    virtual MoveT choose_move() = 0;
    // require: move is a valid move
    virtual void apply_opponent_move( MoveT const& ) = 0; 
};

// for match
template< typename MoveT >
using PlayerFactory = std::function< Player< MoveT >* () >;
