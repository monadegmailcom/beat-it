#pragma once
#include "game.h"

template< typename MoveT, typename StateT >
class Player 
{
public:
    virtual ~Player() = default;
    // promise: return a valid move of game
    // require: game is not finished and has at least one valid move
    virtual MoveT choose( Game< MoveT, StateT > const& ) = 0;
};
