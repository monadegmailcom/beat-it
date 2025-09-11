#include "game.h"

using namespace std;

PlayerIndex toggle( PlayerIndex index )
{
    return index == PlayerIndex::Player1 ? PlayerIndex::Player2 
                                         : PlayerIndex::Player1;
}

ostream& operator<<( ostream& os, GameResult game_result )
{
    using enum GameResult;
    switch (game_result)
    {
        case Draw:
            os << "draw";
            break;
        case Player1Win:
            os << "player 1 wins";
            break;
        case Player2Win:
            os << "player 2 wins";
            break;
        case Undecided:
            os << "undecided";
            break;
    }
    return os;
}