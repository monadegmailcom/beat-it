#include "game.h"

using namespace std;

PlayerIndex toggle( PlayerIndex index )
{
    return index == Player1 ? Player2 : Player1;
}

ostream& operator<<( ostream& os, GameResult game_result )
{
    switch (game_result)
    {
        case GameResult::Draw:
            os << "draw";
            break;
        case GameResult::Player1Win:
            os << "player 1 wins";
            break;
        case GameResult::Player2Win:
            os << "player 2 wins";
            break;
        case GameResult::Undecided:
            os << "undecided";
            break;
    }
    return os;
}