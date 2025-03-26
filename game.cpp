#include "game.h"

using namespace std;

PlayerIndex toggle( PlayerIndex index )
{
    return PlayerIndex( (index + 1) % 2);
    // which one is faster?
    //return index == Player1 ? Player2 : Player1;
}
