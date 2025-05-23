#include "game.h"

using namespace std;

PlayerIndex toggle( PlayerIndex index )
{
    return index == Player1 ? Player2 : Player1;
}
