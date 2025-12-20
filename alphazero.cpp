#include "alphazero.h"

namespace alphazero
{

float game_result_2_score( GameResult game_result, PlayerIndex player_index )
{
    using enum GameResult;
    if ( game_result == Draw )
        return 0.0;
    else if ( game_result == Player1Win && player_index == PlayerIndex::Player1 )
        return 1.0;
    else if ( game_result == Player2Win && player_index == PlayerIndex::Player2 )
        return 1.0;
    else
        return -1.0;
}

} // namespace alphazero
