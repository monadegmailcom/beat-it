#include "alphazero.h"

namespace alphazero {
namespace detail {

double game_result_2_score( GameResult game_result, PlayerIndex player_index )
{
    if (game_result == GameResult::Draw)
        return 0.5;
    else if (game_result == GameResult::Player1Win && player_index == Player1)
        return 1.0;
    else if (game_result == GameResult::Player2Win && player_index == Player2)
        return 1.0;
    else 
        return 0.0;
}

} // namespace detail {
} // namespace alphazero


