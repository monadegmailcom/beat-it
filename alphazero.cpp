#include "alphazero.h"

namespace alphazero {

float game_result_2_score( GameResult game_result, PlayerIndex player_index )
{
    using enum GameResult;
    if (game_result == Draw)
        return 0.0;
    else if (game_result == Player1Win 
             && player_index == PlayerIndex::Player1)
        return 1.0;
    else if (game_result == Player2Win 
             && player_index == PlayerIndex::Player2)
        return 1.0;
    else
        return -1.0;
}

bool atomic_decrement_if_positive(std::atomic< size_t >& atom) {
    // 1. Start by loading the current value.
    size_t current_val = atom.load();

    // 2. Loop until the operation succeeds or the condition is no longer met.
    while (current_val > 0) {
        // 3. Try to swap the current value with the decremented one.
        //    'compare_exchange_weak' will:
        //    - Succeed if 'atom' still equals 'current_val'.
        //    - Fail if 'atom' was changed by another thread, and it will
        //      automatically update 'current_val' with the new value.
        if (atom.compare_exchange_weak(current_val, current_val - 1))
            return true; // Success!
        // If it failed, the loop will retry with the newly loaded 'current_val'.
    }

    return false; // The value was not > 0.
}

} // namespace alphazero


