#include "ultimate_ttt.h"

namespace ultimate_ttt
{

ttt::Symbol game_result_to_symbol( GameResult game_result )
{
    const ttt::Symbol symbols[4] = 
        { ttt::Symbol::Empty, ttt::Symbol::Player1, ttt::Symbol::Player2, 
          ttt::Symbol::Undecided };
    return symbols[game_result];
}

void append_valid_moves( 
    State const& state, ttt::Move big_move, std::vector< Move >& move_stack )
{
    if (state.big_state[big_move] == ttt::Symbol::Undecided)
        for (ttt::Move small_move = 0; small_move != 9; ++small_move)
            if (state.small_states[big_move][small_move] == tic_tac_toe::Symbol::Empty)
                move_stack.push_back( ultimate_ttt::Move {big_move, small_move} );
}

} // namespace ultimate_ttt
