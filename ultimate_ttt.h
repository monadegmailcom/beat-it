#include "tic_tac_toe.h"

namespace ttt = tic_tac_toe;

namespace ultimate_ttt
{

struct Move {
    ttt::Move big_move;
    ttt::Move small_move;
};

// require: small_states, big_state and last_small_move are consistent
struct State 
{
    std::array< ttt::State, 9 > small_states;
    ttt::State big_state;
    ttt::Move last_small_move = ttt::no_move;
};

using Game = ::Game< Move, State >;
using Player = ::Player< Move, State >;

ttt::Symbol game_result_to_symbol( GameResult game_result );

const State empty_state = { {ttt::empty_state, ttt::empty_state, ttt::empty_state,
                              ttt::empty_state, ttt::empty_state, ttt::empty_state,
                              ttt::empty_state, ttt::empty_state, ttt::empty_state},
                            ttt::empty_state, ttt::no_move };

void append_valid_moves( 
    State const& state, ttt::Move big_move, std::vector< Move >& move_stack );

} // namespace ultimate_ttt

template<>
struct GameState< ultimate_ttt::Move, ultimate_ttt::State >
{
    static void append_valid_moves( 
        std::vector< ultimate_ttt::Move >& move_stack, PlayerIndex, 
        ultimate_ttt::State const& state )
    {
        if (state.last_small_move != ttt::no_move) 
            ultimate_ttt::append_valid_moves( state, state.last_small_move, move_stack );
        else
            for (ttt::Move big_move = 0; big_move != 9; ++big_move)
                ultimate_ttt::append_valid_moves( state, state.last_small_move, move_stack );
    }

    static ultimate_ttt::State apply( 
        ultimate_ttt::Move const& move, PlayerIndex player_index, ultimate_ttt::State const& state )
    {
        if (move.big_move >= 9 || move.big_move != state.last_small_move)
            throw std::invalid_argument( "invalid big move" );
        ultimate_ttt::State new_state = state;
        new_state.small_states[move.big_move] = GameState< ttt::Move, ttt::State >::apply( 
            move.small_move, player_index, state.small_states[move.big_move] );
        new_state.big_state[move.big_move] = 
            ultimate_ttt::game_result_to_symbol( GameState< ttt::Move, ttt::State >::result( 
            player_index, new_state.small_states[move.big_move]));
        new_state.last_small_move = move.small_move;

        return new_state;
    }

    static GameResult result( PlayerIndex player_index, ultimate_ttt::State const& state )
    {
        return GameState< ttt::Move, ttt::State >::result( player_index, state.big_state );
    }
};
