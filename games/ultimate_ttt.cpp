#include "ultimate_ttt.h"
#include <algorithm>

using namespace std;

namespace uttt
{

const State empty_state = 
    { { ttt::empty_state, ttt::empty_state, ttt::empty_state,
        ttt::empty_state, ttt::empty_state, ttt::empty_state,
        ttt::empty_state, ttt::empty_state, ttt::empty_state},
      { GameResult::Undecided, GameResult::Undecided, GameResult::Undecided,
        GameResult::Undecided, GameResult::Undecided, GameResult::Undecided,
        GameResult::Undecided, GameResult::Undecided, GameResult::Undecided },
      ttt::no_move };

namespace console {

Move HumanPlayer::choose( Game const& game )
{
    vector< Move > valid_moves;
    game.append_valid_moves( valid_moves );

    cout << "valid moves:\n";
    for (auto const& move : valid_moves)
        cout << "(" << int( move.big_move) << "," << int(move.small_move) << "), ";
    cout << '\n';

    State const& state = game.get_state();
    while (true)
    {
        unsigned big_move;
        if (   state.next_big_move == ttt::no_move 
            || state.big_state[state.next_big_move] != GameResult::Undecided)
        {
            cout << "big move? ";
            cin >> big_move;
            cin.ignore();
            if (big_move >= 9)
            {
                cout << "invalid big move" << endl;
                continue;
            }
        }
        else
            big_move = state.next_big_move;

        cout << "small move? ";
        unsigned small_move;
        cin >> small_move;
        cin.ignore();
        if (small_move >= 9)
        {
            cout << "invalid small move" << endl;
            continue;
        }

        const Move move = Move {ttt::Move( big_move ), ttt::Move( small_move )};
        if (!ranges::contains( valid_moves.begin(), valid_moves.end(), move))
        {
            cout << "not a valid move" << endl;
            continue;
        }

        return move;
    }
}

} // namespace console {

bool operator==( uttt::Move const& lhs, uttt::Move const& rhs )
{
    return lhs.big_move == rhs.big_move && lhs.small_move == rhs.small_move;
}

namespace minimax {

Player::Player( double weight, unsigned depth, std::mt19937& g ) 
    : ::minimax::Player< Move, State >( depth, g ), weight( weight ) {}

double Player::score( Game const& game ) const
{
    State const& state = game.get_state();
    
    double score = 0.0;
    for (auto const& win : ttt::wins)
    {
        uint8_t player1_points = 0;
        uint8_t player2_points = 0;
        uint8_t draw_points = 0;
        for (auto const& index : win)
        {
            if (state.big_state[index] == GameResult::Player1Win)
                ++player1_points;
            else if (state.big_state[index] == GameResult::Player2Win)
                ++player2_points;
            else if (state.big_state[index] == GameResult::Draw)
                ++draw_points;
        }
        if (player1_points == 0 && draw_points == 0)
            score += player2_points;
        else if (player2_points == 0 && draw_points == 0)
            score -= player1_points;
    }

    score *= weight;
    for (uint8_t i = 0; i != 9; ++i)
        if (state.big_state[i] == GameResult::Undecided)
            score += ttt::minimax::score( state.small_states[i] );

    return score;
}

} // namespace minimax {
} // namespace uttt

void append_valid_moves( 
    uttt::State const& state, ttt::Move big_move, std::vector< uttt::Move >& move_stack )
{
    if (state.big_state[big_move] == GameResult::Undecided)
        for (ttt::Move small_move = 0; small_move != 9; ++small_move)
            if (state.small_states[big_move][small_move] == ttt::Symbol::Empty)
                move_stack.push_back( uttt::Move {big_move, small_move} );
}

void GameState< uttt::Move, uttt::State >::append_valid_moves( 
    std::vector< uttt::Move >& move_stack, PlayerIndex, 
    uttt::State const& state )
{
    if (state.next_big_move != ttt::no_move) 
        ::append_valid_moves( state, state.next_big_move, move_stack );
    else
        for (ttt::Move big_move = 0; big_move != 9; ++big_move)
            ::append_valid_moves( state, big_move, move_stack );
}

uttt::State GameState< uttt::Move, uttt::State >::apply( 
    uttt::Move const& move, PlayerIndex player_index, uttt::State const& state )
{
    if (   move.big_move >= 9 
        || (state.next_big_move != ttt::no_move && move.big_move != state.next_big_move))
        throw std::invalid_argument( "invalid big move" );

    uttt::State new_state = state;
    new_state.small_states[move.big_move] = GameState< ttt::Move, ttt::State >::apply( 
        move.small_move, player_index, state.small_states[move.big_move] );
    new_state.big_state[move.big_move] = GameState< ttt::Move, ttt::State >::result( 
        player_index, new_state.small_states[move.big_move]);
    new_state.next_big_move = (new_state.big_state[move.small_move] == GameResult::Undecided)
        ? move.small_move 
        : ttt::no_move;

    return new_state;
}

GameResult GameState< uttt::Move, uttt::State >::result( 
    PlayerIndex, uttt::State const& state )
{
    // check for wins
    for (const auto& win : ttt::wins)
    {
        const GameResult symbol = state.big_state[win[0]];
        if (symbol == GameResult::Player1Win && symbol == state.big_state[win[1]] 
            && symbol == state.big_state[win[2]])
            return GameResult::Player1Win;
        if (symbol == GameResult::Player2Win && symbol == state.big_state[win[1]] 
            && symbol == state.big_state[win[2]])
            return GameResult::Player2Win;
    }

    // check for undecided
    for (auto game_result : state.big_state)
        if (game_result == GameResult::Undecided)
            return GameResult::Undecided;

    // otherwise its a draw
    return GameResult::Draw;
}

char game_result_to_char( GameResult game_result )
{
    const char symbols[4] = { ' ', 'X', 'O', 'D' };
    return symbols[game_result];
}

ostream& operator<<( ostream& stream, uttt::Game const& game )
{
    uttt::State const& state = game.get_state();
    for (size_t i = 0; i != 3; ++i)
    {
        for (size_t i2 = 0; i2 != 3; ++i2)
        {
            for (size_t j = 0; j != 3; ++j)
            {
                const size_t idx = i * 3 + j;
                if (state.big_state[idx] != GameResult::Undecided)
                {
                    for (size_t j2 = 0; j2 != 3; ++j2)
                    {
                        if (i2 == 1 && j2 == 1)
                            stream << state.big_state[idx] << ' ';
                        else
                            stream << "  ";
                    }
                }
                else
                    for (size_t j2 = 0; j2 != 3; ++j2)
                        stream << state.small_states[idx][i2 * 3 + j2] << ' ';
                if (j != 2)
                    stream << ' ';
            }
            stream << '\n';
        }
        if (i != 2)
            stream << '\n';
    }
    stream << '\n';

    // Print the big state and last small move
    stream << "\nbig state:\n";
    for (auto i = 0; i != 5; ++i)
        stream << '-';
    stream << '\n';
    for (size_t index = 0; index != 9; ++index)
    {
        if (index % 3 == 0)
            stream << '|';
        stream << game_result_to_char( game.get_state().big_state[index] );
        if ((index + 1) % 3 == 0)
            stream << "|\n";
    }
    for (auto i = 0; i != 5; ++i)
        stream << '-';
    stream << '\n';
    
    stream << "next big move: (";
    if (game.get_state().next_big_move != ttt::no_move) 
        stream << int( game.get_state().next_big_move );
    else
        stream << "choose free";
    stream << ")\n";

    return stream;
}