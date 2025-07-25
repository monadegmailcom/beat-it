#include "ultimate_ttt.h"
#include "../libtorch_util.h"

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
      ttt::no_move,
      GameResult::Undecided };

namespace console {

Move HumanPlayer::choose_move()
{
    vector< Move > valid_moves;
    for (auto itr = game.begin(), end = game.end(); itr != end; ++itr)
        valid_moves.push_back( *itr );

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
        if (std::find( valid_moves.begin(), valid_moves.end(), move)
            == valid_moves.end())
        {
            cout << "not a valid move" << endl;
            continue;
        }

        game = game.apply( move );
        return move;
    }
}

void HumanPlayer::apply_opponent_move( Move const& move )
{
    game = game.apply( move );
}

} // namespace console {

bool operator==( uttt::Move const& lhs, uttt::Move const& rhs )
{
    return lhs.big_move == rhs.big_move && lhs.small_move == rhs.small_move;
}

double score_function( uttt::State const& state, double weight )
{
    double score = 0.0;
    for (auto const& win : ttt::wins)
    {
        bool skip_win = false;
        uint8_t player1_points = 0;
        uint8_t player2_points = 0;
        for (auto const& index : win)
        {
            const GameResult result = state.big_state[index];
            if (result == GameResult::Draw)
            {
                skip_win = true;
                break;
            }
            if (result == GameResult::Player1Win)
                ++player1_points;
            else if (result == GameResult::Player2Win)
                ++player2_points;
        }
        if (skip_win)
            continue;
        if (player1_points == 0)
            score += player2_points;
        else if (player2_points == 0)
            score -= player1_points;
    }

    score *= weight;
    for (uint8_t i = 0; i != 9; ++i)
        if (state.big_state[i] == GameResult::Undecided)
            score += ttt::minimax::score( state.small_states[i] );

    return score;
}

namespace minimax {

Player::Player( 
    Game const& game, double weight, unsigned depth, Data& data ) 
    : ::minimax::Player< Move, State >( game, depth, data ), weight( weight ) {}

double Player::score( Game const& game ) const
{
    return score_function( game.get_state(), weight );
}

namespace tree {

Player::Player( 
    Game const& game, double weight, unsigned depth, Data& data ) 
    : ::minimax::tree::Player< Move, State >( game, depth, data ), weight( weight ) {}

double Player::score( Game const& game ) const
{
    return score_function( game.get_state(), weight );
}

} // namespace tree {
} // namespace minimax {

namespace alphazero {

pair< float, array< float, P > > Player::predict( std::array< float, G > const& game_state_players )
{   
    // This method is now a client of the InferenceManager.
    // It queues a request and blocks until the result is ready.

    // provide the buffer to copy predicted policies into
    array< float, P > policies;

    // --- Batched/Asynchronous implementation (original) ---
    auto future = inference_manager.queue_request( game_state_players.data(), policies.data());
    auto value = future.get(); // blocking call

    // --- Synchronous/Direct implementation (for comparison) ---

    return make_pair(value, policies);
}

size_t Player::move_to_policy_index( Move const& move ) const
{
    return size_t( move.big_move * 9 + move.small_move );
}

array< float, G > Player::serialize_state( Game const& game ) const
{
    auto const& state = game.get_state();
    array< float, G > game_state_players = { 0.0f };

    const PlayerIndex current_player = game.current_player_index();

    // Define pointers to the start of each 81-cell plane for clarity
    float* plane1_x_pieces = game_state_players.data();
    float* plane2_o_pieces = plane1_x_pieces + 81;
    float* plane3_valid_sub_board = plane2_o_pieces + 81;
    float* plane4_player_indicator = plane3_valid_sub_board + 81;

    // --- Plane 1 & 2: 'X' and 'O' pieces (absolute representation) ---
    for (size_t i = 0; i != 9; ++i) 
    {
        auto& small_state = state.small_states[i];
        for (size_t j = 0; j != 9; ++j)             
        {
            if (small_state[j] == ttt::Symbol::Player1) // 'X'
                plane1_x_pieces[i*9+j] = 1.0f;
            else if (small_state[j] == ttt::Symbol::Player2) // 'O'
                plane2_o_pieces[i*9+j] = 1.0f;
        }
    }
    // --- Plane 3: Valid Sub-board ---
    if (state.next_big_move == ttt::no_move) // Can play anywhere
        fill( plane3_valid_sub_board, plane3_valid_sub_board + 81, 1.0f );
    else // Constrained to a single sub-board
    {
        const size_t start_index = state.next_big_move * 9;
        fill( plane3_valid_sub_board + start_index, plane3_valid_sub_board + start_index + 9, 1.0f);
    }

    // --- Plane 4: Player-to-Move Indicator ---
    // A constant plane indicating whose turn it is (1.0 for P1, 0.0 for P2).
    // All input planes must have the same spatial dimensions for the Conv2d layers.
    // This provides global context to the local convolutional kernels, a technique
    // from the AlphaGo Zero paper.
    if (current_player == PlayerIndex::Player1)
        std::fill(plane4_player_indicator, plane4_player_indicator + 81, 1.0f);

    return game_state_players;
}

} // namespace alphazero {

} // namespace uttt

bool next_move( ttt::Move& small_move, ttt::State const& state )
{    
    for (;small_move < 9; ++small_move)
        if (state[small_move] == ttt::Symbol::Empty)
            return true;
    return false;
}

void GameState<uttt::Move, uttt::State>::next_valid_move(
    optional<uttt::Move>& move, PlayerIndex player_index, uttt::State const& state)
{
    if (!move) // reset first move
        move = uttt::Move {
            state.next_big_move != ttt::no_move ? state.next_big_move : ttt::Move {0}, 
            0 };
    else        
        ++move->small_move; // possible next move

    if (state.next_big_move == ttt::no_move) // free big move
    {
        for (;move->big_move < 9; ++move->big_move, move->small_move = 0) 
            if (   state.big_state[move->big_move] == GameResult::Undecided
                && next_move( move->small_move, state.small_states[move->big_move] ))
                return;
    }    
    else if (next_move( move->small_move, state.small_states[state.next_big_move] )) // fixed big move        
        return;
    
    // no valid move found
    move.reset();
}

void append_valid_moves(
    uttt::State const& state, ttt::Move big_move,
    vector< uttt::Move >::iterator& move_itr )
{
    for (ttt::Move small_move = 0; small_move != 9; ++small_move)
        if (state.small_states[big_move][small_move] == ttt::Symbol::Empty)
            *move_itr++ = uttt::Move( big_move, small_move );
}

void GameState< uttt::Move, uttt::State >::get_valid_moves(
    std::vector< uttt::Move >& moves, PlayerIndex,
    uttt::State const& state )
{
    // allocate enough (only the first time actually)
    moves.resize( 81 );
    auto move_itr = moves.begin();

    if (state.next_big_move != ttt::no_move)
        ::append_valid_moves( state, state.next_big_move, move_itr );
    else
        for (ttt::Move big_move = 0; big_move != 9; ++big_move)
            if (state.big_state[big_move] == GameResult::Undecided)
                ::append_valid_moves( state, big_move, move_itr );
                
    // reduce to correct logical size
    moves.resize( move_itr - moves.begin());
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
    if (new_state.big_state[move.big_move] != state.big_state[move.big_move])
        new_state.game_result_cache.reset();
    return new_state;
}

GameResult GameState< uttt::Move, uttt::State >::result( 
    PlayerIndex, uttt::State const& state )
{
    if (state.game_result_cache)
        return *state.game_result_cache;

    // check for wins
    for (const auto& win : ttt::wins)
    {
        const GameResult symbol = state.big_state[win[0]];
        if (   symbol != GameResult::Undecided 
            && symbol != GameResult::Draw 
            && symbol == state.big_state[win[1]] 
            && symbol == state.big_state[win[2]])
        {
            state.game_result_cache =
                (symbol == GameResult::Player1Win) 
                    ? GameResult::Player1Win
                    : GameResult::Player2Win;
            return *state.game_result_cache;
        }
    }

    // check for undecided
    if (std::any_of(state.big_state.begin(), state.big_state.end(), 
        [](GameResult symbol) { return symbol == GameResult::Undecided; }))
    {
        state.game_result_cache = GameResult::Undecided;
        return GameResult::Undecided;    
    }

    // otherwise its a draw
    state.game_result_cache = GameResult::Draw;
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