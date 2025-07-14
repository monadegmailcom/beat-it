#include "tic_tac_toe.h"
#include "../libtorch_util.h"

#include <iostream>
#include <algorithm>
#include <cmath>

using namespace std;

namespace ttt
{   

const array< Move, 3 > wins[8] =
{ { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 }, // rows
  { 0, 3, 6 }, { 1, 4, 7 }, { 2, 5, 8 }, // columns
  { 0, 4, 8 }, { 2, 4, 6 } }; // diagonals

namespace minimax {
double score( State const& state )
{
    double score = 0.0;
    for (auto const& win : wins)
    {
        uint8_t player1_points = 0;
        uint8_t player2_points = 0;
        for (auto const& index : win)
        {
            if (state[index] == Symbol::Player1)
                ++player1_points;
            else if (state[index] == Symbol::Player2)
                ++player2_points;
        }
        if (player1_points == 0)
            score += player2_points;
        else if (player2_points == 0)
            score -= player1_points;
    }

    return score;
}

} // namespace minimax {

namespace alphazero {

size_t Player::move_to_policy_index( Move const& move ) const
{
    return size_t( move );
}

array< float, G > Player::serialize_state( Game const& game ) const
{
    auto const& state = game.get_state();
    array< float, G > game_state_players = { 0.0f };

    // Pointers to each 9-cell plane for clarity
    float* plane1_x_pieces = game_state_players.data();
    float* plane2_o_pieces = plane1_x_pieces + 9;
    float* plane3_player_indicator = plane2_o_pieces + 9;

    // --- Plane 1 & 2: 'X' and 'O' pieces (absolute representation) ---
    for (size_t i = 0; i < 9; ++i) {
        if (state[i] == Symbol::Player1) { // Player1 is 'X'
            plane1_x_pieces[i] = 1.0f;
        } else if (state[i] == Symbol::Player2) { // Player2 is 'O'
            plane2_o_pieces[i] = 1.0f;
        }
    }

    // --- Plane 3: Player-to-Move Indicator ---
    // A constant plane indicating whose turn it is (1.0 for P1, 0.0 for P2).
    // This provides global context and ensures a consistent input structure for the JIT.
    if (game.current_player_index() == PlayerIndex::Player1)
        std::fill(plane3_player_indicator, plane3_player_indicator + 9, 1.0f);

    return game_state_players;
}

namespace libtorch {
namespace sync {
pair< float, array< float, P > > Player::predict( std::array< float, G > const& game_state_players )
{   
    // provide the buffer to copy predicted policies into
    array< float, P > policies;

    // --- Synchronous/Direct implementation (for comparison) ---
    auto input_tensor = torch::from_blob(
        const_cast<float*>(game_state_players.data()), 
        {1, (long)game_state_players.size()}, torch::kFloat32);

    static auto device = ::libtorch::check_device();

    // Move tensor to the correct device.
    input_tensor = input_tensor.to( device );

    // Run inference.
    torch::jit::IValue output_ivalue = model.forward({input_tensor});
    auto output_tuple = output_ivalue.toTuple();

    // Get results and move them to CPU.
    // The output tensors will have a batch dimension of 1.
    torch::Tensor value_tensor = output_tuple->elements()[0].toTensor().to(torch::kCPU);
    torch::Tensor policy_tensor = output_tuple->elements()[1].toTensor().to(torch::kCPU);

    // Copy policy data to the output buffer.
    float* const policy_ptr = policy_tensor.data_ptr<float>();
    std::copy(policy_ptr, policy_ptr + P, policies.begin());

    const float value = value_tensor[0].item<float>();

    return make_pair( value, policies );
}
} // namespace sync {

namespace async {

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
    // auto value = inference_manager.predict_sync( game_state_players.data(), policies.data());

    return make_pair(value, policies);
}

} // namespace async {
} // namespace libtorch
} // namespace alphazero {

Symbol player_index_to_symbol( PlayerIndex player_index )
{
    const Symbol symbols[] = { Symbol::Player1, Symbol::Player2 };
    return symbols[player_index];
}

namespace console
{

Move HumanPlayer::choose_move()
{
    vector< Move > valid_moves;
    for (auto itr = game.begin(), end = game.end(); itr != end; ++itr)
        valid_moves.push_back( *itr );

    cout << "valid moves:\n";
    for (auto const& move : valid_moves)
        cout << "(" << int( move ) << "), ";
    cout << '\n';

    while (true)
    {
        cout << "\nmove? ";
        unsigned move;
        cin >> move;
        if (move > 8)
        {
            cout << "invalid move" << endl;
            continue;
        }

        if (std::find( valid_moves.begin(), valid_moves.end(), Move( move ))
            == valid_moves.end())
        {
            cout << "not a valid move" << endl;
            continue;
        }

        game = game.apply( Move( move ) );
        return move;
    }
}

void HumanPlayer::apply_opponent_move( Move const& move )
{
    game = game.apply( move );
}

} // namespace console
} // namespace ttt

void GameState< ttt::Move, ttt::State >::next_valid_move( 
    optional< ttt::Move >& move, PlayerIndex, ttt::State const& state )
{
    if (!move)
        move = 0; // first possibly valid move
    else
        ++*move; // next possible move

    while (true)
    {
        if (move >= 9) // no valid move possible anymore
        {
            move.reset();
            break;
        }
        else if (state[*move] == ttt::Symbol::Empty) // move is valid
            break;

        // try next move
        ++*move;
    }
}

void GameState< ttt::Move, ttt::State >::get_valid_moves(
    std::vector< ttt::Move >& moves, PlayerIndex, ttt::State const& state )
{
    // allocate enough (only the first time actually)
    moves.resize( 9 );
    auto move_itr = moves.begin();
    for (char index = 0; index != 9; ++index)
        if (state[index] == ttt::Symbol::Empty)
            *move_itr++ = index;
    // reduce to correct logical size
    moves.resize( move_itr - moves.begin());
}

ttt::State GameState< ttt::Move, ttt::State >::apply( 
    ttt::Move const& move, PlayerIndex player_index, ttt::State const& state )
{
    if (move >= 9)
        throw std::invalid_argument( "invalid move" );
    if (state[move] != ttt::Symbol::Empty)
        throw std::invalid_argument( "invalid move, cell not empty" );

    ttt::State new_state = state;
    new_state[move] = ttt::player_index_to_symbol( player_index );
    return new_state;
}

GameResult GameState< ttt::Move, ttt::State >::result( 
    PlayerIndex player_index, ttt::State const& state )
{
    // check for wins
    for (const auto& win : ttt::wins)
    {
        const ttt::Symbol symbol0 = state[win[0]];
        if (symbol0 != ttt::Symbol::Empty && symbol0 == state[win[1]] && symbol0 == state[win[2]])
            return (symbol0 == ttt::Symbol::Player1) ? GameResult::Player1Win
                                                     : GameResult::Player2Win;
    }

    // check for undecided
    if (std::any_of(state.begin(), state.end(), 
        [](ttt::Symbol symbol) { return symbol == ttt::Symbol::Empty; }))
        return GameResult::Undecided;

    // otherwise its a draw
    return GameResult::Draw;
}    

ostream& operator<<( ostream& os, ttt::Game const& game )
{
    os << "player " << game.current_player_index() + 1 << '\n'
       << "state: " << '\n';
    for (auto i = 0; i != 5; ++i)
        os << '-';
    os << '\n';
    for (size_t index = 0; index != 9; ++index)
    {
        if (index % 3 == 0)
            os << '|';
        os << game.get_state()[index];
        if ((index + 1) % 3 == 0)
            os << "|\n";
    }
    for (auto i = 0; i != 5; ++i)
        os << '-';
    os << '\n';

    return os;
}