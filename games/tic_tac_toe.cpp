#include "tic_tac_toe.h"

#include <torch/script.h> // Main LibTorch header for loading models
#include <torch/torch.h>

#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>

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

float Data::predict( Game const& game, array< float, P >& policies )
{
    // for debug purposes use score function for prediction

    // initialize policies to 0.0
    policies.fill ( 0.0f );

    // transform scores to increasing values from worst to best
    const float f = game.current_player_index() == PlayerIndex::Player1 
        ? -1.0f
        : 1.0f;
    for (auto move : game)
        policies[move_to_policy_index( move )] = f * static_cast< float >( minimax::score( 
            game.apply( move ).get_state()));

    // transform score to target value from -1 (loss) to 1 (win)
    return tanh( f * static_cast< float >( minimax::score( game.get_state())));
}

size_t Data::move_to_policy_index( Move const& move ) const
{
    return size_t( move );
}

void Data::serialize_state( 
    Game const& game,
    array< float, G >& game_state_players ) const
{
    auto const& state = game.get_state();

    game_state_players.fill( 0.0f );
    const ttt::Symbol player_symbol = player_index_to_symbol( 
        game.current_player_index());
    const ttt::Symbol opponent_symbol = player_index_to_symbol( 
        toggle( game.current_player_index()));
    for (size_t i = 0; i != 9; ++i)
        if (state[i] == player_symbol)
            game_state_players[i] = 1.0;    
        else if (state[i] == opponent_symbol)
            game_state_players[i + 9] = 1.0f;    
}

namespace libtorch {

struct Impl
{
    Impl( const char* model_data, size_t model_data_len )
        : model_data_stream( std::string( model_data, model_data_len)),
          module( torch::jit::load( model_data_stream ))
    { init(); }

    Impl( const std::string& model_path )
        : module( torch::jit::load( model_path ))
    { init(); }

    void init()
    {
        module.eval();

        if (torch::cuda::is_available()) 
        {
            device = torch::kCUDA;
            module.to( torch::kCUDA );
        } 
        else 
            device = torch::kCPU;
    }


    istringstream model_data_stream;
    torch::jit::script::Module module; // The loaded TorchScript model
    torch::Device device = torch::kCPU;  // Device to run inference on (CPU or CUDA)
};

Data::Data( mt19937& g, NodeAllocator& allocator, const std::string& model_path )
    : ttt::alphazero::Data( g, allocator ),
      impl( make_unique< Impl >( model_path ))
{}

Data::Data( mt19937& g, NodeAllocator& allocator, const char* model_data, size_t model_data_len )
    : ttt::alphazero::Data( g, allocator ),
      impl( make_unique< Impl >(model_data, model_data_len))
{}

Data::~Data() = default;
Data::Data(Data&&) = default;

float Data::predict( 
    Game const& game, 
    std::array< float, P >& policies )
{   
    array< float, G > game_state_players;
    serialize_state( game, game_state_players );

    torch::Tensor input_tensor = torch::from_blob(
            game_state_players.data(),
            {1, static_cast<long>(G)}, // Shape: [Batch=1, Features=G]
            torch::kFloat32 // Assuming float32 input
        ).to( impl->device ); // Move to the appropriate device (CPU/GPU)

    torch::jit::IValue output_ivalue = impl->module.forward( { input_tensor } );
    auto output_tuple = output_ivalue.toTuple();
    
    // extract policies
    torch::Tensor policy_logits_tensor =
        output_tuple->elements()[1].toTensor().to( torch::kCPU ); // Move to CPU for access
    if (policy_logits_tensor.numel() != P)
        throw runtime_error( "policy tensor size mismatch" );
    policy_logits_tensor = policy_logits_tensor.contiguous();
    float* const policy_data_ptr = policy_logits_tensor.data_ptr<float>();
    copy( policy_data_ptr, policy_data_ptr + P, policies.begin());
          
    // extract target value
    torch::Tensor value_tensor =
        output_tuple->elements()[0].toTensor().to( torch::kCPU ); // Move to CPU for access
    if (value_tensor.numel() != 1) 
        throw runtime_error( "value tensor is not a scalar" );
    return value_tensor.item< float >();
}

} // namespace libtorch

namespace training {

} // namespace training {
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

        if (!ranges::contains( valid_moves.begin(), valid_moves.end(), Move( move )))
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