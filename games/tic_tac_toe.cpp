#include "tic_tac_toe.h"

#include <torch/script.h> // Main LibTorch header for loading models
#include <torch/torch.h>

#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <queue>
#include <atomic>

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
}

namespace libtorch {

const size_t MAX_BATCH_SIZE = 128;
const std::chrono::milliseconds BATCH_TIMEOUT(5);

using PredictionResult = std::pair<float, std::array<float, P>>;

struct InferenceRequest {
    std::array<float, G> state;
    std::promise<PredictionResult> promise;
};

// This class manages a dedicated thread for running batched model inference.
class InferenceManager {
public:
    InferenceManager(torch::jit::script::Module& module, torch::Device device)
        : module(module), device(device), stop_flag(false) {
        inference_thread = std::thread(&InferenceManager::inference_loop, this);
    }

    ~InferenceManager() {
        stop_flag = true;
        cv.notify_one();
        if (inference_thread.joinable()) {
            inference_thread.join();
        }
    }

    // This is called by worker threads to queue a request for inference.
    std::future<PredictionResult> queue_request(std::array<float, G>&& state) {
        std::promise<PredictionResult> promise;
        auto future = promise.get_future();
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            request_queue.push({std::move(state), std::move(promise)});
        }
        cv.notify_one();
        return future;
    }

private:
    void inference_loop() {
        while (!stop_flag) {
            std::vector<InferenceRequest> batch;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                // Wait until the queue has items or a timeout occurs.
                // The timeout is crucial to process incomplete batches with low latency.
                cv.wait_for(lock, BATCH_TIMEOUT, [this] { return !request_queue.empty() || stop_flag; });

                if (stop_flag) return;

                // Pull requests from the queue to form a batch.
                while (!request_queue.empty() && batch.size() < MAX_BATCH_SIZE) {
                    batch.push_back(std::move(request_queue.front()));
                    request_queue.pop();
                }
            }

            if (batch.empty()) continue;

            // Prepare the batch tensor for the model.
            std::vector<torch::Tensor> batch_tensors;
            batch_tensors.reserve(batch.size());
            for (const auto& req : batch) {
                batch_tensors.push_back(torch::from_blob(
                    const_cast<float*>(req.state.data()), {G}, torch::kFloat32));
            }
            torch::Tensor input_batch = torch::stack(batch_tensors).to(device);

            // Run inference on the entire batch at once.
            torch::jit::IValue output_ivalue = module.forward({input_batch});
            auto output_tuple = output_ivalue.toTuple();
            torch::Tensor value_batch = output_tuple->elements()[0].toTensor().to(torch::kCPU);
            torch::Tensor policy_batch = output_tuple->elements()[1].toTensor().to(torch::kCPU);

            // Distribute the results back to the waiting threads.
            for (size_t i = 0; i < batch.size(); ++i) {
                std::array<float, P> policy_result;
                float* policy_ptr = policy_batch[i].data_ptr<float>();
                std::copy(policy_ptr, policy_ptr + P, policy_result.begin());
                
                float value_result = value_batch[i].item<float>();

                batch[i].promise.set_value({value_result, policy_result});
            }
        }
    }

    torch::jit::script::Module& module;
    torch::Device device;
    std::thread inference_thread;
    std::queue<InferenceRequest> request_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> stop_flag;
};

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
    std::unique_ptr<InferenceManager> inference_manager;
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
    // This method is now a client of the InferenceManager.
    // It queues a request and blocks until the result is ready.
    array< float, G > game_state_players;
    serialize_state( game, game_state_players );

    auto future = impl->inference_manager->queue_request(std::move(game_state_players));
    auto [value, policy_result] = future.get();

    policies = policy_result;
    return value;
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