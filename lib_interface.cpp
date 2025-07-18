#include "games/tic_tac_toe.h" // Includes alphazero.h etc.
#include <vector>
#include <iostream>
#include <algorithm>
#include <memory>
#include <cstdint>

#include <torch/script.h> // Main LibTorch header for loading models
#include <torch/torch.h>

using namespace std;

// static global variables keeps them local to this compile unit

// A static buffer to hold the self-play data. This avoids repeated allocations/deallocations
// the selfplay_data_buffer has a different memory layout than the 
// incrementally generated positions. so we need a copy:
// four arrays with positions.size() entries:
// 1. games states, 2.  policies targets, 3. value targets, 4: player indices
static vector< uint8_t > selfplay_data_buffer;

static mt19937 g( random_device{}());

namespace ttt {

static alphazero::NodeAllocator node_allocator;
static unique_ptr< alphazero::libtorch::Data > libtorch_data;
static vector< alphazero::training::Position > positions;
static const size_t data_pointers_size = 
    (alphazero::G + alphazero::P + 1) * sizeof( float ) + sizeof( int32_t );

} // namespace ttt {

// A struct to define the layout of the data pointers. This must be mirrored in Python.
struct DataPointers {
    float* game_states = nullptr; // G floats
    float* policy_targets = nullptr; // P floats
    float* value_targets = nullptr; // 1 float
    int32_t* player_indices = nullptr; // 1 int32_t
};

void check_cuda()
{
    if (torch::cuda::is_available()) 
        cout << "CUDA is available! Moving model to GPU." << endl;
    else 
        cout << "CUDA not available. Using CPU." << endl;
}

// Use C-style linkage to prevent C++ name mangling, making it callable from Python.
extern "C" {

int set_ttt_model( const char* model_data, int32_t model_data_len)
{
    try 
    {
        check_cuda();
        ttt::libtorch_data.reset( new ttt::alphazero::libtorch::Data( 
            g, ttt::node_allocator, model_data, model_data_len ));
        return 0;
    } 
    catch (exception const& e) 
    {
        cerr << "C++ Exception caught: " << e.what() << endl;
        return -1;
    } 
    catch (...) 
    {
        cerr << "C++ Unknown exception caught." << endl;
        return -2;
    }
}

/*
 Runs one self-play game for Tic-Tac-Toe. Allocates memory for the training data
 and provides pointers to it via the data_pointers_out struct. The memory is managed
 by the library.
  
 This function is designed to be called from a foreign language interface like Python's ctypes. The model
 is passed as an in-memory buffer.
  
 data_pointers_out A pointer to a struct that will be filled with the addresses of the allocated data buffers.
 returns the number of game positions actually generated. Returns a negative value on error.
 */
int run_ttt_selfplay(
    int32_t runs, // number of runs
    float c_base, // 19652
    float c_init, // 1.25
    float dirichlet_alpha, // 0.3
    float dirichlet_epsilon, // 0.25
    int32_t simulations, // 100
    int32_t opening_moves, // 1    
    DataPointers* data_pointers_out ) 
{
    try 
    {
        if (!ttt::libtorch_data)
            throw runtime_error( "no model loaded" );

        ttt::positions.clear();
        PlayerIndex player_index = Player1;

        for (;runs;--runs)
        {
            // 2. Run self-play to generate training data
            ttt::alphazero::training::SelfPlay self_play(
                ttt::Game( player_index, ttt::empty_state ), c_base, c_init, dirichlet_alpha, dirichlet_epsilon, 
                simulations, opening_moves, *ttt::libtorch_data, ttt::positions );
            self_play.run();
            player_index = toggle( player_index );
        }

        // 3. Calculate memory layout and resize the static buffer
        const size_t num_positions = ttt::positions.size();

        selfplay_data_buffer.resize( num_positions * ttt::data_pointers_size);

        // 4. Get pointers to the start of each section in the buffer
        float* current_ptr = reinterpret_cast< float*>( selfplay_data_buffer.data());

        data_pointers_out->game_states = current_ptr;
        current_ptr += num_positions * ttt::alphazero::G;

        data_pointers_out->policy_targets = current_ptr;
        current_ptr += num_positions * ttt::alphazero::P;

        data_pointers_out->value_targets = current_ptr;
        current_ptr += num_positions;

        data_pointers_out->player_indices = reinterpret_cast< int32_t* >( current_ptr );

        // 5. Copy the generated data into the contiguous out buffer
        for (size_t i = 0; i < num_positions; ++i) 
        {
            const auto& pos = ttt::positions[i];
            ranges::copy( pos.game_state_players, data_pointers_out->game_states + i * ttt::alphazero::G);
            ranges::copy( pos.target_policy, data_pointers_out->policy_targets + i * ttt::alphazero::P);
            data_pointers_out->value_targets[i] = pos.target_value;
            data_pointers_out->player_indices[i] = static_cast< int32_t >( pos.current_player );
        }

        return static_cast<int>( num_positions );
    } 
    catch (const std::exception& e) 
    {
        cerr << "C++ Exception caught: " << e.what() << endl;
        return -1;
    } 
    catch (...) 
    {
        cerr << "C++ Unknown exception caught." << endl;
        return -2;
    }
}

} // extern "C"