#include "games/tic_tac_toe.h" // Includes alphazero.h etc.
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdint>

using namespace std;

// static global variables keeps them local to this compile unit

// A static buffer to hold the self-play data. This avoids repeated allocations/deallocations
static vector< uint8_t > selfplay_data_buffer;

static mt19937 g( random_device{}());
static ttt::alphazero::NodeAllocator node_allocator;

// the selfplay_data_buffer has a different memory layout than the 
// incrementally generated positions. so we need another variable.
static vector< ttt::alphazero::training::Position > positions;

// A struct to define the layout of the data pointers. This must be mirrored in Python.
struct DataPointers {
    float* game_states = nullptr; // G floats
    float* policy_targets= nullptr; // P floats
    float* value_targets= nullptr; // 1 float
    int32_t* player_indices= nullptr; // 1 int32_t

    static const size_t size = 
        (ttt::alphazero::G + ttt::alphazero::P + 1) * sizeof( float ) + sizeof( int32_t );
};

// Use C-style linkage to prevent C++ name mangling, making it callable from Python.
extern "C" {

/**
 * @brief Runs one self-play game for Tic-Tac-Toe. Allocates memory for the training data
 *        and provides pointers to it via the data_pointers_out struct. The memory is managed
 *        by the library.
 * 
 * This function is designed to be called from a foreign language interface like Python's ctypes.
 * 
 * @param model_path Path to the TorchScript model file.
 * @param data_pointers_out A pointer to a struct that will be filled with the addresses of the allocated data buffers.
 * @return The number of game positions actually generated. Returns a negative value on error.
 */
int run_ttt_selfplay(
    const char* model_path,
    int8_t current_player, // 0: player 1, 1: player 2
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
        // Use aliases for Tic-Tac-Toe specific types for clarity
        using Game = ttt::Game;
        using SelfPlay = ttt::alphazero::training::SelfPlay;
        using Data = ttt::alphazero::libtorch::Data;

        // 1. Setup the game and data structures
        Data data(g, node_allocator, model_path);

        Game initial_game( 
            static_cast< PlayerIndex >( current_player ), ttt::empty_state );

        positions.clear();

        // 2. Run self-play to generate training data
        SelfPlay self_play(
            initial_game, c_base, c_init, dirichlet_alpha, dirichlet_epsilon, simulations, 
            opening_moves, data, positions );
        self_play.run();

        // 3. Calculate memory layout and resize the static buffer
        const size_t num_positions = positions.size();

        selfplay_data_buffer.resize( num_positions * DataPointers::size);

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
            const auto& pos = positions[i];
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