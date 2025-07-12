#include "games/tic_tac_toe.h" // Includes alphazero.h etc.
#include "libtorch_util.h"

#include <list>

using namespace std;

// static global variables keeps them local to this compile unit

// A static buffer to hold the self-play data. This avoids repeated allocations/deallocations
// the selfplay_data_buffer has a different memory layout than the 
// incrementally generated positions. so we need a copy:
// four arrays with positions.size() entries:
// 1. games states, 2.  policies targets, 3. value targets, 4: player indices
static vector< uint8_t > selfplay_data_buffer;

namespace ttt {

struct ThreadLocalStorage
{
    // This struct is not copyable because each instance owns unique resources
    // (random generator, memory pool) that should not be shared across threads
    // or copied unintentionally.
    ThreadLocalStorage() = default;
    ThreadLocalStorage(const ThreadLocalStorage&) = delete;
    ThreadLocalStorage& operator=(const ThreadLocalStorage&) = delete;
    ThreadLocalStorage(ThreadLocalStorage&&) = default;
    ThreadLocalStorage& operator=(ThreadLocalStorage&&) = default;

    // Each thread needs its own random number generator and memory pool allocator.
    mt19937 g{random_device{}()};
    vector< ::alphazero::training::Position< alphazero::G, alphazero::P > > positions;
    alphazero::NodeAllocator node_allocator;
    future< void > selfplay_future;   
};

static unique_ptr< libtorch::InferenceManager > inference_manager;
// use list to prevent reallocations when appending
static list< ThreadLocalStorage > thread_local_storage_list;
static const size_t data_pointers_size = (alphazero::G + alphazero::P + 1) * sizeof( float ) + sizeof( int32_t );

// run self play in worker thread
void selfplay_worker( 
    ThreadLocalStorage& local_storage,
    libtorch::InferenceManager& inference_manager,
    size_t runs_per_thread,
    float c_base,
    float c_init,
    float dirichlet_alpha,
    float dirichlet_epsilon,
    int32_t simulations,
    int32_t opening_moves )
{
    PlayerIndex player_index = PlayerIndex::Player1;
    local_storage.positions.clear();
    for (; runs_per_thread; --runs_per_thread) 
    {
        alphazero::libtorch::Player player( Game( player_index, empty_state ), c_base, c_init,
            simulations, local_storage.node_allocator, inference_manager );
        alphazero::training::SelfPlay self_play(
            player, dirichlet_alpha, dirichlet_epsilon,
            opening_moves, local_storage.g, local_storage.positions );
        self_play.run();
        player_index = toggle(player_index);
    }
}

} // namespace ttt {

// A struct to define the layout of the data pointers. This must be mirrored in Python.
struct DataPointers {
    float* game_states = nullptr; // G floats
    float* policy_targets = nullptr; // P floats
    float* value_targets = nullptr; // 1 float
    int32_t* player_indices = nullptr; // 1 int32_t
};

// Use C-style linkage to prevent C++ name mangling, making it callable from Python.
extern "C" {

int set_ttt_model( const char* model_data, int32_t model_data_len)
{
    try 
    {
        std::string content( model_data, model_data_len );

        ttt::inference_manager.reset( new libtorch::InferenceManager( 
            std::move( content ), ttt::alphazero::G, ttt::alphazero::P ));

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
 Runs some self-play games for Tic-Tac-Toe with some worker threads. 
 Allocates memory for the training data and provides pointers to it via 
 the data_pointers_out struct. The memory is managed by the library.
  
 This function is designed to be called from a foreign language interface like Python's ctypes. The model
 is passed as an in-memory buffer.
  
 data_pointers_out A pointer to a struct that will be filled with the addresses of the allocated data buffers.
 returns the number of game positions actually generated. Returns a negative value on error.
 */
int run_ttt_selfplay(
    int32_t threads, // number of worker threads
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
        if (!threads)
            throw invalid_argument( "threads must be > 0" );
             
        const auto [runs_per_thread, rem_runs] = ldiv( runs, threads );

        //const unsigned int num_threads = std::thread::hardware_concurrency();
        if (threads > ttt::thread_local_storage_list.size())
            ttt::thread_local_storage_list.resize( threads );

        // add remainder runs to the first one, so we add up to original number of runs
        size_t rpt = runs_per_thread + rem_runs;

        if (!ttt::inference_manager)
            throw runtime_error( "no model loaded" );
        
        for (auto itr = ttt::thread_local_storage_list.begin(); 
             itr != ttt::thread_local_storage_list.end(); ++itr) 
        {
            itr->selfplay_future = async( ttt::selfplay_worker,
                ref(*itr), ref(*ttt::inference_manager), rpt, c_base, c_init,
                dirichlet_alpha, dirichlet_epsilon, simulations, opening_moves );
            rpt = runs_per_thread;
        }

        // wait until all worker threads are done and accumulate total positions
        size_t total_positions = 0;
        for (auto& tls : ttt::thread_local_storage_list) 
        {
            tls.selfplay_future.wait();
            total_positions += tls.positions.size();
        }

        selfplay_data_buffer.resize( total_positions * ttt::data_pointers_size);

        // Get pointers to the start of each section in the buffer
        float* current_ptr = reinterpret_cast< float*>( selfplay_data_buffer.data());

        data_pointers_out->game_states = current_ptr;
        current_ptr += total_positions * ttt::alphazero::G;

        data_pointers_out->policy_targets = current_ptr;
        current_ptr += total_positions * ttt::alphazero::P;

        data_pointers_out->value_targets = current_ptr;
        current_ptr += total_positions;

        data_pointers_out->player_indices = reinterpret_cast< int32_t* >( current_ptr );

        // Copy the generated data into the contiguous out buffer
        size_t current_pos_idx = 0;
        for (auto& tls : ttt::thread_local_storage_list)
            for (const auto& pos : tls.positions)
            {
                ranges::copy( 
                    pos.game_state_players, 
                    data_pointers_out->game_states + current_pos_idx * ttt::alphazero::G);
                ranges::copy( 
                    pos.target_policy, 
                    data_pointers_out->policy_targets + current_pos_idx * ttt::alphazero::P);
                data_pointers_out->value_targets[current_pos_idx] = pos.target_value;
                data_pointers_out->player_indices[current_pos_idx] = 
                    static_cast< int32_t >( pos.current_player );
                ++current_pos_idx;
            }

        return static_cast<int>( total_positions );
    } 
    catch (const exception& e) 
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