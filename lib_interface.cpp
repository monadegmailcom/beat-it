#include "games/tic_tac_toe.h" // Includes alphazero.h etc.
#include "libtorch_util.h"

#include <list>

using namespace std;

// static global variables keeps them local to this compile unit

// one global inference manager used by all worker threads,
// gives access to the nn model asynchronously while batching requests
static unique_ptr< libtorch::InferenceManager > inference_manager;

// on cleanup the inference manager is deleted, this is important to avoid issues 
// in the order the multithreading components are teared down
static atomic< bool > cleanup_requested( false );

static vector< future< void > > thread_pool( thread::hardware_concurrency());

// if position queue gets too large threads will be suspended
static atomic< bool > threads_suspended( false );

static mutex position_queue_mutex;
static condition_variable position_queue_cv;
static size_t position_queue_max_size = 10000;
static libtorch::Hyperparameters hyperparameters;
static shared_mutex hp_mutex;

void set_model(
    const char* model_data, int32_t model_data_len,
    const char* metadata_json, int32_t metadata_len,
    size_t state_size, size_t policies_size, function< void () > selfplay_worker )
{
    if (cleanup_requested)
        return;

    auto [model, hp] = libtorch::load_model( 
        model_data, model_data_len, metadata_json, metadata_len );
    if (!inference_manager) // First time call: 
    {
        // create the InferenceManager instance
        inference_manager.reset( new libtorch::InferenceManager(
            std::move( model ), state_size, policies_size ));
        hyperparameters = hp;
        
        // and start worker threads
        cout << "start " << thread_pool.size() << " selfplay worker threads" << endl;
        for (auto& future : thread_pool)
            future = async( selfplay_worker );
    }        
    else // Subsequent calls: update the model in-place for efficiency.
    {
        inference_manager->update_model( std::move( model ));
        lock_guard< shared_mutex > lock( hp_mutex );
        hyperparameters = hp;
    }
}

namespace ttt {

// global position fifo queue, selfplay workers feed new positions into it and client fetches them
// there is a mechanism to stop the position queue to grow infinitly by suspending the workers
// if it gets too large
static queue< alphazero::training::Position > position_queue;

// run self play in worker thread
void selfplay_worker()
{
    // random number generator may not be threadsafe
    mt19937 g { random_device{}() };

    // thread local memory allocator and position buffer avoid synchronization delays 
    alphazero::NodeAllocator node_allocator;
    vector< ::alphazero::training::Position< alphazero::G, alphazero::P > > positions;

    // start with player 1, toggle for each self play run    
    for (PlayerIndex player_index = PlayerIndex::Player1; true; 
         player_index = toggle( player_index ))
    {
        if (threads_suspended)
        {
            cout << "selfplay threads are suspended, waiting..." << endl;
            // block and wait for notification if threads_suspended is true
            threads_suspended.wait( true );
        }
        if (cleanup_requested)
            break;

        shared_lock< shared_mutex > lock( hp_mutex );
        alphazero::libtorch::async::Player player( 
            Game( player_index, empty_state ), 
            hyperparameters.c_base, hyperparameters.c_init, hyperparameters.simulations, 
            node_allocator, *inference_manager );
        positions.clear();
        alphazero::training::SelfPlay self_play(
            player, hyperparameters.dirichlet_alpha, hyperparameters.dirichlet_epsilon,
            hyperparameters.opening_moves, g, positions );
        lock.unlock();

        self_play.run();
        {
            lock_guard< mutex > lock( position_queue_mutex );
            position_queue.push_range( positions );
        }

        position_queue_cv.notify_one();
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

int set_ttt_model( const char* model_data, int32_t model_data_len, const char* metadata_json, int32_t metadata_len )
{
    try 
    {
        set_model( 
            model_data, model_data_len, 
            metadata_json, metadata_len,
            ttt::alphazero::G, ttt::alphazero::P, ttt::selfplay_worker );

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
 copy number_of_positions training data position to the memory locations provided by
 the data_pointers_out struct.
  
 This function is designed to be called from a foreign language interface like Python's ctypes. The model
 is passed as an in-memory buffer.
  
 data_pointers_out A pointer to a struct that will be filled with the addresses of the allocated data buffers.
 returns number of queued position or a negative value on error. */
int fetch_ttt_selfplay_data( DataPointers data_pointers_out, int32_t number_of_positions )
{
    try 
    {
        if (!inference_manager)
            throw runtime_error( "no model loaded" );

        while (!cleanup_requested)
        {
            unique_lock< mutex > lock( position_queue_mutex );

            if (ttt::position_queue.size() < number_of_positions)
                position_queue_cv.wait( lock );

            // check again, may be waked up spurious!
            if (const size_t queue_size = ttt::position_queue.size(); 
                queue_size >= number_of_positions)
            {
                for (size_t i = 0; i < number_of_positions; ++i)
                {
                    auto const& pos = ttt::position_queue.front();
                    ranges::copy( 
                        pos.game_state_players,  
                        data_pointers_out.game_states + i * ttt::alphazero::G );
                    ranges::copy( 
                        pos.target_policy, 
                        data_pointers_out.policy_targets + i * ttt::alphazero::P );
                    data_pointers_out.value_targets[i] = pos.target_value;
                    data_pointers_out.player_indices[i] = 
                        static_cast< int32_t >( pos.current_player );

                    ttt::position_queue.pop();
                }

                // too many queued position updates remaining, stop generating new ones
                if (ttt::position_queue.size() > position_queue_max_size)
                {
                    cout << "suspend selfplay workers" << endl;
                    threads_suspended = true;
                    threads_suspended.notify_all();
                }
                else if (threads_suspended)
                {
                    cout << "resume selfplay workers" << endl;
                    threads_suspended = false;
                    threads_suspended.notify_all();
                }

                return queue_size;
            }
        }

        return 0;
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

int get_inference_histogram(size_t* data_out, int max_size)
{
    try
    {
        if (!inference_manager)
            return 0; // No manager, no data.
        
        // This moves the log data out of the manager.
        std::vector<size_t> log = inference_manager->get_batch_sizes_log();

        if (data_out != nullptr && max_size > 0)
        {
            size_t num_to_copy = std::min((size_t)max_size, log.size());
            std::copy(log.begin(), log.begin() + num_to_copy, data_out);
        }
        
        return static_cast<int>(log.size());
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception in get_inference_histogram: " << e.what() << std::endl;
        return -1;
    }
}

void cleanup_resources()
{
    try
    {
        cleanup_requested = true;        
        threads_suspended = false;
        threads_suspended.notify_all();

        cout << "Wait for all worker threads to finish..." << endl;
        for (auto& future : thread_pool)
            if (future.valid())
                future.wait();

        // Explicitly destroy the inference manager. This is the most critical step,
        // as it ensures the background inference thread is properly joined before
        // the library is unloaded, preventing race conditions during shutdown.
        if (inference_manager)
        {
            cout << "Cleaning up C++ inference manager..." << endl;
            inference_manager.reset();
        }
    }
    catch (const exception& e)
    {
        // Log any errors during cleanup, but don't re-throw.
        cerr << "Exception during C++ resource cleanup: " << e.what() << endl;
    }
    catch (...)
    {
        cerr << "Unknown exception during C++ resource cleanup." << endl;
    }
}

} // extern "C"