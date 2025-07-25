#include "games/ultimate_ttt.h" // Includes alphazero.h etc.

#include <list>

using namespace std;

// static global variables keeps them local to this compile unit

// one global inference manager used by all worker threads,
// gives access to the nn model asynchronously while batching requests
static unique_ptr< libtorch::InferenceManager > inference_manager;

// on cleanup the inference manager is deleted, this is important to avoid issues
// in the order the multithreading components are teared down
static atomic< bool > cleanup_requested( false );

static vector< future< void > > thread_pool;

// if position queue gets too large threads will be suspended
static atomic< bool > threads_suspended( false );

static mutex position_queue_mutex;
static condition_variable position_queue_cv;
static size_t position_queue_max_size = 10000;
static libtorch::Hyperparameters hyperparameters;
static shared_mutex hp_mutex;

// A struct to define the layout of the data pointers. This must be mirrored in Python.
struct DataPointers {
    float* game_states = nullptr; // G floats
    float* policy_targets = nullptr; // P floats
    float* value_targets = nullptr; // 1 float
    int32_t* player_indices = nullptr; // 1 int32_t
};

void set_model(
    const char* model_data, int32_t model_data_len,
    const char* metadata_json, int32_t metadata_len,
    size_t state_size, size_t policies_size, function< void () > worker )
{
    if (cleanup_requested)
        return;

    auto [model, hp] = libtorch::load_model(
        model_data, model_data_len, metadata_json, metadata_len );
    if (!inference_manager) // First time call:
    {
        // create the InferenceManager instance
        inference_manager.reset( new libtorch::InferenceManager(
            std::move( model ), hp, state_size, policies_size ));
        hyperparameters = hp;

        // Set the thread pool size based on the model's hyperparameters
        thread_pool.resize(hyperparameters.threads);

        // and start worker threads
        cout << "start " << thread_pool.size() << " selfplay worker threads" << endl;
        for (auto& future : thread_pool)
            future = async( worker );
    }
    else // Subsequent calls: update the model in-place for efficiency.
    {
        inference_manager->update_model( std::move( model ), hp);
        lock_guard< shared_mutex > lock( hp_mutex );
        hyperparameters = hp;
    }
}

template< typename MoveT, typename StateT, size_t G, size_t P >
using AlphazeroPlayerFactory = alphazero::Player< MoveT, StateT, G, P >* (*)(
    Game< MoveT, StateT > const&,
    libtorch::Hyperparameters const&,
    alphazero::NodeAllocator< MoveT, StateT >&);

template< typename MoveT, typename StateT, size_t G, size_t P >
using SelfPlayFactory = alphazero::training::SelfPlay<MoveT, StateT, G, P>* (*)(
    alphazero::Player<MoveT, StateT, G, P>&, std::vector<alphazero::training::Position<G, P>>&, std::mt19937&);

// run self play in worker thread
template< typename MoveT, typename StateT, size_t G, size_t P >
void selfplay_worker(
    AlphazeroPlayerFactory< MoveT, StateT, G, P > player_factory,
    SelfPlayFactory< MoveT, StateT, G, P > selfplay_factory,
    StateT const& initial_state,
    queue< alphazero::training::Position< G, P >>& position_queue )
{
    // random number generator may not be threadsafe
    mt19937 g { random_device{}() };

    // thread local memory allocator and position buffer avoid synchronization delays
    alphazero::NodeAllocator< MoveT, StateT > node_allocator;
    vector< alphazero::training::Position< G, P > > positions;

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

        libtorch::Hyperparameters hp;
        {
            shared_lock< shared_mutex > lock( hp_mutex );
            hp = hyperparameters;
        }

        unique_ptr< alphazero::Player< MoveT, StateT, G, P > > player(
            player_factory( Game< MoveT, StateT >( player_index, initial_state ),
                            hp, node_allocator ));

        positions.clear();
        unique_ptr< alphazero::training::SelfPlay< MoveT, StateT, G, P > > selfplay(
            selfplay_factory( *player, positions, g ));
        selfplay->run();

        {
            lock_guard< mutex > lock( position_queue_mutex );
            position_queue.push_range( positions );
        }

        position_queue_cv.notify_one();
    }
}

template< size_t G, size_t P >
int fetch_selfplay_data(
    DataPointers data_pointers_out, int32_t number_of_positions,
    queue< ::alphazero::training::Position< G, P >>& position_queue )
{
    if (!inference_manager)
        throw runtime_error( "no model loaded" );

    while (!cleanup_requested)
    {
        unique_lock< mutex > lock( position_queue_mutex );

        if (position_queue.size() < number_of_positions)
            position_queue_cv.wait( lock );

        // check again, may be waked up spurious!
        if (const size_t queue_size = position_queue.size();
            queue_size >= number_of_positions)
        {
            for (size_t i = 0; i < number_of_positions; ++i)
            {
                auto const& pos = position_queue.front();
                ranges::copy(
                    pos.game_state_players,
                    data_pointers_out.game_states + i * G );
                ranges::copy(
                    pos.target_policy,
                    data_pointers_out.policy_targets + i * P );
                data_pointers_out.value_targets[i] = pos.target_value;
                data_pointers_out.player_indices[i] =
                    static_cast< int32_t >( pos.current_player );

                position_queue.pop();
            }

            // too many queued position updates remaining, stop generating new ones
            if (position_queue.size() > position_queue_max_size)
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

template< typename PlayerT >
using AlphazeroPlayer = ::alphazero::Player<
    typename PlayerT::game_type::move_type,
    typename PlayerT::game_type::state_type,
    PlayerT::game_size,
    PlayerT::policy_size >;

template< typename PlayerT >
AlphazeroPlayer< PlayerT >* player_factory(
    typename PlayerT::game_type const& game,
    libtorch::Hyperparameters const& hp,
    ::alphazero::NodeAllocator<
        typename PlayerT::game_type::move_type,
        typename PlayerT::game_type::state_type > & node_allocator )
{
    return new PlayerT(
        game, hp.c_base, hp.c_init, hp.simulations,
        node_allocator, *inference_manager );
}

template< typename PlayerT >
using AlphazeroSelfPlay = ::alphazero::training::SelfPlay<
    typename PlayerT::game_type::move_type,
    typename PlayerT::game_type::state_type,
    PlayerT::game_size,
    PlayerT::policy_size >;

template< typename PlayerT >
AlphazeroSelfPlay< PlayerT >* selfplay_factory(
    AlphazeroPlayer< PlayerT >& player,
    vector< ::alphazero::training::Position< PlayerT::game_size, PlayerT::policy_size >>& positions,
     mt19937& g )
{
    return new AlphazeroSelfPlay< PlayerT >(
        player, hyperparameters.dirichlet_alpha, hyperparameters.dirichlet_epsilon,
        hyperparameters.opening_moves, g, positions );
}

namespace ttt {

// global position fifo queue, selfplay workers feed new positions into it and client fetches them
// there is a mechanism to stop the position queue to grow infinitly by suspending the workers
// if it gets too large
static queue< alphazero::training::Position > position_queue;

} // namespace ttt {

namespace uttt {
static queue< alphazero::training::Position > position_queue;
} // namespace uttt {

// Use C-style linkage to prevent C++ name mangling, making it callable from Python.
extern "C" {

int set_ttt_model( const char* model_data, int32_t model_data_len, const char* metadata_json, int32_t metadata_len )
{
    try
    {
        set_model(
            model_data, model_data_len,
            metadata_json, metadata_len,
            ttt::alphazero::G, ttt::alphazero::P,
            []()
            {
                selfplay_worker(
                    player_factory< ttt::alphazero::libtorch::async::Player >,
                    selfplay_factory< ttt::alphazero::libtorch::async::Player >,
                    ttt::empty_state,
                    ttt::position_queue);
            });

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

int set_uttt_model( const char* model_data, int32_t model_data_len, const char* metadata_json, int32_t metadata_len )
{
    try
    {
        set_model(
            model_data, model_data_len,
            metadata_json, metadata_len,
            uttt::alphazero::G, uttt::alphazero::P,
            []()
            {
                selfplay_worker(
                    player_factory< uttt::alphazero::libtorch::async::Player >,
                    selfplay_factory< uttt::alphazero::libtorch::async::Player >,
                    uttt::empty_state,
                    uttt::position_queue);
            });

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
        return fetch_selfplay_data( data_pointers_out, number_of_positions, ttt::position_queue );
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

int fetch_uttt_selfplay_data( DataPointers data_pointers_out, int32_t number_of_positions )
{
    try
    {
        return fetch_selfplay_data( data_pointers_out, number_of_positions, uttt::position_queue );
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

        // Get a reference to the histogram data.
        vector<size_t> const& histogram = inference_manager->get_inference_histogram();

        if (data_out != nullptr && max_size > 0)
        {
            size_t num_to_copy = std::min((size_t)max_size, histogram.size());
            std::copy(histogram.begin(), histogram.begin() + num_to_copy, data_out);
        }

        return static_cast<int>(histogram.size());
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