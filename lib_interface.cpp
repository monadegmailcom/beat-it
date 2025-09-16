#include "games/ultimate_ttt.h" // Includes alphazero.h etc.

#include <list>

using namespace std;

// Encapsulate a queue with its own synchronization primitives to avoid
// cross-talk between different game types (e.g., a uttt worker waking up a
// ttt consumer).
template<typename PositionT>
struct PositionQueue {
    std::queue<PositionT> queue;
    mutable std::mutex mutex;
    std::condition_variable cv;
};

struct SuspensionManager {
    std::mutex mutex;
    std::condition_variable cv;
    bool suspended = false;
};

// The Session struct encapsulates all state for a single training run.
// This avoids global variables and allows for multiple independent sessions.
struct Session {
    unique_ptr<libtorch::InferenceManager> inference_manager;
    atomic<bool> cleanup_requested{false};
    vector<future<void>> thread_pool;
    SuspensionManager suspension_manager;
    size_t position_queue_max_size = 10000;
    libtorch::Hyperparameters hyperparameters;
    shared_mutex hp_mutex;

    // Game-specific position queues
    PositionQueue<ttt::alphazero::training::Position> ttt_position_queue;
    PositionQueue<uttt::alphazero::training::Position> uttt_position_queue;
    
    Statistics root_node_entropy_stat;
};

// A struct to define the layout of the data pointers. This must be mirrored in
//   Python.
struct DataPointers {
    float* game_states = nullptr; // G floats
    float* policy_targets = nullptr; // P floats
    float* value_targets = nullptr; // 1 float
    int32_t* player_indices = nullptr; // 1 int32_t
};

template< typename PlayerT >
using AlphazeroPlayer = ::alphazero::Player<
    typename PlayerT::game_type::move_type,
    typename PlayerT::game_type::state_type,
    PlayerT::game_size,
    PlayerT::policy_size >;

template< typename PlayerT >
using AlphazeroNodeAllocator = alphazero::NodeAllocator<
    typename PlayerT::game_type::move_type,
    typename PlayerT::game_type::state_type >;

template< typename PlayerT >
using AlphazeroPosition = alphazero::training::Position<
    PlayerT::game_size, PlayerT::policy_size >;

void set_model(
    libtorch::DataBuffer model_buffer,
    libtorch::DataBuffer metadata_buffer, size_t state_size,
    size_t policies_size, Session* session,
    function< void( Session* ) > const& worker )
{
    if (session->cleanup_requested)
        return;

    torch::Device device = libtorch::get_device();
    auto [model, hp] =
        libtorch::load_model( model_buffer, metadata_buffer, device );
    if (!session->inference_manager) // First time call:
    {
        // create the InferenceManager instance
        session->inference_manager.reset( new libtorch::InferenceManager(
            std::move( model ), device, state_size, policies_size,
            hp.max_batch_size ));
        session->hyperparameters = hp;

        // Set the thread pool size based on the model's hyperparameters
        session->thread_pool.resize(session->hyperparameters.threads);

        // and start worker threads
        cout << "start " << session->thread_pool.size()
            << " selfplay worker threads" << endl;
        for (auto& future : session->thread_pool) {
            // Capture 'hp' by value [hp] to avoid a dangling reference.
            future = async( [session, worker]() { worker(session); });
        }
    }
    else // Subsequent calls: update the model in-place for efficiency.
    {
        session->inference_manager->update_model( std::move( model ));
        scoped_lock _( session->hp_mutex );
        session->hyperparameters = hp;
    }
}

template< size_t G, size_t P >
size_t fetch_selfplay_data(
    Session* session, DataPointers& data_pointers_out,
    uint32_t number_of_positions,
    PositionQueue< ::alphazero::training::Position< G, P >>& pq )
{
    if (!session->inference_manager)
        throw invalid_argument( "no model loaded" );

    while (!session->cleanup_requested)
    {
        unique_lock< mutex > lock( pq.mutex );

        // Wait until this specific queue has enough positions.
        pq.cv.wait(
            lock,
            [&pq, session, number_of_positions]
            {
                return pq.queue.size() >=
                    number_of_positions || session->cleanup_requested;
            });

        // check again, may be spurious wake up!
        if (const size_t queue_size = pq.queue.size();
            queue_size >= number_of_positions)
        {
            for (size_t i = 0; i < number_of_positions; ++i)
            {
                auto const& pos = pq.queue.front();
                ranges::copy(
                    pos.game_state_players,
                    data_pointers_out.game_states + i * G );
                ranges::copy(
                    pos.target_policy,
                    data_pointers_out.policy_targets + i * P );
                data_pointers_out.value_targets[i] = pos.target_value;
                data_pointers_out.player_indices[i] =
                    static_cast< int32_t >( pos.current_player );

                pq.queue.pop();
            }

            // too many queued position updates remaining,
            //  stop generating new ones
            const bool should_suspend =
                pq.queue.size() > session->position_queue_max_size;
            bool notify = false;
            {
                scoped_lock _(session->suspension_manager.mutex);
                if (session->suspension_manager.suspended != should_suspend)
                {
                    session->suspension_manager.suspended = should_suspend;
                    notify = true;
                    cout << (should_suspend ? "suspend" : "resume")
                         << " selfplay workers" << endl;
                }
            }

            // Notify workers outside the lock to reduce contention.
            if (notify)
                session->suspension_manager.cv.notify_all();

            return queue_size;
        }
    }

    return 0;
}

template< typename PlayerT >
unique_ptr< AlphazeroPlayer< PlayerT >> player_factory(
    Session* session, typename PlayerT::game_type game, unsigned seed,
    AlphazeroNodeAllocator< PlayerT >& node_allocator )
{
    const auto& hp = session->hyperparameters;
    alphazero::params::Ucb ucb_params
        { .c_base = hp.c_base, .c_init = hp.c_init };
    alphazero::params::GamePlay gameplay_params{
        .simulations = hp.simulations,
        .opening_moves = hp.opening_moves,
        .max_batch_size = hp.max_batch_size };

    return make_unique< PlayerT >(
        std::move(game), ucb_params, gameplay_params, seed,
        node_allocator, *session->inference_manager );
}

template< typename PlayerT >
using AlphazeroSelfPlay = ::alphazero::training::SelfPlay<
    typename PlayerT::game_type::move_type,
    typename PlayerT::game_type::state_type,
    PlayerT::game_size,
    PlayerT::policy_size >;


template< typename PlayerT >
unique_ptr< AlphazeroSelfPlay< PlayerT >> selfplay_factory(
    Session* session, AlphazeroPlayer< PlayerT >& player,
    vector< AlphazeroPosition< PlayerT >>& positions,
    Statistics& root_node_entropy_stat,
    mt19937& g ) //NOSONAR
{
    return make_unique< AlphazeroSelfPlay< PlayerT >>(
        player, session->hyperparameters.dirichlet_alpha,
        session->hyperparameters.dirichlet_epsilon, g, positions,
        root_node_entropy_stat );
}

// run self play in worker thread
template< typename PlayerT >
void selfplay_worker( Session* session,
    typename PlayerT::game_type::state_type const& initial_state,
    PositionQueue< AlphazeroPosition< PlayerT >>& pq )
{
    // use some tls resources

    mt19937 g { random_device{}() }; //NOSONAR

    // thread local memory allocator and position buffer avoid synchronization
    //  delays
    AlphazeroNodeAllocator< PlayerT > node_allocator;
    vector< AlphazeroPosition< PlayerT >> positions;

    // start with player 1, toggle for each self play run
    for (PlayerIndex player_index = PlayerIndex::Player1; true;
         player_index = toggle( player_index ))
    {
        {
            unique_lock lock( session->suspension_manager.mutex);
            // The wait predicate handles spurious wake-ups and race conditions.
            session->suspension_manager.cv.wait(
                lock,
                [session]
                {
                    return !session->suspension_manager.suspended
                        || session->cleanup_requested;
                });
        }

        if (session->cleanup_requested)
            break;

        libtorch::Hyperparameters hp;
        {
            scoped_lock _( session->hp_mutex );
            hp = session->hyperparameters;
        }

        auto player =
            player_factory< PlayerT >( session,
                typename PlayerT::game_type( player_index, initial_state ),
                g(), node_allocator );

        positions.clear();
        auto selfplay =
            selfplay_factory< PlayerT >( session, *player, positions, 
            session->root_node_entropy_stat, g );
        selfplay->run();

        {
            scoped_lock _( pq.mutex );
            for (auto const& pos : positions)
                pq.queue.push( pos );
        }

        pq.cv.notify_one();
    }
}

template< typename PlayerT >
uint32_t measure_worker(
    Session* session,
    typename PlayerT::game_type::state_type const& initial_state,
    uint32_t simulations_per_move, uint32_t number_of_games,
    uint32_t number_of_threads_per_selfplay_worker )
{
    // tld
    auto g = mt19937( random_device{}());
    uttt::alphazero::NodeAllocator node_allocator;
    PlayerIndex player_index = PlayerIndex::Player1;
    const auto& hp = session->hyperparameters;
    vector< uttt::alphazero::training::Position > positions;
    uint32_t total_positions = 0;
    const alphazero::params::Ucb ucb_params
        { .c_base = hp.c_base, .c_init = hp.c_init };

    Statistics root_node_entropy_stat;

    for (; number_of_games > 0; --number_of_games)
    {
        alphazero::params::GamePlay gameplay_params{
            .simulations = simulations_per_move,
            .opening_moves = hp.opening_moves,
            .selfplay_threads = number_of_threads_per_selfplay_worker };

        PlayerT player(
            typename PlayerT::game_type( player_index, initial_state ),
            ucb_params, gameplay_params, g(), node_allocator,
            *session->inference_manager );
        alphazero::training::SelfPlay self_play(
            player, hp.dirichlet_alpha, hp.dirichlet_epsilon, g, positions,
            root_node_entropy_stat );
        self_play.run();
        total_positions += positions.size();
        positions.clear();
        player_index = toggle( player_index );
    }
    return total_positions;
}

struct OptimizerParams
{
    uint32_t number_of_selfplay_workers;
    uint32_t number_of_threads_per_selfplay_worker;
    uint32_t max_batch_size;
};

struct FixParams
{
    uint32_t simulations_per_move;
    uint32_t number_of_games;
};

struct CppStats {
    float min;
    float max;
    float mean;
    float stddev;
};


// Use C-style linkage to prevent C++ name mangling, making it callable from
// Python.
extern "C" {

Session* create_session()
{
    try
    {
        return new Session(); //NOSONAR lifetime is managed on caller side
    }
    catch (const std::exception& e)
    {
        cerr << "Failed to create session: " << e.what() << endl;
        return nullptr;
    }
}

void destroy_session(Session* session)
{
    if (!session) return;

    try
    {
        unique_ptr< Session > delete_on_scope_exit;
        delete_on_scope_exit.reset( session );

        cout << "Requesting C++ worker thread cleanup..." << endl;
        session->cleanup_requested = true;

        // Wake up any suspended threads so they can check the cleanup flag.
        {
            scoped_lock _(session->suspension_manager.mutex);
            session->suspension_manager.suspended = false;
        }
        session->suspension_manager.cv.notify_all();

        // Wait for all self-play worker threads to finish.
        session->inference_manager.reset();
        cout << "Waiting for self-play worker threads to join..." << endl;
        for (auto const& future : session->thread_pool)
        {
            if (future.valid()) {
                future.wait();
            }
        }
        cout << "All self-play worker threads joined." << endl;
    }
    catch (const std::exception& e)
    {
        cerr << "Exception during C++ resource cleanup: " << e.what() << endl;
    }

    cout << "C++ session cleanup complete." << endl;
}

// return number of created positions
uint32_t measure_uttt_selfplay_throughput(
    Session* session, FixParams const& fix_params,
    OptimizerParams const& opt_params )
{
    try
    {
        if (!session)
            throw invalid_argument( "session is null" );
        if (!opt_params.number_of_selfplay_workers)
            throw invalid_argument( "number_of_selfplay_workers is zero" );

        vector< future< uint32_t > > thread_pool(
            opt_params.number_of_selfplay_workers );

        uint32_t number_of_games_per_worker =
            fix_params.number_of_games
            + opt_params.number_of_selfplay_workers - 1;
        number_of_games_per_worker /= opt_params.number_of_selfplay_workers;

        // start parallel workers
        for (auto& future : thread_pool)
            future = async(
                measure_worker< uttt::alphazero::libtorch::async::Player >,
                session, uttt::empty_state, fix_params.simulations_per_move,
                number_of_games_per_worker,
                opt_params.number_of_threads_per_selfplay_worker );

        // collect all results
        uint32_t total_positions = 0;
        for (auto& future : thread_pool)
            total_positions += future.get();

        return total_positions;
    }
    catch (exception const& e)
    {
        cerr << "C++ Exception caught: " << e.what() << endl;
        return -1;
    }
}

int set_ttt_model(
    Session* session, const char* model_data, int32_t model_data_len,
    const char* metadata_json, int32_t metadata_len )
{
    if (!session) return -1;
    try
    {
        set_model(
            {model_data, model_data_len}, {metadata_json, metadata_len},
            ttt::alphazero::G, ttt::alphazero::P, session, [](Session* s) {
                selfplay_worker<ttt::alphazero::libtorch::async::Player>(
                    s, ttt::empty_state, s->ttt_position_queue );
            } );

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

int set_uttt_model(
    Session* session, const char* model_data, int32_t model_data_len,
    const char* metadata_json, int32_t metadata_len )
{
    if (!session) return -1;
    try
    {
        set_model(
            {model_data, model_data_len}, {metadata_json, metadata_len},
            uttt::alphazero::G, uttt::alphazero::P, session, [](Session* s) {
                selfplay_worker<uttt::alphazero::libtorch::async::Player>(
                    s, uttt::empty_state, s->uttt_position_queue );
            } );

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
copy number_of_positions training data position to the memory locations
    provided by the data_pointers_out struct.
returns number of queued positions or a negative value on error. */
int fetch_ttt_selfplay_data(
    Session* session, DataPointers& data_pointers_out,
    uint32_t number_of_positions )
{
    if (!session) return -1;
    try
    {
        return static_cast< int >( fetch_selfplay_data(
            session, data_pointers_out, number_of_positions,
            session->ttt_position_queue ));
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

int fetch_uttt_selfplay_data(
    Session* session, DataPointers& data_pointers_out,
    uint32_t number_of_positions )
{
    if (!session) return -1;
    try
    {
        return static_cast< int > (fetch_selfplay_data(
            session, data_pointers_out, number_of_positions,
            session->uttt_position_queue ));
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

CppStats get_inference_queue_stats(Session* session)
{
    if (!session || !session->inference_manager)
        return {0.0f, 0.0f, 0.0f, 0.0f};

    try
    {
        auto const& stats = session->inference_manager->queue_size_stats();
        CppStats result = {
            stats.min(), stats.max(), stats.mean(), stats.stddev()};

        session->inference_manager->reset_stats();

        return result;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception in get_inference_queue_stats: " << e.what()
            << std::endl;
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }
}

} // extern "C"