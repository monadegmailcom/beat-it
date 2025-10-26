#include "games/ultimate_ttt.h" // Includes alphazero.h etc.

using namespace std;

// A struct to define the layout of the data pointers. This must be mirrored in
//   Python.
struct DataPointers {
    float* game_states = nullptr; // G floats
    float* policy_targets = nullptr; // P floats
    float* value_targets = nullptr; // 1 float
    int32_t* player_indices = nullptr; // 1 int32_t
};

// The Session struct encapsulates all state for a single training run.
// This avoids global variables and allows for multiple independent sessions.
template< typename PlayerT >
struct Session 
{
    using game_type = PlayerT::game_type;
    using move_type = game_type::move_type;
    using state_type = game_type::state_type;
    using payload_type = PlayerT::payload_type;
    using node_type = Node< move_type, state_type, payload_type >;
    using allocator_type = GenerationalArenaAllocator;
    using inference_service_type = 
        libtorch::InferenceService< PlayerT::game_size, PlayerT::policy_size >;
    using request_type = inference_service_type::request_type;
    using position_type = alphazero::training::Position< 
        PlayerT::game_size, PlayerT::policy_size >;
    using player_type = PlayerT;
    using selfplay_type = alphazero::training::SelfPlay< 
        typename game_type::move_type, state_type, PlayerT::game_size, 
        PlayerT::policy_size >;

    Session( 
        state_type const& initial_state, 
        std::unique_ptr< torch::jit::script::Module >&& model,
        libtorch::Hyperparameters const& hp )
    : queue( hp.max_batch_size ), hp( hp ), initial_state( initial_state ),
      inference_service( 
        std::move( model ), libtorch::get_device(), hp.max_batch_size)
    {
        thread_pool.resize( hp.parallel_games );
        cout << "start " << thread_pool.size()
            << " parallel games" << endl;
        for (auto& future : thread_pool) 
            future = async( &Session::worker, this );
    } 

    void set_model( libtorch::DataBuffer model_buffer )
    {
        if (cleanup_requested)
            return;

        torch::Device device = libtorch::get_device();
        auto model = libtorch::load_model( model_buffer, device );

        inference_service.update_model( std::move( model ));
    }

    void fetch_selfplay_data(
        DataPointers& data_pointers_out, uint32_t number_of_positions )
    {
        position_type pos;
        for (size_t i = 0; i < number_of_positions; ++i)
        {
            // pop next position from queue, enter blocking wait if queue is 
            // empty.
            while (!cleanup_requested && !queue.pop( pos ))
            {
                // notify workers.
                queue_cv.notify_all();

                unique_lock lock( queue_mutex );
                queue_cv.wait(
                    lock,
                    [this] { return !queue.empty() || cleanup_requested; });
            }             
 
            // copy position into buffer.
            ranges::copy(
                pos.game_state,
                data_pointers_out.game_states + i * PlayerT::game_size );
            ranges::copy(
                pos.target_policy,
                data_pointers_out.policy_targets + i * PlayerT::policy_size );
            data_pointers_out.value_targets[i] = pos.target_value;
            data_pointers_out.player_indices[i] =
                static_cast< int32_t >( pos.current_player );
        }

        // notify workers.
        queue_cv.notify_all();
    }

    void worker()
    {
        // use some tls resources
        mt19937 g { random_device{}() }; //NOSONAR

        // thread local memory allocator and position buffer avoid 
        // synchronization delays
        allocator_type allocator( hp.nodes_per_block * sizeof( node_type ));
        vector< position_type > positions;

        // start with player 1, toggle for each self play run
        for (PlayerIndex player_index = PlayerIndex::Player1; 
             !cleanup_requested; player_index = toggle( player_index ))
        {
            alphazero::params::Ucb ucb_params
                { .c_base = hp.c_base, .c_init = hp.c_init };
            alphazero::params::GamePlay gameplay_params{
                .simulations = hp.simulations,
                .opening_moves = hp.opening_moves,
                .parallel_simulations = hp.parallel_simulations };

            auto player = make_unique< player_type >(
                game_type( player_index, initial_state ), ucb_params, 
                gameplay_params, g(), allocator, inference_service );

            positions.clear();
            auto selfplay = make_unique< selfplay_type >(
                *player, hp.dirichlet_alpha, hp.dirichlet_epsilon, g, 
                positions, root_node_entropy_stat );
            selfplay->run();

            // push positions into queue.
            for (auto const& pos : positions)
                // enter blocking wait if queue is full.
                while (!queue.push( pos ))
                {
                    unique_lock lock( queue_mutex );
                    queue_cv.wait( lock );
                }                    

            queue_cv.notify_all();
        }
    }

    // return number of created positions
    uint32_t measure_selfplay_throughput( uint32_t number_of_positions )
    {
        uint32_t total_positions = 0;
        unique_lock lock( queue_mutex );
        position_type pos;
        while (total_positions < number_of_positions)
        {
            queue_cv.wait( lock, [this] { return !queue.empty();});
            while (!queue.pop( pos ))
                ++total_positions;
        }

        return total_positions;
    }

    boost::lockfree::queue< position_type > queue;
    mutex queue_mutex;
    condition_variable queue_cv;

    vector< future< void >> thread_pool;

    libtorch::Hyperparameters hp;

    atomic< bool > cleanup_requested = false;

    Statistics root_node_entropy_stat;

    state_type initial_state;
    inference_service_type inference_service;
};

struct CppStats {
    float min;
    float max;
    float mean;
    float stddev;
};

enum class GameType : int32_t
{
    TTT = 1,
    UTTT = 2
};

using ttt_session_type = Session< ttt::alphazero::Player >;
using uttt_session_type = Session< uttt::alphazero::Player >;

template< typename SessionT >
void destroy_session( SessionT* session )
{
    session->cleanup_requested = true;
    session->queue_cv.notify_all();
    for (auto const& future : session->thread_pool)
        if (future.valid()) 
            future.wait();
    delete session; // NOSONAR
}

template< typename SessionT >
void get_inference_queue_stats(
    SessionT* ses, CppStats& inference_batch_size, CppStats& inference_time)
{
    auto const& batch_size_stats = 
        ses->inference_service.batch_size_stats();
    inference_batch_size = {
        batch_size_stats.min(), batch_size_stats.max(), 
        batch_size_stats.mean(), batch_size_stats.stddev()};

    auto const& inference_time_stats = 
        ses->inference_service.inference_time_stats();
    inference_time = {
        inference_time_stats.min(), inference_time_stats.max(), 
        inference_time_stats.mean(), inference_time_stats.stddev()};

    ses->inference_service.reset_stats();
}

// Use C-style linkage to prevent C++ name mangling, making it callable from
// Python.
extern "C" {

void* create_session( GameType game_type, const char* model_data, 
    uint32_t model_data_len, libtorch::Hyperparameters const& hp )
{
    torch::Device device = libtorch::get_device();
    auto model = libtorch::load_model( {model_data, model_data_len}, device );
    if (game_type == GameType::TTT)
        return new ttt_session_type( // NOSONAR
            ttt::empty_state, std::move( model ), hp ); 
    else if (game_type == GameType::UTTT)
        return new uttt_session_type( // NOSONAR
            uttt::empty_state, std::move( model ), hp ); 
    else
        return 0;
}

void destroy_session( GameType game_type, void* session)
{
    if (!session) 
        return;

    cout << "Requesting C++ worker thread cleanup..." << endl
         << "Waiting for self-play worker threads to join..." << endl;
    if (game_type == GameType::TTT)
        destroy_session( static_cast<ttt_session_type*>( session )); 
    else if (game_type == GameType::UTTT)
        destroy_session( static_cast<uttt_session_type*>( session )); 
}

// return number of created positions
uint32_t measure_selfplay_throughput(
    GameType game_type, const char* model_data, 
    uint32_t model_data_len, libtorch::Hyperparameters const& hp, 
    uint32_t number_of_positions )
{
    void* session = create_session( game_type, model_data, model_data_len, hp );
    uint32_t total_positions = 0;
    if (game_type == GameType::TTT)
    {
        auto ttt_session = static_cast<ttt_session_type*>( session );
        total_positions = ttt_session->measure_selfplay_throughput( 
            number_of_positions );
    }
    else if (game_type == GameType::UTTT)
    {
        auto uttt_session = static_cast<uttt_session_type*>( session );
        total_positions = uttt_session->measure_selfplay_throughput( 
            number_of_positions );
    }

    destroy_session( game_type, session );
    return total_positions;
}

void set_model(
    void* session, GameType game_type, const char* model_data, 
    uint32_t model_data_len )
{
    if (game_type == GameType::TTT)
    {
        auto ttt_session = static_cast<ttt_session_type*>( session );
        ttt_session->set_model( { model_data, model_data_len });
    }
    else if (game_type == GameType::UTTT)
    {
        auto uttt_session = static_cast<uttt_session_type*>( session );
        uttt_session->set_model( { model_data, model_data_len });
    }
}

/*
copy number_of_positions training data position to the memory locations
    provided by the data_pointers_out struct. */
void fetch_selfplay_data(
    void* session, GameType game_type, DataPointers& data_pointers_out,
    uint32_t number_of_positions )
{
    if (!session) 
        return;
    if (game_type == GameType::TTT)
    {
        auto ses = static_cast<ttt_session_type*>( session );
        ses->fetch_selfplay_data( data_pointers_out, number_of_positions ); 
    }
    else if (game_type == GameType::UTTT)
    {
        auto ses = static_cast<uttt_session_type*>( session );
        ses->fetch_selfplay_data( data_pointers_out, number_of_positions ); 
    }
}

void get_inference_queue_stats(
    void* session, GameType game_type, CppStats& inference_batch_size,
    CppStats& inference_time)
{
    if (game_type == GameType::TTT)
    {
        get_inference_queue_stats( 
            static_cast<ttt_session_type*>( session ),
            inference_batch_size, inference_time );
    }
    else if (game_type == GameType::UTTT)
    {
        get_inference_queue_stats( 
            static_cast<uttt_session_type*>( session ),
            inference_batch_size, inference_time );
    }
}

} // extern "C"