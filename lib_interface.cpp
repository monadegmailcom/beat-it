#include "evaluation.h"
#include "statistics.h"
#include <future>

using namespace std;

enum class GameType : int32_t
{
    TTT = 1,
    UTTT = 2
};

// Use C-style linkage to prevent C++ name mangling, making it callable from
// Python.
extern "C"
{
    evaluation::EvaluationStats
    evaluate_models( int32_t game_type, const char* model1_data,
                     uint32_t model1_len, const char* model2_data,
                     uint32_t model2_len, libtorch::Hyperparameters const& hp,
                     int32_t rounds, const char* save_path_c,
                     const char* run_name_c, int32_t step )
    {
        std::string m1( model1_data, model1_len );
        std::string m2( model2_data, model2_len );
        std::string save_path( save_path_c );
        std::string run_name( run_name_c );

        unsigned seed = 0;
        evaluation::EvaluationStats stats;
        if ( game_type == static_cast< int32_t >( GameType::TTT ) )
        {
            auto f = evaluation::evaluate< ttt::Move, ttt::State,
                                           ttt::alphazero::Player >;
            stats = f( m1, m2, hp, rounds, save_path, run_name, step,
                       ttt::empty_state, seed,
                       50 * hp.simulations * sizeof( ttt::alphazero::Node ) );
        }
        else if ( game_type == static_cast< int32_t >( GameType::UTTT ) )
        {
            auto f = evaluation::evaluate< uttt::Move, uttt::State,
                                           uttt::alphazero::Player >;
            stats = f( m1, m2, hp, rounds, save_path, run_name, step,
                       uttt::empty_state, seed,
                       50 * hp.simulations * sizeof( uttt::alphazero::Node ) );
        }
        else
        {
            throw std::runtime_error( "Unsupported game type" );
        }

        return { static_cast< uint32_t >( stats.wins_p1 ),
                 static_cast< uint32_t >( stats.wins_p2 ),
                 static_cast< uint32_t >( stats.draws ) };
    }
}

// A struct to define the layout of the data pointers. This must be mirrored in
//   Python.
struct DataPointers
{
    float* game_states = nullptr;      // G floats
    float* policy_targets = nullptr;   // P floats
    float* value_targets = nullptr;    // 1 float
    int32_t* player_indices = nullptr; // 1 int32_t
};

struct CppStats
{
    float min;
    float max;
    float mean;
    float stddev;
};

void statistic_to_cpp_stat( Statistics const& stat, CppStats& cpp )
{
    cpp.min = stat.min();
    cpp.max = stat.max();
    cpp.mean = stat.mean();
    cpp.stddev = stat.stddev();
}

// The Session struct encapsulates all state for a single training run.
// This avoids global variables and allows for multiple independent sessions.
template < typename PlayerT > struct Session
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
    using position_type = alphazero::training::Position< PlayerT::game_size,
                                                         PlayerT::policy_size >;
    using player_type = PlayerT;
    using selfplay_type =
        alphazero::training::SelfPlay< typename game_type::move_type,
                                       state_type, PlayerT::game_size,
                                       PlayerT::policy_size >;

    Session( state_type const& initial_state,
             std::unique_ptr< torch::jit::script::Module >&& model,
             libtorch::Hyperparameters const& hp )
        : queue( hp.max_batch_size ), hp( hp ), initial_state( initial_state ),
          thread_pool( hp.parallel_games ),
          inference_service( std::move( model ), libtorch::get_device(),
                             hp.max_batch_size )
    {
        allocators.reserve( hp.parallel_games );
        for ( size_t i = 0; i < hp.parallel_games; ++i )
            allocators.push_back( make_unique< allocator_type >(
                hp.nodes_per_block * sizeof( node_type ) ) );

        cout << "start " << thread_pool.size() << " parallel games" << endl;
        size_t i = 0;
        for ( auto& future : thread_pool )
            future =
                async( &Session::worker, this, std::ref( *allocators[i++] ) );
    }

    void set_model( libtorch::DataBuffer model_buffer,
                    CppStats& inference_batch_size, CppStats& inference_time )
    {
        if ( cleanup_requested )
            return;

        torch::Device device = libtorch::get_device();
        auto model = libtorch::load_model( model_buffer, device );

        Statistics batch_size_stat;
        Statistics inference_time_stat;

        inference_service.update_model( std::move( model ), batch_size_stat,
                                        inference_time_stat );
        statistic_to_cpp_stat( batch_size_stat, inference_batch_size );
        statistic_to_cpp_stat( inference_time_stat, inference_time );
    }

    void fetch_selfplay_data( DataPointers& data_pointers_out,
                              uint32_t number_of_positions )
    {
        position_type pos;
        for ( size_t i = 0; i < number_of_positions; ++i )
        {
            // pop next position from queue, enter blocking wait if queue is
            // empty.
            while ( !cleanup_requested && !queue.pop( pos ) )
            {
                // notify workers.
                queue_cv.notify_all();

                unique_lock lock( queue_mutex );
                queue_cv.wait(
                    lock,
                    [this] { return !queue.empty() || cleanup_requested; } );
            }

            // copy position into buffer.
            ranges::copy( pos.game_state, data_pointers_out.game_states +
                                              i * PlayerT::game_size );
            ranges::copy( pos.target_policy, data_pointers_out.policy_targets +
                                                 i * PlayerT::policy_size );
            data_pointers_out.value_targets[i] = pos.target_value;
            data_pointers_out.player_indices[i] =
                static_cast< int32_t >( pos.current_player );
        }

        // notify workers.
        queue_cv.notify_all();
    }

    void pause() { paused = true; }

    void resume()
    {
        paused = false;
        pause_cv.notify_all();
    }

    void worker( allocator_type& allocator )
    {
        // use some tls resources
        mt19937 g{ random_device{}() }; // NOSONAR

        // thread local memory allocator and position buffer avoid
        // synchronization delays
        vector< position_type > positions;

        // start with player 1, toggle for each self play run
        for ( PlayerIndex player_index = PlayerIndex::Player1;
              !cleanup_requested; player_index = toggle( player_index ) )
        {
            // Check for pause
            {
                unique_lock lock( pause_mutex );
                pause_cv.wait( lock, [this]
                               { return !paused || cleanup_requested; } );
            }
            if ( cleanup_requested )
                break;

            alphazero::params::Ucb ucb_params{ .c_base = hp.c_base,
                                               .c_init = hp.c_init };
            alphazero::params::GamePlay gameplay_params{
                .simulations = hp.simulations,
                .opening_moves = hp.opening_moves,
                .parallel_simulations = hp.parallel_simulations };

            player_type player( game_type( player_index, initial_state ),
                                ucb_params, gameplay_params, g(), allocator,
                                inference_service );

            positions.clear();
            selfplay_type selfplay(
                player, hp.dirichlet_alpha, hp.dirichlet_epsilon, g, positions,
                root_node_entropy_stat, informed_selection_stat );
            selfplay.run();
            {
                lock_guard lock( allocator_mutex );
                allocator_stat.update( allocator.allocated_size() );
            }

            // push positions into queue. block if queue is full.
            for ( auto const& pos : positions )
                while ( !cleanup_requested && !queue.push( pos ) )
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
        while ( total_positions < number_of_positions )
        {
            queue_cv.wait( lock, [this] { return !queue.empty(); } );
            while ( queue.pop( pos ) )
                ++total_positions;
        }
        return total_positions;
    }

    boost::lockfree::queue< position_type > queue;
    libtorch::Hyperparameters hp;
    state_type initial_state;
    mutex queue_mutex;
    condition_variable queue_cv;

    vector< future< void > > thread_pool;
    vector< unique_ptr< allocator_type > > allocators;

    atomic< bool > cleanup_requested = false;
    atomic< bool > paused = false;
    mutex pause_mutex;
    condition_variable pause_cv;

    Statistics root_node_entropy_stat;
    Statistics informed_selection_stat;
    Statistics allocator_stat;
    mutex allocator_mutex;

    inference_service_type inference_service;
};

using ttt_session_type = Session< ttt::alphazero::Player >;
using uttt_session_type = Session< uttt::alphazero::Player >;

template < typename SessionT > void destroy_session( SessionT* session )
{
    session->cleanup_requested = true;
    session->queue_cv.notify_all();
    for ( auto const& future : session->thread_pool )
        if ( future.valid() )
            future.wait();
    delete session; // NOSONAR
}

// Use C-style linkage to prevent C++ name mangling, making it callable from
// Python.
extern "C"
{

    void* create_session( GameType game_type, const char* model_data,
                          uint32_t model_data_len,
                          libtorch::Hyperparameters const& hp )
    {
        torch::Device device = libtorch::get_device();
        auto model =
            libtorch::load_model( { model_data, model_data_len }, device );
        if ( game_type == GameType::TTT )
            return new ttt_session_type( // NOSONAR
                ttt::empty_state, std::move( model ), hp );
        else if ( game_type == GameType::UTTT )
            return new uttt_session_type( // NOSONAR
                uttt::empty_state, std::move( model ), hp );
        else
            return 0;
    }

    void destroy_session( GameType game_type, void* session )
    {
        if ( !session )
            return;

        cout << "Requesting C++ worker thread cleanup..." << endl
             << "Waiting for self-play worker threads to join..." << endl;
        if ( game_type == GameType::TTT )
            destroy_session( static_cast< ttt_session_type* >( session ) );
        else if ( game_type == GameType::UTTT )
            destroy_session( static_cast< uttt_session_type* >( session ) );
    }

    // return number of created positions
    uint32_t measure_selfplay_throughput( GameType game_type,
                                          const char* model_data,
                                          uint32_t model_data_len,
                                          libtorch::Hyperparameters const& hp,
                                          uint32_t number_of_positions )
    {
        void* session =
            create_session( game_type, model_data, model_data_len, hp );
        uint32_t total_positions = 0;
        if ( game_type == GameType::TTT )
        {
            auto ttt_session = static_cast< ttt_session_type* >( session );
            total_positions =
                ttt_session->measure_selfplay_throughput( number_of_positions );
        }
        else if ( game_type == GameType::UTTT )
        {
            auto uttt_session = static_cast< uttt_session_type* >( session );
            total_positions = uttt_session->measure_selfplay_throughput(
                number_of_positions );
        }

        destroy_session( game_type, session );
        return total_positions;
    }

    void set_model( void* session, GameType game_type, const char* model_data,
                    uint32_t model_data_len, CppStats& inference_batch_size,
                    CppStats& inference_time, CppStats& allocator_size )
    {
        if ( game_type == GameType::TTT )
        {
            auto ttt_session = static_cast< ttt_session_type* >( session );
            ttt_session->set_model( { model_data, model_data_len },
                                    inference_batch_size, inference_time );
            scoped_lock _( ttt_session->allocator_mutex );
            statistic_to_cpp_stat( ttt_session->allocator_stat,
                                   allocator_size );
        }
        else if ( game_type == GameType::UTTT )
        {
            auto uttt_session = static_cast< uttt_session_type* >( session );
            uttt_session->set_model( { model_data, model_data_len },
                                     inference_batch_size, inference_time );
            scoped_lock _( uttt_session->allocator_mutex );
            statistic_to_cpp_stat( uttt_session->allocator_stat,
                                   allocator_size );
        }
    }

    /*
    copy number_of_positions training data position to the memory locations
        provided by the data_pointers_out struct. */
    void fetch_selfplay_data( void* session, GameType game_type,
                              DataPointers* data_pointers_out,
                              uint32_t number_of_positions,
                              CppStats& inference_batch_size,
                              CppStats& inference_time,
                              CppStats& allocator_size )
    {
        if ( !session || !data_pointers_out )
            return;
        if ( game_type == GameType::TTT )
        {
            auto ses = static_cast< ttt_session_type* >( session );
            ses->fetch_selfplay_data( *data_pointers_out, number_of_positions );

            statistic_to_cpp_stat( ses->inference_service.batch_size_stats(),
                                   inference_batch_size );
            statistic_to_cpp_stat(
                ses->inference_service.inference_time_stats(), inference_time );
            ses->inference_service.reset_stats();

            scoped_lock _( ses->allocator_mutex );
            statistic_to_cpp_stat( ses->allocator_stat, allocator_size );
            ses->allocator_stat.reset();
        }
        else if ( game_type == GameType::UTTT )
        {
            auto ses = static_cast< uttt_session_type* >( session );
            ses->fetch_selfplay_data( *data_pointers_out, number_of_positions );

            statistic_to_cpp_stat( ses->inference_service.batch_size_stats(),
                                   inference_batch_size );
            statistic_to_cpp_stat(
                ses->inference_service.inference_time_stats(), inference_time );
            ses->inference_service.reset_stats();

            scoped_lock _( ses->allocator_mutex );
            statistic_to_cpp_stat( ses->allocator_stat, allocator_size );
            ses->allocator_stat.reset();
        }
    }

    void pause_session( void* session, GameType game_type )
    {
        if ( !session )
            return;
        if ( game_type == GameType::TTT )
            static_cast< ttt_session_type* >( session )->pause();
        else if ( game_type == GameType::UTTT )
            static_cast< uttt_session_type* >( session )->pause();
    }

    void resume_session( void* session, GameType game_type )
    {
        if ( !session )
            return;
        if ( game_type == GameType::TTT )
            static_cast< ttt_session_type* >( session )->resume();
        else if ( game_type == GameType::UTTT )
            static_cast< uttt_session_type* >( session )->resume();
    }

} // extern "C"