#pragma once

#include "games/ultimate_ttt.h"
#include "match.h"
#include <boost/json.hpp>
#include <fstream>
#include <mutex>
#include <sstream>

namespace evaluation
{

struct EvaluationStats
{
    size_t wins_p1;
    size_t wins_p2;
    size_t draws;
};

template < typename MoveT, typename StateT >
class RecordingMatch : public MultiMatch< MoveT, StateT >
{
  public:
    RecordingMatch( Game< MoveT, StateT > const& game,
                    PlayerFactory< MoveT, StateT > fst_player_factory,
                    PlayerFactory< MoveT, StateT > snd_player_factory,
                    typename MultiMatch< MoveT, StateT >::AllocatorFactory
                        allocator_factory,
                    int rounds, size_t number_of_threads, unsigned seed,
                    std::string const& save_path,
                    boost::json::object const& metadata )
        : MultiMatch< MoveT, StateT >( game, fst_player_factory,
                                       snd_player_factory, allocator_factory,
                                       rounds, number_of_threads, seed ),
          save_path( save_path ), metadata( metadata )
    {
    }

    ~RecordingMatch()
    {
        std::ofstream file( save_path, std::ios_base::app );
        print_match_report( file );
    }

  private:
    std::string save_path;
    boost::json::object metadata;
    std::mutex games_mutex;
    std::vector< boost::json::object > game_records;
    using game_moves_type = std::vector< std::pair< PlayerIndex, MoveT > >;
    std::map< std::thread::id, game_moves_type > current_games;
    std::vector< std::pair< GameResult, game_moves_type > > game_store;

    void print_match_report( std::ostream& os )
    {
        // Build final JSON
        boost::json::object report_obj;
        report_obj["metadata"] = metadata;

        boost::json::array games_array;

        for ( auto const& [result, moves] : game_store )
        {
            boost::json::object game_obj;
            std::stringstream ss;
            ss << result;

            game_obj["result"] = ss.str();
            game_obj["count"] = static_cast< int >( moves.size() );

            boost::json::array moves_json;
            for ( auto const& [player, move] : moves )
            {
                boost::json::object move_obj;

                ss.clear();
                ss << TaggedDispatch< StateT, MoveT >( move );
                move_obj["move"] = ss.str();

                ss.clear();
                ss << TaggedDispatch< StateT, PlayerIndex >( player );
                move_obj["player"] = ss.str();

                moves_json.push_back( move_obj );
            }
            game_obj["moves"] = moves_json;
            games_array.push_back( game_obj );
        }
        report_obj["games"] = games_array;

        os << boost::json::serialize( report_obj ) << "\n";
    }

    void report( Game< MoveT, StateT > const& game, MoveT const& move ) override
    {
        const auto thread_id = std::this_thread::get_id();
        std::lock_guard< std::mutex > _( games_mutex );
        auto& moves = current_games[thread_id];
        moves.push_back( { game.current_player_index(), move } );
        if ( game.result() != GameResult::Undecided )
        {
            game_store.push_back( { game.result(), moves } );
            current_games.erase( thread_id );
        }
    }
};

template < typename MoveT >
class TimingPlayerWrapper : public ::Player< MoveT >
{
  public:
    TimingPlayerWrapper( std::unique_ptr< ::Player< MoveT > >&& player,
                         Statistics* time_stats )
        : player( std::move( player ) ), time_stats( time_stats )
    {
    }

    MoveT choose_move() override
    {
        auto start = std::chrono::high_resolution_clock::now();
        MoveT move = player->choose_move();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration_us = std::chrono::duration_cast< std::chrono::microseconds >( end - start ).count();
        time_stats->update( static_cast< float >( duration_us ) );
        
        return move;
    }

    void apply_opponent_move( MoveT const& move ) override
    {
        player->apply_opponent_move( move );
    }

  private:
    std::unique_ptr< ::Player< MoveT > > player;
    Statistics* time_stats;
};

template < typename MoveT, typename StateT, typename PlayerT >
EvaluationStats
evaluate_matchup( [[maybe_unused]] int32_t game_type_val,
                  libtorch::MatchupPlayerConfig const& p1,
                  libtorch::MatchupPlayerConfig const& p2,
                  int rounds,
                  std::string const& save_path,
                  std::string const& run_name,
                  int step,
                  StateT const& initial_state,
                  unsigned seed,
                  unsigned block_size )
{
    using namespace std;

    // Setup metadata
    boost::json::object metadata;
    metadata["run_name"] = run_name;
    metadata["step"] = step;
    metadata["model1"] = p1.type == 1 ? "mcts" : ( p1.type == 2 ? "minimax" : "tree_minimax" );
    metadata["model2"] = p2.type == 1 ? "mcts" : ( p2.type == 2 ? "minimax" : "tree_minimax" );

    torch::Device device = libtorch::get_device();
    auto make_buf = []( const string& s )
    { return libtorch::DataBuffer{ s.data(), (uint32_t)s.size() }; };

    using game_type = Game< MoveT, StateT >;
    using inference_service =
        libtorch::InferenceService< PlayerT::game_size, PlayerT::policy_size >;

    unique_ptr< inference_service > service1;
    unique_ptr< inference_service > service2;

    if ( p1.type == 1 )
    {
        std::cout << "Loading Player 1 MCTS model..." << std::endl;
        std::string m1( p1.model_data, p1.model_data_len );
        auto model1 = libtorch::load_model( make_buf( m1 ), device );
        service1 = make_unique< inference_service >( std::move( model1 ), device, p1.hp.max_batch_size );
    }

    if ( p2.type == 1 )
    {
        std::cout << "Loading Player 2 MCTS model..." << std::endl;
        std::string m2( p2.model_data, p2.model_data_len );
        auto model2 = libtorch::load_model( make_buf( m2 ), device );
        service2 = make_unique< inference_service >( std::move( model2 ), device, p2.hp.max_batch_size );
    }

    if ( service1 || service2 )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
    }

    std::cout << "Running matchup match..." << std::endl;

    Statistics p1_time;
    Statistics p2_time;

    auto make_factory = [&]( libtorch::MatchupPlayerConfig const& p_cfg,
                             inference_service* service,
                             Statistics* time_stats ) -> PlayerFactory< MoveT, StateT >
    {
        return [&, service, p_cfg, time_stats]( game_type const& g, unsigned seed,
                                                GenerationalArenaAllocator* allocator )
            -> std::unique_ptr< ::Player< MoveT > >
        {
            std::unique_ptr< ::Player< MoveT > > raw_player;

            if ( p_cfg.type == 1 ) // MCTS
            {
                alphazero::params::Ucb ucb{ p_cfg.hp.c_base, p_cfg.hp.c_init };
                alphazero::params::GamePlay gp{ p_cfg.hp.simulations, p_cfg.hp.opening_moves,
                                                p_cfg.hp.parallel_simulations };

                raw_player = std::make_unique< PlayerT >( g, ucb, gp, seed, *allocator,
                                                          *service );
            }
            else if ( p_cfg.type == 2 ) // Standard Minimax
            {
                if constexpr ( std::is_same_v< StateT, uttt::State > )
                {
                    raw_player = std::make_unique< uttt::minimax::Player >( g, 9.0, p_cfg.simulations_or_depth, seed );
                }
                else
                {
                    raw_player = std::make_unique< ttt::minimax::Player >( g, p_cfg.simulations_or_depth, seed );
                }
            }
            else if ( p_cfg.type == 3 ) // Tree Minimax
            {
                if constexpr ( std::is_same_v< StateT, uttt::State > )
                {
                    raw_player = std::make_unique< uttt::minimax::tree::Player >( g, 9.0, p_cfg.simulations_or_depth, seed, *allocator );
                }
                else
                {
                    raw_player = std::make_unique< ttt::minimax::tree::Player >( g, p_cfg.simulations_or_depth, seed, *allocator );
                }
            }
            else
            {
                throw std::runtime_error( "Unknown player type in factory" );
            }

            return std::make_unique< TimingPlayerWrapper< MoveT > >( std::move( raw_player ), time_stats );
        };
    };

    auto factory1 = make_factory( p1, service1.get(), &p1_time );
    auto factory2 = make_factory( p2, service2.get(), &p2_time );

    auto allocator_factory = [block_size]()
    { return make_unique< GenerationalArenaAllocator >( block_size ); };

    unsigned num_parallel = 1;
    if ( p1.type == 1 ) num_parallel = std::max( num_parallel, (unsigned)p1.hp.parallel_games );
    if ( p2.type == 1 ) num_parallel = std::max( num_parallel, (unsigned)p2.hp.parallel_games );

    Game< MoveT, StateT > game( PlayerIndex::Player1, initial_state );
    RecordingMatch< MoveT, StateT > match(
        game, factory1, factory2, allocator_factory, rounds,
        num_parallel, seed, save_path, metadata );
    match.run();

    if ( service1 )
    {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Model 1 Inference Timing & Batch Statistics:" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Batch Size Stats:      " << service1->batch_size_stats() << std::endl;
        std::cout << "Inference Time (μs):   " << service1->inference_time_stats() << std::endl;
    }

    if ( service2 )
    {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Model 2 Inference Timing & Batch Statistics:" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Batch Size Stats:      " << service2->batch_size_stats() << std::endl;
        std::cout << "Inference Time (μs):   " << service2->inference_time_stats() << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Summarized thinking time per player:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Player 1 (Type: " << ( p1.type == 1 ? "MCTS" : ( p1.type == 2 ? "Minimax" : "Tree Minimax" ) ) << "):" << std::endl;
    std::cout << "  " << p1_time;
    std::cout << "Player 2 (Type: " << ( p2.type == 1 ? "MCTS" : ( p2.type == 2 ? "Minimax" : "Tree Minimax" ) ) << "):" << std::endl;
    std::cout << "  " << p2_time;
    std::cout << "========================================\n" << std::endl;

    return { match.get_fst_player_wins(), match.get_snd_player_wins(),
             match.get_draws() };
}

template < typename MoveT, typename StateT, typename MinimaxPlayerT1, typename MinimaxPlayerT2 >
EvaluationStats
evaluate_minimax_vs_minimax( int rounds, unsigned depth1, unsigned depth2,
          std::string const& save_path, std::string const& run_name, int step,
          StateT const& initial_state, unsigned seed, int parallel_games )
{
    using namespace std;

    // Setup metadata
    boost::json::object metadata;
    metadata["run_name"] = run_name;
    metadata["step"] = step;
    metadata["model1"] = "minimax_depth_" + to_string(depth1);
    metadata["model2"] = "minimax_depth_" + to_string(depth2);

    using game_type = Game< MoveT, StateT >;

    std::cout << "Running Minimax (Depth " << depth1 << ") vs Minimax (Depth " << depth2 << ")..." << std::endl;

    auto factory1 = [&]( game_type const& g, unsigned seed,
                         GenerationalArenaAllocator* )
        -> std::unique_ptr< ::Player< MoveT > >
    {
        if constexpr ( std::is_same_v< StateT, uttt::State > )
        {
            return std::make_unique< MinimaxPlayerT1 >( g, 9.0, depth1, seed );
        }
        else
        {
            return std::make_unique< MinimaxPlayerT1 >( g, depth1, seed );
        }
    };

    auto factory2 = [&]( game_type const& g, unsigned seed,
                         GenerationalArenaAllocator* )
        -> std::unique_ptr< ::Player< MoveT > >
    {
        if constexpr ( std::is_same_v< StateT, uttt::State > )
        {
            return std::make_unique< MinimaxPlayerT2 >( g, 9.0, depth2, seed );
        }
        else
        {
            return std::make_unique< MinimaxPlayerT2 >( g, depth2, seed );
        }
    };

    Game< MoveT, StateT > game( PlayerIndex::Player1, initial_state );
    RecordingMatch< MoveT, StateT > match(
        game, factory1, factory2, [] { return nullptr; }, rounds,
        parallel_games, seed, save_path, metadata );
    match.run();

    return { match.get_fst_player_wins(), match.get_snd_player_wins(),
             match.get_draws() };
}

} // namespace evaluation
