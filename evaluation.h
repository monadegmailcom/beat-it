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

template < typename MoveT, typename StateT, typename PlayerT >
EvaluationStats
evaluate( std::string const& model1_data, std::string const& model2_data,
          libtorch::Hyperparameters const& hp, int rounds,
          std::string const& save_path, std::string const& run_name, int step,
          StateT const& initial_state, unsigned seed, unsigned block_size )
{
    using namespace std;

    // Setup metadata
    boost::json::object metadata;
    metadata["run_name"] = run_name;
    metadata["step"] = step;
    metadata["model1"] = "current";
    metadata["model2"] = "baseline";

    torch::Device device = libtorch::get_device();

    auto make_buf = []( const string& s )
    { return libtorch::DataBuffer{ s.data(), (uint32_t)s.size() }; };

    auto model1 = libtorch::load_model( make_buf( model1_data ), device );
    auto model2 = libtorch::load_model( make_buf( model2_data ), device );

    using game_type = Game< MoveT, StateT >;

    using inference_service =
        libtorch::InferenceService< PlayerT::game_size, PlayerT::policy_size >;
    inference_service service1( std::move( model1 ), device,
                                hp.max_batch_size );
    inference_service service2( std::move( model2 ), device,
                                hp.max_batch_size );

    auto factory1 = [&]( game_type const& g, unsigned seed,
                         GenerationalArenaAllocator* allocator )
        -> std::unique_ptr< PlayerT >
    {
        alphazero::params::Ucb ucb{ hp.c_base, hp.c_init };
        alphazero::params::GamePlay gp{ hp.simulations, hp.opening_moves,
                                        hp.parallel_simulations };

        return std::make_unique< PlayerT >( g, ucb, gp, seed, *allocator,
                                            service1 );
    };

    auto factory2 = [&]( game_type const& g, unsigned seed,
                         GenerationalArenaAllocator* allocator )
        -> std::unique_ptr< PlayerT >
    {
        alphazero::params::Ucb ucb{ hp.c_base, hp.c_init };
        alphazero::params::GamePlay gp{ hp.simulations, hp.opening_moves,
                                        hp.parallel_simulations };

        return std::make_unique< PlayerT >( g, ucb, gp, seed, *allocator,
                                            service2 );
    };

    auto allocator_factory = [block_size]()
    { return make_unique< GenerationalArenaAllocator >( block_size ); };

    Game< MoveT, StateT > game( PlayerIndex::Player1, initial_state );
    RecordingMatch< MoveT, StateT > match(
        game, factory1, factory2, allocator_factory, rounds,
        1, seed, save_path, metadata );
    match.run();

    return { match.get_fst_player_wins(), match.get_snd_player_wins(),
             match.get_draws() };
}

} // namespace evaluation
