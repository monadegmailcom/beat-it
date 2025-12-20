#include "alphazero.h"
#include "games/nim.h"
#include "games/ultimate_ttt.h"
#include "libtorch_util.h"
#include "match.h"
#include "montecarlo.h"
#include "node.h"
#include "statistics.h"

#include <boost/json.hpp>

#include <cassert>
#include <exception>
#include <future>
#include <iostream>
#include <map>
#include <source_location>
#include <utility>
#include <vector>

using namespace std;

random_device rd; // NOSONAR
const unsigned seed = rd();
const bool verbose = true;
const bool interactive = false;
const bool extensive = true;

template <>
struct GameState< char, GameResult >
    : public GameStateBase< GameState< char, GameResult >, char, GameResult >
{
    static void next_valid_move( optional< char > &move, PlayerIndex,
                                 GameResult const & )
    {
        if ( move.has_value() )
            move = 'a';
        else if ( move == 'a' )
            move = 'b';
        else
            move.reset();
    }

    static GameResult apply( char const &, PlayerIndex,
                             GameResult const &state )
    {
        return state;
    }

    static GameResult result( PlayerIndex, GameResult const &state )
    {
        return state;
    }
};

namespace test
{

void toggle_player()
{
    cout << source_location::current().function_name() << endl;
    using enum PlayerIndex;
    if ( toggle( Player1 ) != Player2 || toggle( Player2 ) != Player1 )
        assert( false );
}

using TestGame = Game< char, GameResult >;

struct TestPlayer : public Player< char >
{
    char choose_move() override { return 0; }
};

void build_game()
{
    cout << std::source_location::current().function_name() << endl;

    TestGame game( PlayerIndex::Player2, GameResult::Undecided );
    minimax::Player player( game, 0, seed );

    assert( game.current_player_index() == PlayerIndex::Player2 );
    if ( game.current_player_index() != PlayerIndex::Player2 )
        assert( !"invalid current player index" );
}

void eval_won_game()
{
    cout << source_location::current().function_name() << endl;
    mt19937 g;
    vector< char > move_stack;
    minimax::ScoreFunction< char, GameResult > score =
        []( Game< char, GameResult > const & ) { return 0.0; };
    size_t calls = 0;
    if ( minimax::eval< char >(
             TestGame( PlayerIndex::Player2, GameResult::Player2Win ), score, 0,
             0.0, 1.0, g, calls ) != INFINITY )
        assert( !"wrong score for won game" );
    if ( minimax::eval< char >(
             TestGame( PlayerIndex::Player2, GameResult::Player1Win ), score, 0,
             0.0, 1.0, g, calls ) != -INFINITY )
        assert( !"wrong score for won game" );
}

void eval_drawn_game()
{
    cout << source_location::current().function_name() << endl;
    mt19937 g;
    vector< char > move_stack;

    minimax::ScoreFunction< char, GameResult > score = []( TestGame const & )
    { return 0.0; };
    size_t calls = 0;
    if ( minimax::eval< char >(
             TestGame( PlayerIndex::Player2, GameResult::Draw ), score, 0, 0.0,
             1.0, g, calls ) != 0.0 )
        assert( !"wrong score for drawn game" );
}

void eval_undecided_game()
{
    cout << source_location::current().function_name() << endl;

    TestGame undecided_game( PlayerIndex::Player2, GameResult::Undecided );

    minimax::ScoreFunction< char, GameResult > score = []( TestGame const & )
    { return 42.0; };
    mt19937 g;
    vector< char > move_stack;
    size_t calls = 0;

    if ( minimax::eval( undecided_game, score, 0, 0.0, 1.0, g, calls ) != 42.0 )
        assert( !"wrong score for undecided game" );
}

struct TestNimPlayer : public Player< nim::Move >
{
    nim::Move choose_move() override { return next_move; }

    void apply_opponent_move( nim::Move const & ) override
    {
        // do nothing
    }

    nim::Move next_move;
};

void nim_game()
{
    cout << source_location::current().function_name() << endl;

    nim::Game< 2 > game( PlayerIndex::Player1, array< size_t, 2 >{ 1, 2 } );

    {
        auto moves =
            vector{ nim::Move{ 0, 1 }, nim::Move{ 1, 1 }, nim::Move{ 1, 2 } };
        auto valid_move = game.begin();
        assert( std::find( moves.begin(), moves.end(), *valid_move ) !=
                moves.end() );
        erase( moves, *valid_move );
        ++valid_move;

        assert( std::find( moves.begin(), moves.end(), *valid_move ) !=
                moves.end() );
        erase( moves, *valid_move );
        ++valid_move;

        assert( ranges::contains( moves.begin(), moves.end(), *valid_move ) );
        erase( moves, *valid_move );
        valid_move++; // try post-increment

        assert( valid_move == game.end() );
        assert( moves.empty() );
    }

    game = game.apply( nim::Move{ 1, 1 } );
    assert( game.current_player_index() == PlayerIndex::Player2 );
    assert( ranges::is_permutation(
        vector{ nim::Move{ 0, 1 }, nim::Move{ 1, 1 } },
        vector< nim::Move >( game.begin(), game.end() ) ) );

    game = game.apply( nim::Move{ 1, 1 } );
    assert( game.current_player_index() == PlayerIndex::Player1 );
    assert( ranges::is_permutation(
        vector{ nim::Move{ 0, 1 } },
        vector< nim::Move >( game.begin(), game.end() ) ) );

    game = game.apply( nim::Move{ 0, 1 } );
    assert( game.current_player_index() == PlayerIndex::Player2 );
    assert( game.result() == GameResult::Player2Win );
}

void nim_match()
{
    if ( extensive )
        cout << source_location::current().function_name() << endl;
    else
    {
        cout << source_location::current().function_name()
             << " (extensive mode off)" << endl;
        return;
    }

    const size_t HEAPS = 5;

    nim::Game< HEAPS > game( PlayerIndex::Player1, { 1, 2, 3, 4, 5 } );

    PlayerFactory< nim::Move, nim::State< HEAPS > > factory1 =
        []( nim::Game< HEAPS > const &game, unsigned seed )
    {
        return std::make_unique< nim::minimax::Player< HEAPS > >( game, 2,
                                                                  seed );
    };
    PlayerFactory< nim::Move, nim::State< HEAPS > > factory2 =
        []( nim::Game< HEAPS > const &game, unsigned seed )
    {
        return std::make_unique< nim::minimax::Player< HEAPS > >( game, 3,
                                                                  seed );
    };

    MultiMatch match( game, factory1, factory2, 100, 1, seed );

    match.run();

    if ( verbose )
        cout << "fst player wins: " << match.get_fst_player_wins() << '\n'
             << "snd player wins: " << match.get_snd_player_wins() << '\n'
             << "draws: " << match.get_draws() << '\n'
             << endl;

    assert( match.get_fst_player_wins() < 50 );
    assert( match.get_snd_player_wins() > 50 );
    assert( match.get_draws() == 0 );
}

void ttt_game()
{
    cout << source_location::current().function_name() << endl;

    ttt::Game game( PlayerIndex::Player1, ttt::empty_state );

    assert( game.result() == GameResult::Undecided );

    assert(
        ranges::is_permutation( vector< ttt::Move >( game.begin(), game.end() ),
                                vector{ 0, 1, 2, 3, 4, 5, 6, 7, 8 } ) );
    game = game.apply( ttt::Move( 4 ) );
    assert( game.current_player_index() == PlayerIndex::Player2 );

    vector< ttt::Move > moves{ 0, 1, 2, 3, 5, 6, 7, 8 };
    assert( ranges::is_permutation(
        vector< ttt::Move >( game.begin(), game.end() ), moves ) );

    auto valid_move = game.begin();
    assert( *valid_move == ttt::Move( 0 ) );
    assert( ranges::contains( moves.begin(), moves.end(), *valid_move ) );
    erase( moves, *valid_move );
    ++valid_move;

    assert( *valid_move == ttt::Move( 1 ) );
    assert( ranges::contains( moves.begin(), moves.end(), *valid_move ) );
    erase( moves, *valid_move );

    assert( *valid_move++ == ttt::Move( 1 ) );
    assert( *valid_move == ttt::Move( 2 ) );

    assert( *game.begin() == ttt::Move( 0 ) );

    game = game.apply( ttt::Move( 0 ) );
    assert( game.current_player_index() == PlayerIndex::Player1 );
    assert( ranges::is_permutation( vector( game.begin(), game.end() ),
                                    vector{ 1, 2, 3, 5, 6, 7, 8 } ) );

    game = game.apply( ttt::Move( 7 ) );
    assert( game.current_player_index() == PlayerIndex::Player2 );
    assert( ranges::is_permutation( vector( game.begin(), game.end() ),
                                    vector{ 1, 2, 3, 5, 6, 8 } ) );

    game = game.apply( ttt::Move( 2 ) );
    assert( game.current_player_index() == PlayerIndex::Player1 );
    assert( ranges::is_permutation( vector( game.begin(), game.end() ),
                                    vector{ 1, 3, 5, 6, 8 } ) );

    game = game.apply( ttt::Move( 8 ) );
    assert( game.current_player_index() == PlayerIndex::Player2 );
    assert( ranges::is_permutation( vector( game.begin(), game.end() ),
                                    vector{ 1, 3, 5, 6 } ) );

    assert( game.result() == GameResult::Undecided );
    game = game.apply( ttt::Move( 1 ) );
    assert( game.current_player_index() == PlayerIndex::Player1 );

    assert( game.result() == GameResult::Player2Win );
}

struct TicTacToeMatch : public Match< ttt::Move, ttt::State >
{
    explicit TicTacToeMatch( Player< ttt::Move > const &player )
        : player( player )
    {
    }

    Player< ttt::Move > const &player;

    void report( ttt::Game const &game, ttt::Move const &move ) override
    {
        cout << "player " << game.current_player_index() << " (" << (int)move
             << ")\n"
             << "resulting board:\n"
             << game << '\n';
        if ( auto p = dynamic_cast< const ttt::minimax::Player * >( &player ) )
            cout << "score: " << p->score( game ) << endl;
        else if ( auto mp = dynamic_cast< const ttt::montecarlo::Player * >(
                      &player ) )
            cout << "point ratio: " << mp->root_node().get_payload().visits
                 << endl;
    }
};

void ttt_human()
{
    if ( interactive )
        cout << source_location::current().function_name() << endl;
    else
    {
        cout << source_location::current().function_name()
             << " (interactive mode off)" << endl;
        return;
    }

    ttt::Game game( Player1, ttt::empty_state );

    ttt::console::HumanPlayer human( game );
    chrono::microseconds human_duration;
    ttt::minimax::Player player( game, 0, seed );
    chrono::microseconds player_duration;
    TicTacToeMatch match( player );
    if ( GameResult result =
             match.play( game, human, human_duration, player, player_duration );
         result == GameResult::Player1Win )
        cout << "player 1 wins\n";
    else if ( result == GameResult::Player2Win )
        cout << "player 2 wins\n";
    else
        cout << "draw\n";
    cout << endl;
}

void tic_tac_toe_match()
{
    if ( extensive )
        cout << source_location::current().function_name() << endl;
    else
    {
        cout << source_location::current().function_name()
             << " (extensive mode off)" << endl;
        return;
    }

    ttt::Game game( Player1, ttt::empty_state );

    PlayerFactory< ttt::Move, ttt::State > factory1 =
        []( ttt::Game const &game, unsigned seed )
    { return make_unique< ttt::minimax::Player >( game, 0, seed ); };
    PlayerFactory< ttt::Move, ttt::State > factory2 =
        []( ttt::Game const &game, unsigned seed )
    { return make_unique< ttt::minimax::Player >( game, 5, seed ); };
    MultiMatch match( game, factory1, factory2, 100, 1, seed );
    match.run();

    if ( verbose )
        cout << "fst player wins: " << match.get_fst_player_wins() << '\n'
             << "snd player wins: " << match.get_snd_player_wins() << '\n'
             << "draws: " << match.get_draws() << endl;

    assert( match.get_fst_player_wins() == 0 );
    assert( match.get_draws() > 50 );
}

struct UltimateTicTacToeMatch : public Match< uttt::Move, uttt::State >
{
    explicit UltimateTicTacToeMatch(
        uttt::minimax::Player const &minimax_player )
        : minimax_player( minimax_player )
    {
    }

    uttt::minimax::Player const &minimax_player;

    void report( uttt::Game const &game, uttt::Move const &move ) override
    {
        cout << "player " << game.current_player_index() << " ("
             << (int)move.big_move << "," << (int)move.small_move << ")\n"
             << "resulting board:\n"
             << game << '\n'
             << "score: " << minimax_player.score( game ) << endl;
    }
};

void uttt_game()
{
    cout << source_location::current().function_name() << endl;

    uttt::Game game( Player1, uttt::empty_state );

    assert( game.result() == GameResult::Undecided );

    assert( vector( game.begin(), game.end() ).size() == 81 );

    game = game.apply( uttt::Move{ 4, 4 } );
    assert( game.current_player_index() == Player2 );
    assert( ranges::is_permutation( vector( game.begin(), game.end() ),
                                    vector< uttt::Move >{ { 4, 0 },
                                                          { 4, 1 },
                                                          { 4, 2 },
                                                          { 4, 3 },
                                                          { 4, 5 },
                                                          { 4, 6 },
                                                          { 4, 7 },
                                                          { 4, 8 } } ) );

    game = game.apply( uttt::Move{ 4, 1 } );
    assert( game.current_player_index() == Player1 );
    assert( ranges::is_permutation( vector( game.begin(), game.end() ),
                                    vector< uttt::Move >{ { 1, 0 },
                                                          { 1, 1 },
                                                          { 1, 2 },
                                                          { 1, 3 },
                                                          { 1, 4 },
                                                          { 1, 5 },
                                                          { 1, 6 },
                                                          { 1, 7 },
                                                          { 1, 8 } } ) );
}

void uttt_human()
{
    if ( interactive )
        cout << source_location::current().function_name() << endl;
    else
    {
        cout << source_location::current().function_name()
             << " (interactive mode off)" << endl;
        return;
    }

    uttt::Game game( Player1, uttt::empty_state );

    uttt::console::HumanPlayer human( game );
    chrono::microseconds human_duration;

    uttt::minimax::Player player( game, 9.0, 5, seed );
    chrono::microseconds player_duration;

    UltimateTicTacToeMatch match( player );
    if ( GameResult result =
             match.play( game, human, human_duration, player, player_duration );
         result == GameResult::Player1Win )
        cout << "player 1 wins\n";
    else if ( result == GameResult::Player2Win )
        cout << "player 2 wins\n";
    else
        cout << "draw" << endl;
}

void uttt_match()
{
    if ( extensive )
        cout << source_location::current().function_name() << endl;
    else
    {
        cout << source_location::current().function_name()
             << " (extensive mode off)" << endl;
        return;
    }

    mt19937 g( seed );

    uttt::Game game( Player1, uttt::empty_state );

    PlayerFactory< uttt::Move, uttt::State > factory1 =
        []( uttt::Game const &game, unsigned seed )
    { return make_unique< uttt::minimax::Player >( game, 9.0, 1, seed ); };
    PlayerFactory< uttt::Move, uttt::State > factory2 =
        []( uttt::Game const &game, unsigned seed )
    { return make_unique< uttt::minimax::Player >( game, 9.0, 4, seed ); };
    MultiMatch match( game, factory1, factory2, 100, 1, seed );
    match.run();

    if ( verbose )
        cout << "fst player wins: " << match.get_fst_player_wins() << '\n'
             << "snd player wins: " << match.get_snd_player_wins() << '\n'
             << "draws: " << match.get_draws() << endl;

    assert( match.get_fst_player_wins() < 50 );
    assert( match.get_snd_player_wins() > 50 );
    assert( match.get_draws() > 0 );
}

void montecarlo_node()
{
    cout << source_location::current().function_name() << endl;

    vector< ttt::Move > move_stack;
    using Node = ttt::montecarlo::Node;
    using PreNode = ttt::montecarlo::PreNode;
    GenerationalArenaAllocator allocator( 50 * sizeof( Node ) );
    ttt::Game game( Player1, ttt::empty_state );

    using Payload = ttt::montecarlo::Payload;

    auto *node = new ( allocator.allocate< PreNode >() )
        PreNode( game, ttt::no_move, Payload{ .next_move_itr = game.begin() } );
    assert( node->get_children().size() == 0 );
    assert( node_count( *node ) == 1 );

    for ( auto const &move : game )
        node->get_children().push_front(
            *( new ( allocator.allocate< PreNode >() )
                   PreNode( game.apply( move ), move,
                            Payload{ .next_move_itr = game.begin() } ) ) );

    assert( node->get_children().size() == 9 );
    assert( node_count( *node ) == 10 );
}

void montecarlo_player()
{
    cout << source_location::current().function_name() << endl;

    using Node = ttt::montecarlo::Node;
    GenerationalArenaAllocator allocator( 50 * sizeof( Node ) );
    ttt::Game game( Player1, ttt::empty_state );
    ttt::montecarlo::Player player( game, 1.0, 2, seed, allocator );
    player.apply_opponent_move( ttt::Move( 4 ) );
    game = game.apply( ttt::Move( 4 ) );

    assert( player.root_node().get_move() == ttt::Move( 4 ) );
#ifdef DEBUG
    ttt::Move move = player.choose_move();
    vector< ttt::Move > valid_moves( game.begin(), game.end() );

    assert( std::find( valid_moves.begin(), valid_moves.end(), move ) !=
            valid_moves.end() );
#endif
}

void montecarlo_ttt_human()
{
    if ( interactive )
        cout << source_location::current().function_name() << endl;
    else
    {
        cout << source_location::current().function_name()
             << " (interactive mode off)" << endl;
        return;
    }

    ttt::Game game( Player1, ttt::empty_state );

    ttt::console::HumanPlayer human( game );
    chrono::microseconds human_duration;
    using Node = ttt::montecarlo::Node;
    GenerationalArenaAllocator allocator( 50 * sizeof( Node ) );
    ttt::montecarlo::Player player( game, 0.4f, 100, seed, allocator );
    chrono::microseconds player_duration;
    TicTacToeMatch match( player );
    if ( GameResult result =
             match.play( game, human, human_duration, player, player_duration );
         result == GameResult::Player1Win )
        cout << "player 1 wins\n";
    else if ( result == GameResult::Player2Win )
        cout << "player 2 wins\n";
    else
        cout << "draw" << endl;
}

void montecarlo_ttt_match()
{
    if ( extensive )
        cout << source_location::current().function_name() << endl;
    else
    {
        cout << source_location::current().function_name()
             << " (extensive mode off)" << endl;
        return;
    }

    using Node = ttt::montecarlo::Node;
    GenerationalArenaAllocator allocator( 50 * sizeof( Node ) );

    ttt::Game game( Player1, ttt::empty_state );

    const double exploration = 0.4;
    const size_t rounds = 100;
    ttt::PlayerFactory factory1 =
        [&allocator, exploration]( ttt::Game const &game, unsigned seed )
    {
        return make_unique< ttt::montecarlo::Player >( game, exploration, 100,
                                                       seed, allocator );
    };
    ttt::PlayerFactory factory2 =
        [&allocator, exploration]( ttt::Game const &game, unsigned seed )
    {
        return make_unique< ttt::montecarlo::Player >( game, exploration, 500,
                                                       seed, allocator );
    };
    MultiMatch match( game, factory1, factory2, rounds, 1, seed );
    match.run();

    if ( verbose )
        cout << "fst player wins: " << match.get_fst_player_wins() << '\n'
             << "snd player wins: " << match.get_snd_player_wins() << '\n'
             << "draws: " << match.get_draws() << endl;

    assert( match.get_draws() > 0 );
}

void montecarlo_minimax_ttt_match()
{
    if ( extensive )
        cout << source_location::current().function_name() << endl;
    else
    {
        cout << source_location::current().function_name()
             << " (extensive mode off)" << endl;
        return;
    }

    using Node = ttt::montecarlo::Node;
    GenerationalArenaAllocator allocator( 50 * sizeof( Node ) );

    ttt::Game game( Player1, ttt::empty_state );

    const double exploration = 0.4;
    const size_t rounds = 100;
    ttt::PlayerFactory factory1 =
        [&allocator, exploration]( ttt::Game const &game, unsigned seed )
    {
        return make_unique< ttt::montecarlo::Player >( game, exploration, 400,
                                                       seed, allocator );
    };
    ttt::PlayerFactory factory2 = []( ttt::Game const &game, unsigned seed )
    { return make_unique< ttt::minimax::Player >( game, 2, seed ); };
    MultiMatch match( game, factory1, factory2, rounds, 1, seed );
    match.run();

    if ( verbose )
        cout << "fst player wins: " << match.get_fst_player_wins() << '\n'
             << "snd player wins: " << match.get_snd_player_wins() << '\n'
             << "draws: " << match.get_draws() << endl;

    assert( match.get_draws() > 0 );
}

void montecarlo_minimax_uttt_match()
{
    if ( extensive )
        cout << source_location::current().function_name() << endl;
    else
    {
        cout << source_location::current().function_name()
             << " (extensive mode off)" << endl;
        return;
    }

    uttt::Game game( Player1, uttt::empty_state );

    using Node = uttt::montecarlo::Node;
    GenerationalArenaAllocator allocator( 50 * sizeof( Node ) );
    const double exploration = 0.4;
    size_t simulations = 3200;

    size_t depth = 6;
    const double factor = 9.0;

    const size_t rounds = 10;
    uttt::PlayerFactory factory1 = [exploration, &allocator, simulations](
                                       uttt::Game const &game, unsigned seed )
    {
        return make_unique< uttt::montecarlo::Player >(
            game, exploration, simulations, seed, allocator );
    };
    uttt::PlayerFactory factory2 =
        [factor, depth]( uttt::Game const &game, unsigned seed )
    {
        return make_unique< uttt::minimax::Player >( game, factor, depth,
                                                     seed );
    };
    MultiMatch match( game, factory1, factory2, rounds, 1, seed );
    match.run();

    if ( verbose )
        cout << "fst player wins: " << match.get_fst_player_wins() << '\n'
             << "snd player wins: " << match.get_snd_player_wins() << '\n'
             << "draws: " << match.get_draws() << '\n'
             << "fst player simulations: " << simulations << '\n'
             << "fst player exploration: " << exploration << '\n'
             << "fst player duration: " << match.get_fst_player_duration()
             << '\n'
             << "snd player depth: " << depth << '\n'
             << "snd player duration: " << match.get_snd_player_duration()
             << '\n'
             << "fst/snd player duration ratio: "
             << static_cast< double >(
                    chrono::duration_cast< std::chrono::microseconds >(
                        match.get_fst_player_duration() )
                        .count() ) /
                    static_cast< double >(
                        chrono::duration_cast< std::chrono::microseconds >(
                            match.get_snd_player_duration() )
                            .count() )
             << '\n';

    assert( match.get_draws() > 0 );
}

void uttt_match_mm_vs_tree_mm()
{
    if ( extensive )
        cout << source_location::current().function_name() << endl;
    else
    {
        cout << source_location::current().function_name()
             << " (extensive mode off)" << endl;
        return;
    }

    using Node = uttt::minimax::tree::Node;
    GenerationalArenaAllocator allocator( 50 * 100 * sizeof( Node ) );

    size_t fst_depth = 2;
    size_t snd_depth = 2;

    uttt::Game game( Player1, uttt::empty_state );

    uttt::PlayerFactory factory1 =
        [&fst_depth]( uttt::Game const &game, unsigned seed )
    {
        return make_unique< uttt::minimax::Player >( game, 9.0, fst_depth,
                                                     seed );
    };
    uttt::PlayerFactory factory2 =
        [snd_depth, &allocator]( uttt::Game const &game, unsigned seed )
    {
        return make_unique< uttt::minimax::tree::Player >( game, 9.0, snd_depth,
                                                           seed, allocator );
    };
    MultiMatch match( game, factory1, factory2, 100, 1, seed );
    match.run();

    if ( verbose )
        cout << "fst player wins: " << match.get_fst_player_wins() << '\n'
             << "snd player wins: " << match.get_snd_player_wins() << '\n'
             << "draws: " << match.get_draws() << '\n'
             << "fst player depth: " << fst_depth << '\n'
             << "fst player duration: " << match.get_fst_player_duration()
             << '\n'
             << "snd player depth: " << snd_depth << '\n'
             << "snd player duration: " << match.get_snd_player_duration()
             << '\n'
             << "fst/snd player duration ratio: "
             << static_cast< double >(
                    chrono::duration_cast< std::chrono::microseconds >(
                        match.get_fst_player_duration() )
                        .count() ) /
                    static_cast< double >(
                        chrono::duration_cast< std::chrono::microseconds >(
                            match.get_snd_player_duration() )
                            .count() )
             << '\n'
             << endl;

    assert( match.get_fst_player_wins() != 0 );
    assert( match.get_snd_player_wins() != 0 );
    assert( match.get_draws() > 0 );
}

vector< ttt::alphazero::training::Position >
selfplay_worker( ttt::alphazero::libtorch::InferenceService &inference_manager,
                 libtorch::Hyperparameters const &hp, size_t runs_per_thread,
                 size_t parallel_simulations )
{
    auto g = mt19937( random_device{}() );
    GenerationalArenaAllocator allocator( 50 * hp.simulations *
                                          sizeof( ttt::alphazero::Node ) );
    vector< ttt::alphazero::training::Position > positions;
    PlayerIndex player_index = PlayerIndex::Player1;
    Statistics root_node_entropy_stat;
    Statistics informed_selection_stat;
    for ( ; runs_per_thread; --runs_per_thread )
    {
        alphazero::params::Ucb ucb_params{ .c_base = hp.c_base,
                                           .c_init = hp.c_init };
        alphazero::params::GamePlay gameplay_params{
            .simulations = hp.simulations,
            .opening_moves = hp.opening_moves,
            .parallel_simulations = parallel_simulations,
        };
        ttt::alphazero::Player player(
            ttt::Game( player_index, ttt::empty_state ), ucb_params,
            gameplay_params, seed, allocator, inference_manager );
        alphazero::training::SelfPlay self_play(
            player, hp.dirichlet_alpha, hp.dirichlet_epsilon, g, positions,
            root_node_entropy_stat, informed_selection_stat );
        self_play.run();
        player_index = toggle( player_index );
    }

    return positions;
}

void alphazero_training()
{
    if ( extensive )
        cout << source_location::current().function_name() << endl;
    else
    {
        cout << source_location::current().function_name()
             << " (extensive mode off)" << endl;
        return;
    }

    torch::Device device = libtorch::get_device();
    // Adjust if needed
    const char *const model_path =
        "runs/models/ttt_alphazero_experiment_6/final_model.pt";
    auto [model, hp] = libtorch::load_model( model_path, device );
    const size_t parallel_simulations = 10;
    ttt::alphazero::libtorch::InferenceService inference_service(
        std::move( model ), device, hp.max_batch_size );

    vector< future< vector< ttt::alphazero::training::Position > > >
        thread_pool( 8 );
    cout << "start " << thread_pool.size() << " worker threads" << endl;
    for ( auto &future : thread_pool )
        future = async( selfplay_worker, ref( inference_service ), hp, 500,
                        parallel_simulations );

    cout << "wait for all threads to finish..." << endl;
    size_t total_positions = 0;
    for ( auto &future : thread_pool )
    {
        auto positions = future.get();
        total_positions += positions.size();
    }
    cout << "total positions: " << total_positions << endl;
}

struct WorkerResult
{
    vector< uttt::alphazero::training::Position > positions;
    Statistics root_node_entropy_stats;
    Statistics informed_selection_stats;
    size_t allocator_offset;
    size_t allocated_blocks;
};

WorkerResult uttt_selfplay_worker(
    uttt::alphazero::libtorch::InferenceService &inference_service,
    libtorch::Hyperparameters const &hp, size_t parallel_simulations,
    size_t runs_per_thread, size_t simulations, unsigned local_seed,
    size_t allocator_block_size )
{
    auto g = mt19937( local_seed );
    GenerationalArenaAllocator allocator( allocator_block_size );
    WorkerResult result;
    PlayerIndex player_index = PlayerIndex::Player1;
    for ( ; runs_per_thread; --runs_per_thread )
    {
        alphazero::params::Ucb ucb_params{ .c_base = hp.c_base,
                                           .c_init = hp.c_init };
        alphazero::params::GamePlay gameplay_params{
            .simulations = simulations,
            .opening_moves = hp.opening_moves,
            .parallel_simulations = parallel_simulations };
        uttt::alphazero::Player player(
            uttt::Game( player_index, uttt::empty_state ), ucb_params,
            gameplay_params, g(), allocator, inference_service );
        alphazero::training::SelfPlay self_play(
            player, hp.dirichlet_alpha, hp.dirichlet_epsilon, g,
            result.positions, result.root_node_entropy_stats,
            result.informed_selection_stats );
        self_play.run();
        allocator.reset();
        player_index = toggle( player_index );
    }

    result.allocated_blocks =
        allocator.get_fst_arena_allocator().allocated_blocks() +
        allocator.get_snd_arena_allocator().allocated_blocks();
    result.allocator_offset =
        allocator.get_fst_arena_allocator().get_current_offset() +
        allocator.get_snd_arena_allocator().get_current_offset();
    return result;
}

void uttt_alphazero_training()
{
    if ( extensive )
        cout << source_location::current().function_name() << endl;
    else
    {
        cout << source_location::current().function_name()
             << " (extensive mode off)" << endl;
        return;
    }

    torch::Device device = libtorch::get_device();
    const char *const model_path = "models/test/model_31000.pt";
    auto [model, hp] = libtorch::load_model( model_path, device );
    size_t simulations = 10; // 80;
    const size_t worker_threads = 1;
    const size_t selfplay_threads = 8;
    const size_t max_batch_size = 64; // 320;
    uttt::alphazero::libtorch::InferenceService inference_service(
        std::move( model ), device, max_batch_size );
    vector< future< WorkerResult > > thread_pool( worker_threads );
    const size_t number_of_games = 8;
    const size_t runs_per_worker_thread = number_of_games / worker_threads;
    using pre_node_type =
        PreNode< uttt::Move, uttt::State, uttt::alphazero::Payload >;
    unsigned local_seed = seed;
    size_t rounds = 7;
    ostringstream oss;
    oss << "games;parallelGames;parallelSims;maxBatchSize;batchMean;"
           "batchStddef;"
           "timeMean;timeStddef;Entr;InfSel;DurPerPos;Sims;Dur\n";
    for ( size_t i = rounds; i; --i )
    {
        const size_t allocator_block_size =
            50 * simulations * sizeof( pre_node_type );
        cout << "worker threads: " << thread_pool.size() << "\n"
             << "selfplay threads: " << selfplay_threads << "\n"
             << "max batch size: " << max_batch_size << "\n"
             << "simulations: " << simulations << "\n"
             << "games: " << number_of_games << "\n"
             << "allocator block size: " << allocator_block_size << "\n"
             << endl;

        auto start = std::chrono::steady_clock::now();
        for ( auto &future : thread_pool )
        {
            future = async( uttt_selfplay_worker, ref( inference_service ), hp,
                            selfplay_threads, runs_per_worker_thread,
                            simulations, local_seed, allocator_block_size );
            ++local_seed;
        }
        cout << "wait for all threads to finish..." << endl;
        size_t total_positions = 0;
        Statistics root_node_entropy_stats;
        Statistics informed_selection_stats;
        size_t allocator_offset = 0;
        size_t allocated_blocks = 0;
        for ( auto &future : thread_pool )
        {
            auto result = future.get();
            total_positions += result.positions.size();
            root_node_entropy_stats.join( result.root_node_entropy_stats );
            informed_selection_stats.join( result.informed_selection_stats );
            allocator_offset += result.allocator_offset;
            allocated_blocks += result.allocated_blocks;
        }
        const std::chrono::duration< float > duration =
            std::chrono::steady_clock::now() - start;

        cout << "total positions: " << total_positions << endl
             << "inference manager batch size stats:\n"
             << inference_service.batch_size_stats() << '\n'
             << "inference manager time stats:\n"
             << inference_service.inference_time_stats() << '\n'
             << "root node entropy stats:\n"
             << root_node_entropy_stats << '\n'
             << "informed selection stats:\n"
             << informed_selection_stats << '\n'
             << "allocator offset:\n"
             << allocator_offset << '\n'
             << "allocated blocks: " << allocated_blocks << '\n'
             << "selfplay run duration: " << duration << '\n'
             << endl;

        oss << number_of_games << ";" << thread_pool.size() << ";"
            << selfplay_threads << ";" << max_batch_size << ";"
            << inference_service.batch_size_stats().mean() << ";"
            << inference_service.batch_size_stats().stddev() << ";"
            << inference_service.inference_time_stats().mean() << ";"
            << inference_service.inference_time_stats().stddev() << ";"
            << root_node_entropy_stats.mean() << ";"
            << informed_selection_stats.mean() << ";"
            << static_cast< float >( duration.count() ) / total_positions << ";"
            << simulations << ";" << duration << endl;

        simulations *= 2;
    }
    cout << oss.str() << endl;
}

/*
run tests with seed 1392513404

uttt_alphazero_training
start 10 worker threads
wait for all threads to finish...
selfplay run duration for 0x16b6eb000: 142906320µs
selfplay run duration for 0x16b65f000: 147978799µs
selfplay run duration for 0x16b777000: 157836637µs
selfplay run duration for 0x16babf000: 162950441µs
selfplay run duration for 0x16b91b000: 164458507µs
selfplay run duration for 0x16b9a7000: 166672635µs
selfplay run duration for 0x16b803000: 166960549µs
selfplay run duration for 0x16b88f000: 168363974µs
selfplay run duration for 0x16b5d3000: 176948216µs
selfplay run duration for 0x16ba33000: 180757133µs
total positions: 573
inference manager queue size stats:
mean = 3.62442, stddev = 2.31959
min = 1, max = 10
count = 55964

inference manager time stats:
mean = 1049.96, stddev = 936.701
min = 183, max = 102504
count = 55964

everything ok
obj/test  196,74s user 28,04s system 123% cpu 3:01,46 total
*/

template < typename MoveT, typename StateT >
class LogMultiMatch : public MultiMatch< MoveT, StateT >
{
  public:
    LogMultiMatch( Game< MoveT, StateT > const &game,
                   PlayerFactory< MoveT, StateT > fst_player_factory,
                   PlayerFactory< MoveT, StateT > snd_player_factory,
                   int rounds, size_t number_of_threads, unsigned seed )
        : MultiMatch< MoveT, StateT >( game, fst_player_factory,
                                       snd_player_factory, rounds,
                                       number_of_threads, seed )
    {
    }

  private:
    map< thread::id, vector< pair< PlayerIndex, MoveT > > > games;
    mutex games_mutex;
    mutex cout_mutex;
    void report( Game< MoveT, StateT > const &game, MoveT const &move ) override
    {
        vector< pair< PlayerIndex, MoveT > > *moves = nullptr;
        {
            // only get/insert of new thread id has to be synchronized
            scoped_lock lock( games_mutex );
            moves = &games[this_thread::get_id()];
        }

        moves->push_back( make_pair( game.current_player_index(), move ) );
        if ( game.result() != GameResult::Undecided )
        {
            {
                // print game moves and result to console
                scoped_lock lock( cout_mutex );
                for ( auto const &[player_index, m] : *moves )
                    // toggle the player index because it's the opponent's move
                    cout << uttt::PlayerIndexDispatch( toggle( player_index ) )
                         << ":" << uttt::MoveDispatch( m ) << ", " << flush;
                cout << "[" << moves->size() << "] -> " << game.result() << "\n"
                     << endl;
            }
            // clear moves container for next game
            scoped_lock lock( games_mutex );
            games[this_thread::get_id()].clear();
        }
    }
};

void uttt_alphazero_nn_vs_minimax()
{
    if ( extensive )
        cout << source_location::current().function_name() << endl;
    else
    {
        cout << source_location::current().function_name()
             << " (extensive mode off)" << endl;
        return;
    }

    torch::Device device = libtorch::get_device();
    const char *const model_path = "models/test/checkpoint.pt";
    cout << "load model " << model_path << " to device " << device << endl;
    auto [model, hp] = libtorch::load_model( model_path, device );
    const size_t threads = 10;
    uttt::alphazero::libtorch::InferenceService inference_service(
        std::move( model ), device, hp.max_batch_size );

    uttt::Game game( Player1, uttt::empty_state );
    GenerationalArenaAllocator allocator( 50 * hp.simulations *
                                          sizeof( uttt::alphazero::Node ) );

    const size_t rounds = 20;
    uttt::PlayerFactory factory1 = [&hp, &allocator, &inference_service](
                                       uttt::Game const &game, unsigned seed )
    {
        const size_t simulations = 800;
        const size_t parallel_simulations = 8;
        alphazero::params::Ucb ucb_params{ .c_base = hp.c_base,
                                           .c_init = hp.c_init };
        alphazero::params::GamePlay gameplay_params{
            .simulations = simulations,
            .opening_moves = hp.opening_moves,
            .parallel_simulations = parallel_simulations };
        return make_unique< uttt::alphazero::Player >(
            game, ucb_params, gameplay_params, seed, allocator,
            inference_service );
    };
    uttt::PlayerFactory factory2 = []( uttt::Game const &game, unsigned seed )
    { return make_unique< uttt::minimax::Player >( game, 9.0, 3, seed ); };
    LogMultiMatch match( game, factory1, factory2, rounds, threads, seed );
    match.run();

    if ( verbose )
        cout << "rounds = " << rounds << ", threads = " << threads << '\n'
             << "fst player wins: " << match.get_fst_player_wins() << '\n'
             << "fst player duration: " << match.get_fst_player_duration()
             << '\n'
             << "inference service queue size stats:\n"
             << inference_service.batch_size_stats() << '\n'
             << "inference service time stats:\n"
             << inference_service.inference_time_stats() << '\n'
             << "snd player wins: " << match.get_snd_player_wins() << '\n'
             << "snd player duration: " << match.get_snd_player_duration()
             << '\n'
             << "draws: " << match.get_draws() << '\n'
             << "fst/snd player duration ratio: "
             << static_cast< double >(
                    chrono::duration_cast< std::chrono::microseconds >(
                        match.get_fst_player_duration() )
                        .count() ) /
                    static_cast< double >(
                        chrono::duration_cast< std::chrono::microseconds >(
                            match.get_snd_player_duration() )
                            .count() )
             << '\n'
             << endl;
}

void uttt_alphazero_nn_vs_alphazero()
{
    if ( extensive )
        cout << std::source_location::current().function_name() << endl;
    else
    {
        cout << std::source_location::current().function_name()
             << " (extensive mode off)" << endl;
        return;
    }

    torch::Device device = libtorch::get_device();
    const char *const model_path = "models/test/model_31000.pt";
    const char *const model_path2 = "models/test/checkpoint.pt";
    cout << "load models for player1 " << model_path << " and player2"
         << model_path2 << " to device " << device << endl;
    auto [model, hp] = libtorch::load_model( model_path, device );
    auto [model2, hp2] = libtorch::load_model( model_path2, device );
    const size_t threads = 10;
    uttt::alphazero::libtorch::InferenceService inference_service(
        std::move( model ), device, hp.max_batch_size );
    uttt::alphazero::libtorch::InferenceService inference_service2(
        std::move( model2 ), device, hp.max_batch_size );

    uttt::Game game( Player1, uttt::empty_state );

    const size_t rounds = 100;
    const size_t parallel_simulations = 10;
    uttt::PlayerFactory factory1 =
        [&hp, &inference_service]( uttt::Game const &game, unsigned seed )
    {
        static thread_local GenerationalArenaAllocator allocator(
            50 * hp.simulations * sizeof( uttt::alphazero::Node ) );
        alphazero::params::Ucb ucb_params{ .c_base = hp.c_base,
                                           .c_init = hp.c_init };
        alphazero::params::GamePlay gameplay_params{
            .simulations = 400,
            .opening_moves = hp.opening_moves,
            .parallel_simulations = parallel_simulations };
        return make_unique< uttt::alphazero::Player >(
            game, ucb_params, gameplay_params, seed, allocator,
            inference_service );
    };
    uttt::PlayerFactory factory2 =
        [&hp2, &inference_service2]( uttt::Game const &game, unsigned seed )
    {
        static thread_local GenerationalArenaAllocator allocator(
            50 * hp2.simulations * sizeof( uttt::alphazero::Node ) );
        alphazero::params::Ucb ucb_params{ .c_base = hp2.c_base,
                                           .c_init = hp2.c_init };
        alphazero::params::GamePlay gameplay_params{
            .simulations = 400,
            .opening_moves = hp2.opening_moves,
            .parallel_simulations = parallel_simulations };
        return make_unique< uttt::alphazero::Player >(
            game, ucb_params, gameplay_params, seed, allocator,
            inference_service2 );
    };

    LogMultiMatch match( game, factory1, factory2, rounds, threads, seed );
    match.run();

    if ( verbose )
        cout << "rounds = " << rounds << ", threads = " << threads << '\n'
             << "fst player wins: " << match.get_fst_player_wins() << '\n'
             << "fst player duration: " << match.get_fst_player_duration()
             << '\n'
             << "inference service queue size stats:\n"
             << inference_service.batch_size_stats() << '\n'
             << "inference service time stats:\n"
             << inference_service.inference_time_stats() << '\n'
             << "snd player wins: " << match.get_snd_player_wins() << '\n'
             << "snd player duration: " << match.get_snd_player_duration()
             << '\n'
             << "draws: " << match.get_draws() << '\n'
             << "fst/snd player duration ratio: "
             << static_cast< double >(
                    chrono::duration_cast< std::chrono::microseconds >(
                        match.get_fst_player_duration() )
                        .count() ) /
                    static_cast< double >(
                        chrono::duration_cast< std::chrono::microseconds >(
                            match.get_snd_player_duration() )
                            .count() )
             << '\n'
             << endl;
}

} // namespace test

int main()
{
    try
    {
        cout << "run tests with seed " << seed << endl << endl;
        test::uttt_alphazero_nn_vs_alphazero();
        //        test::uttt_alphazero_training();
        return 0;
    }
    catch ( source_location const &e )
    {
        cout << "exception caught: " << e.file_name() << ": "
             << e.function_name() << ": " << e.line() << endl;
    }

    catch ( exception const &e )
    {
        cout << "exception caught: " << e.what() << endl;
    }

    return -1;
}