#include <iostream>
#include <exception>
#include <cassert>
#include <fstream>

#include "games/nim.h"
#include "games/ultimate_ttt.h"
#include "match.h"
#include "montecarlo.h"
#include "minimax-tree.h"
#include "alphazero.h"
#include "libtorch_util.h"
#include <boost/json.hpp>

using namespace std;

unsigned seed = 0;
bool verbose = true;
bool interactive = false;
bool extensive = true;

template<>
struct GameState< char, GameResult > 
    : public GameStateBase<GameState<char, GameResult>, char, GameResult>
{
    static void next_valid_move( 
        optional< char >& move, PlayerIndex, GameResult const& )
    {
        if (!move)
            move = 'a'; 
        else if (move == 'a')
            move = 'b'; 
        else
            move.reset();
    }

    // get_valid_moves will be inherited from GameStateDefaultGetValidMovesProvider

    static GameResult apply( char const& move, PlayerIndex, GameResult const& state )
    {
        return state;
    }

    static GameResult result( PlayerIndex player_index, GameResult const& state )
    {
        return state;
    }
};

namespace test {

void toggle_player()
{
    cout << __func__ << endl;
        
    if( toggle( Player1 ) != Player2 ||
        toggle( Player2 ) != Player1 )
        assert( false );
}

using TestGame = Game< char, GameResult >;

struct TestPlayer : public Player< char >
{
    char choose_move() override
    {
        return 0;
    }
};

void build_game()
{
    cout << __func__ << endl;

    TestGame game( Player2, GameResult::Undecided );
    std::mt19937 g;
    minimax::Data< char > data( g );
    minimax::Player< char, GameResult > player( game, 0, data );

    assert( game.current_player_index() == Player2);
    if (game.current_player_index() != Player2)
        assert( !"invalid current player index" );
}

void eval_won_game()
{
    cout << __func__ << endl;
    mt19937 g;
    vector< char > move_stack;
    minimax::ScoreFunction< char, GameResult > score = []( Game< char, GameResult > const& ) 
        { return 0.0; };
    size_t calls = 0;
    if (minimax::eval< char >( TestGame( 
            Player2, GameResult::Player2Win ), 
            score, 0, 0.0, 1.0, g, calls ) != INFINITY)
        assert( !"wrong score for won game" );
    if (minimax::eval< char >( TestGame( 
            Player2, GameResult::Player1Win ), 
            score, 0, 0.0, 1.0, g, calls ) != -INFINITY)
        assert( !"wrong score for won game" );
}

void eval_drawn_game()
{
    cout << __func__ << endl;
    mt19937 g;
    vector< char > move_stack;
    
    minimax::ScoreFunction< char, GameResult > score = []( TestGame const& ) 
        { return 0.0; };
    size_t calls = 0;
    if (minimax::eval< char >( TestGame( 
        Player2, GameResult::Draw ), score, 
        0, 0.0, 1.0, g, calls ) != 0.0)
        assert( !"wrong score for drawn game" );
}

void eval_undecided_game()
{
    cout << __func__ << endl;

    TestGame undecided_game( Player2, GameResult::Undecided );

    minimax::ScoreFunction< char, GameResult > score = []( TestGame const& ) 
        { return 42.0; };
    mt19937 g;
    vector< char > move_stack;
    size_t calls = 0;

    if (minimax::eval( undecided_game, score, 0, 0.0, 1.0, g, calls ) != 42.0)
        assert( !"wrong score for undecided game" );

}

struct TestNimPlayer : public Player< nim::Move >
{
    nim::Move choose_move() override
    {
        return next_move;
    }

    void apply_opponent_move( nim::Move const& ) override
    {
        // do nothing
    }

    nim::Move next_move;
};

void nim_game()
{
    cout << __func__ << endl;

    nim::Game< 2 > game( Player1, array< size_t, 2 >{ 1, 2 } );

    {
        auto moves = vector{ nim::Move{ 0, 1 }, nim::Move{ 1, 1 }, nim::Move{ 1, 2 } };
        auto valid_move = game.begin();
        assert (std::find(moves.begin(), moves.end(), *valid_move)
            != moves.end());
        moves.erase( remove( moves.begin(), moves.end(), *valid_move ));
        ++valid_move;

        assert (std::find(moves.begin(), moves.end(), *valid_move)
            != moves.end());
        moves.erase( remove( moves.begin(), moves.end(), *valid_move ));
        ++valid_move;

        assert (ranges::contains(moves.begin(), moves.end(), *valid_move));
        moves.erase( remove( moves.begin(), moves.end(), *valid_move ));
        valid_move++; // try post-increment

        assert( valid_move == game.end());
        assert (moves.empty());
    }
    
    game = game.apply( nim::Move{ 1, 1 } );    
    assert( game.current_player_index() == Player2 );
    assert (ranges::is_permutation(
        vector{ nim::Move{ 0, 1 }, nim::Move{ 1, 1 }}, 
        vector< nim::Move >( game.begin(), game.end())));

    game = game.apply( nim::Move{ 1, 1 } );
    assert( game.current_player_index() == Player1 );
    assert( ranges::is_permutation( 
        vector{ nim::Move{ 0, 1 }}, 
        vector< nim::Move >( game.begin(), game.end())));
    
    game = game.apply( nim::Move{ 0, 1 } );
    assert( game.current_player_index() == Player2 );
    assert (game.result() == GameResult::Player2Win);
}

void nim_match()
{
    if (extensive)
        cout << __func__ << endl;
    else
    {
        cout << __func__ << " (extensive mode off)" << endl;
        return;
    }

    const size_t HEAPS = 5;

    mt19937 g( seed );
    nim::minimax::Data< HEAPS > data1( g );
    nim::minimax::Data< HEAPS > data2( g );

    nim::Game< HEAPS > game( Player1, { 1, 2, 3, 4, 5 } );

    MultiMatch< nim::Move, nim::State< HEAPS > > match;
    match.play_match( 
        game, 
        [&game, &data1]() { return new nim::minimax::Player< HEAPS >( game, 2, data1 ); }, 
        [&game, &data2]() { return new nim::minimax::Player< HEAPS >( game, 3, data2 ); }, 
        100 );

    if (verbose)
        cout 
            << "fst player wins: " << match.fst_player_wins << '\n'
            << "snd player wins: " << match.snd_player_wins << '\n'
            << "draws: " << match.draws << '\n'
            << "fst player move stack capacity: " << data1.move_stack.capacity() << '\n'
            << "fst player eval calls: " << data1.eval_calls << '\n'
            << "snd player move stack capacity: " << data2.move_stack.capacity() << '\n'
            << "snd player eval calls: " << data2.eval_calls << endl;

    assert (match.fst_player_wins < 50);
    assert (match.snd_player_wins > 50);
    assert (match.draws == 0);
    assert (data1.move_stack.capacity() == 16);
    assert (data2.move_stack.capacity() == 16);
    assert (data1.eval_calls > 50000 && data1.eval_calls < 100000);
    assert (data2.eval_calls > 200000 && data2.eval_calls < 400000);
}

void ttt_game()
{
    cout << __func__ << endl;

    ttt::Game game( Player1, ttt::empty_state );
    
    assert (game.result() == GameResult::Undecided);
    
    assert (ranges::is_permutation( 
                vector< ttt::Move >( game.begin(), game.end() ), 
                vector{ 0, 1, 2, 3, 4, 5, 6, 7, 8 }));
    game = game.apply( ttt::Move( 4 ) );
    assert( game.current_player_index() == Player2 );

    vector< ttt::Move > moves { 0, 1, 2, 3, 5, 6, 7, 8 };
    assert (ranges::is_permutation( 
        vector< ttt::Move >( game.begin(), game.end() ), moves ));

    auto valid_move = game.begin();
    assert (*valid_move == ttt::Move( 0 ));
    assert (ranges::contains(moves.begin(), moves.end(), *valid_move ));
    moves.erase( remove( moves.begin(), moves.end(), *valid_move ));
    ++valid_move;

    assert (*valid_move == ttt::Move( 1 ));
    assert (ranges::contains(moves.begin(), moves.end(), *valid_move ));
    moves.erase( remove( moves.begin(), moves.end(), *valid_move ));
    
    auto itr = valid_move++; // try post-increment
    assert (*itr == ttt::Move( 1 ));
    assert (*valid_move == ttt::Move( 2 ));
    
    assert (*game.begin() == ttt::Move( 0));

    game = game.apply( ttt::Move( 0 ));    
    assert( game.current_player_index() == Player1 );
    assert (ranges::is_permutation(
        vector( game.begin(), game.end()),
        vector{ 1, 2, 3, 5, 6, 7, 8 }));

    game = game.apply( ttt::Move( 7 ));    
    assert( game.current_player_index() == Player2 );
    assert (ranges::is_permutation(
        vector( game.begin(), game.end()),
        vector{ 1, 2, 3, 5, 6, 8 }));

    game = game.apply( ttt::Move( 2 ));    
    assert( game.current_player_index() == Player1 );
    assert (ranges::is_permutation(
        vector( game.begin(), game.end()),
        vector{ 1, 3, 5, 6, 8 }));

    game = game.apply( ttt::Move( 8 ));    
    assert( game.current_player_index() == Player2 );
    assert (ranges::is_permutation(
        vector( game.begin(), game.end()),
        vector{ 1, 3, 5, 6 }));

    assert (game.result() == GameResult::Undecided);
    game = game.apply( ttt::Move( 1 ));    
    assert( game.current_player_index() == Player1 );

    assert (game.result() == GameResult::Player2Win);
}

struct TicTacToeMatch : public Match< ttt::Move, ttt::State >
{
    TicTacToeMatch( Player< ttt::Move > const& player)
        : player( player ) {}

    Player< ttt::Move > const& player;

    void report( ttt::Game const& game, ttt::Move const& move ) override
    {
        cout 
            << "player " << game.current_player_index() << " (" 
            << (int)move << ")\n"
            << "resulting board:\n" << game << '\n';
        if (auto p = dynamic_cast< const ttt::minimax::Player* >( &player ))
            cout << "score: " << p->score( game ) << endl;
        else if (auto p = dynamic_cast< const ttt::montecarlo::Player* >( &player ))
            cout << "point ratio: " 
                 << p->root_node().get_value().points / p->root_node().get_value().visits 
                 << endl;
    }
};

void ttt_human()
{
    if (interactive)
        cout << __func__ << endl;
    else
    {
        cout << __func__ << " (interactive mode off)" << endl;
        return;
    }

    ttt::Game game( Player1, ttt::empty_state );

    ttt::console::HumanPlayer human( game );
    chrono::microseconds human_duration;
    mt19937 g( seed );
    ttt::minimax::Data data( g );
    ttt::minimax::Player player( game, 0, data );
    chrono::microseconds player_duration;
    TicTacToeMatch match( player);
    if (GameResult result = match.play( game, human, human_duration, 
                                        player, player_duration ); 
        result == GameResult::Player1Win)
        cout << "player 1 wins\n";
    else if (result == GameResult::Player2Win)
        cout << "player 2 wins\n";
    else
        cout << "draw\n";
    cout << '\n'
        << "player move stack capacity: " << data.move_stack.capacity() << '\n'
        << "player eval calls: " << data.eval_calls << endl;
}

void tic_tac_toe_match()
{
    if (extensive)
        cout << __func__ << endl;
    else
    {
        cout << __func__ << " (extensive mode off)" << endl;
        return;
    }

    mt19937 g( seed );

    ttt::minimax::Data data1( g );
    ttt::minimax::Data data2( g );

    ttt::Game game( Player1, ttt::empty_state );
    
    MultiMatch< ttt::Move, ttt::State > match;
    match.play_match( 
        game, 
        [&game, &data1]() { return new ttt::minimax::Player( game, 0, data1 ); }, 
        [&game, &data2]() { return new ttt::minimax::Player( game, 5, data2 ); }, 
        100 );

    if (verbose)
        cout 
            << "fst player wins: " << match.fst_player_wins << '\n'
            << "snd player wins: " << match.snd_player_wins << '\n'
            << "draws: " << match.draws << '\n'
            << "fst player move stack capacity: " << data1.move_stack.capacity() << '\n'
            << "fst player eval calls: " << data1.eval_calls << '\n'
            << "snd player move stack capacity: " << data2.move_stack.capacity() << '\n'
            << "snd player eval calls: " << data2.eval_calls << endl;

    assert (match.fst_player_wins == 0);
    assert (match.draws > 50);
    assert (data1.move_stack.capacity() == 9);
    assert (data2.move_stack.capacity() == 9);
    assert (data1.eval_calls > 1000 && data1.eval_calls < 3000);
    assert (data2.eval_calls > 300000 && data2.eval_calls < 800000);
}

struct UltimateTicTacToeMatch : public Match< uttt::Move, uttt::State >
{
    UltimateTicTacToeMatch( 
        uttt::minimax::Player const& minimax_player,
        uttt::minimax::Data const& data )
        : minimax_player( minimax_player ), data( data ) {}

    uttt::minimax::Player const& minimax_player;
    uttt::minimax::Data const& data;

    void report( uttt::Game const& game, uttt::Move const& move ) override
    {
        cout 
            << "player " << game.current_player_index() << " (" 
            << (int)move.big_move << "," << (int)move.small_move << ")\n"
            << "resulting board:\n" << game << '\n'
            << "score: " << minimax_player.score( game ) << '\n'
            << "best score: " << data.best_score << endl;
    }
};

void uttt_game()
{
    cout << __func__ << endl;

    uttt::Game game( Player1, uttt::empty_state );
    
    assert (game.result() == GameResult::Undecided);
    
    assert (vector( game.begin(), game.end()).size() == 81);

    game = game.apply( uttt::Move( 4, 4 ) );
    assert( game.current_player_index() == Player2 );
    assert (ranges::is_permutation(
        vector( game.begin(), game.end()),
        vector< uttt::Move >{ {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 5}, {4, 6}, {4, 7}, {4, 8} }));

    game = game.apply( uttt::Move( 4, 1 ) );
    assert( game.current_player_index() == Player1 );
    assert (ranges::is_permutation(
        vector( game.begin(), game.end()),
        vector< uttt::Move >{ {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8} }));
}

void uttt_human()
{
    if (interactive)
        cout << __func__ << endl;
    else
    {
        cout << __func__ << " (interactive mode off)" << endl;
        return;
    }

    uttt::Game game( Player1, uttt::empty_state );

    uttt::console::HumanPlayer human( game );
    chrono::microseconds human_duration;

    mt19937 g( seed );
    uttt::minimax::Data data( g );

    uttt::minimax::Player player( game, 9.0, 5, data );
    chrono::microseconds player_duration;

    UltimateTicTacToeMatch match( player, data );
    if (GameResult result = match.play( game, human, human_duration, 
                                        player, player_duration ); 
        result == GameResult::Player1Win)
        cout << "player 1 wins\n";
    else if (result == GameResult::Player2Win)
        cout << "player 2 wins\n";
    else
        cout << "draw\n";
    cout << '\n'
        << "player move stack capacity: " << data.move_stack.capacity() << '\n'
        << "player eval calls: " << data.eval_calls << endl;
}

void uttt_match()
{
    if (extensive)
        cout << __func__ << endl;
    else
    {
        cout << __func__ << " (extensive mode off)" << endl;
        return;
    }

    mt19937 g( seed );
    uttt::minimax::Data data1( g );
    uttt::minimax::Data data2( g );

    uttt::Game game( Player1, uttt::empty_state );

    MultiMatch< uttt::Move, uttt::State > match;
    match.play_match( 
        game, 
        [&game, &data1]() { return new uttt::minimax::Player( game, 9.0, 1, data1 ); }, 
        [&game, &data2]() { return new uttt::minimax::Player( game, 9.0, 4, data2 ); }, 
        100 );

    if (verbose)
        cout 
            << "fst player wins: " << match.fst_player_wins << '\n'
            << "snd player wins: " << match.snd_player_wins << '\n'
            << "draws: " << match.draws << '\n'
            << "fst player move stack capacity: " << data1.move_stack.capacity() << '\n'
            << "fst player eval calls: " << data1.eval_calls << '\n'
            << "snd player move stack capacity: " << data2.move_stack.capacity() << '\n'
            << "snd player eval calls: " << data2.eval_calls << endl;

    assert (match.fst_player_wins < 50);
    assert (match.snd_player_wins > 50);
    assert (match.draws > 0);
    assert (data1.move_stack.capacity() == 81);
    assert (data2.move_stack.capacity() == 81);
    assert (data1.eval_calls > 100000 && data1.eval_calls < 300000);
    assert (data2.eval_calls > 12000000 && data2.eval_calls < 15000000);
}

void montecarlo_node()
{
    cout << __func__ << endl;

    vector< ttt::Move > move_stack;
    ttt::montecarlo::NodeAllocator allocator;
    ttt::Game game( Player1, ttt::empty_state );

    using Value = montecarlo::detail::Value< ttt::Move, ttt::State >;

    Node< Value >* node = new (allocator.allocate(1)) Node< Value >( 
                     Value( game, ttt::no_move ), allocator );
    assert (node->get_children().size() == 0);
    assert (node_count( *node) == 1);

    for (auto const& move : game)
        node->get_children().push_front( 
            *(new (allocator.allocate(1)) Node< Value >( 
                Value( game.apply( move ), move), allocator )));

    assert (node->get_children().size() == 9);
    assert (node_count( *node) == 10);
    node->~Node();
    allocator.deallocate( node, 1);
}

void montecarlo_player()
{
    cout << __func__ << endl;

    mt19937 g( seed );
    ttt::montecarlo::NodeAllocator allocator;
    ttt::montecarlo::Data data( g, allocator );
    ttt::Game game( Player1, ttt::empty_state );
    ttt::montecarlo::Player player( game, 1.0, 2, data );
    player.apply_opponent_move( ttt::Move( 4 ) );
    game = game.apply( ttt::Move( 4 ) );

    using Value = montecarlo::detail::Value< ttt::Move, ttt::State >;
    
    Node< Value > const& root = player.root_node();
    assert (root.get_value().move == ttt::Move( 4 ));
    ttt::Move move = player.choose_move();
    vector< ttt::Move > valid_moves( game.begin(), game.end() );

    assert (std::find(valid_moves.begin(), valid_moves.end(), move)
        != valid_moves.end());
}

void montecarlo_ttt_human()
{
    if (interactive)
        cout << __func__ << endl;
    else
    {
        cout << __func__ << " (interactive mode off)" << endl;
        return;
    }

    ttt::Game game( Player1, ttt::empty_state );

    ttt::console::HumanPlayer human( game );
    chrono::microseconds human_duration;
    mt19937 g( seed );
    ttt::montecarlo::NodeAllocator allocator;
    ttt::montecarlo::Data data( g, allocator );
    ttt::montecarlo::Player player( game, 0.4, 100, data );
    chrono::microseconds player_duration;
    TicTacToeMatch match( player );
    if (GameResult result = match.play( game, human, human_duration, 
                                        player, player_duration ); 
        result == GameResult::Player1Win)
        cout << "player 1 wins\n";
    else if (result == GameResult::Player2Win)
        cout << "player 2 wins\n";
    else
        cout << "draw\n";
    cout << '\n'
        << "player move stack capacity: " << data.move_stack.capacity() << '\n'
        << "player playout count: " << data.playout_count << endl;
}

void montecarlo_ttt_match()
{
    if (extensive)
        cout << __func__ << endl;
    else
    {
        cout << __func__ << " (extensive mode off)" << endl;
        return;
    }

    mt19937 g( seed );
    ttt::montecarlo::NodeAllocator allocator;

    ttt::montecarlo::Data data1( g, allocator );
    ttt::montecarlo::Data data2( g, allocator );

    ttt::Game game( Player1, ttt::empty_state );

    MultiMatch< ttt::Move, ttt::State > match;
    const double exploration = 0.4;
    const size_t rounds = 100;
    match.play_match( 
        game, 
        [&game, exploration, &data1]() { return new ttt::montecarlo::Player( game, exploration, 100, data1 ); }, 
        [&game, exploration, &data2]() { return new ttt::montecarlo::Player( game, exploration, 500, data2 ); }, 
        rounds );

    if (verbose)
        cout 
            << "fst player wins: " << match.fst_player_wins << '\n'
            << "snd player wins: " << match.snd_player_wins << '\n'
            << "draws: " << match.draws << '\n'
            << "fst player move stack capacity: " << data1.move_stack.capacity() << '\n'
            << "fst player playouts: " << data1.playout_count << '\n'
            << "snd player move stack capacity: " << data2.move_stack.capacity() << '\n'
            << "snd player playouts: " << data2.playout_count << endl;

    assert (match.draws > 0);
}

void montecarlo_minimax_ttt_match()
{
    if (extensive)
        cout << __func__ << endl;
    else
    {
        cout << __func__ << " (extensive mode off)" << endl;
        return;
    }

    mt19937 g( seed );
    ttt::montecarlo::NodeAllocator allocator;

    ttt::montecarlo::Data data1( g, allocator );
    ttt::minimax::Data data2( g );

    ttt::Game game( Player1, ttt::empty_state );

    MultiMatch< ttt::Move, ttt::State > match;
    const double exploration = 0.4;
    const size_t rounds = 100;
    match.play_match( 
        game, 
        [&game, exploration, &data1]() { return new ttt::montecarlo::Player( game, exploration, 400, data1 ); }, 
        [&game, &data2]() { return new ttt::minimax::Player( game, 2, data2 ); }, 
        rounds );

    if (verbose)
        cout 
            << "fst player wins: " << match.fst_player_wins << '\n'
            << "snd player wins: " << match.snd_player_wins << '\n'
            << "draws: " << match.draws << '\n'
            << "fst player move stack capacity: " << data1.move_stack.capacity() << '\n'
            << "fst player playouts: " << data1.playout_count << endl;

    assert (match.draws > 0);
}

void montecarlo_minimax_uttt_match()
{
    if (extensive)
        cout << __func__ << endl;
    else
    {
        cout << __func__ << " (extensive mode off)" << endl;
        return;
    }

    mt19937 g( seed );

    uttt::Game game( Player1, uttt::empty_state );

    uttt::montecarlo::NodeAllocator allocator;
    uttt::montecarlo::Data data1( g, allocator );
    const double exploration = 0.4;
//    const size_t simulations = 3200;
    const size_t simulations = 100;

    uttt::minimax::Data data2( g );
//    const size_t depth = 6;
    const size_t depth = 2;
    const double factor = 9.0;

    const size_t rounds = 100;

    MultiMatch< uttt::Move, uttt::State > match;
    match.play_match( 
        game,
        [&game, exploration, &data1]() 
            { return new uttt::montecarlo::Player( game, exploration, simulations, data1 ); }, 
        [&game, factor, &data2]() 
            { return new uttt::minimax::Player( game, factor, depth, data2 ); }, 
        rounds );

    if (verbose)
        cout 
            << "fst player wins: " << match.fst_player_wins << '\n'
            << "snd player wins: " << match.snd_player_wins << '\n'
            << "draws: " << match.draws << '\n'
            << "fst player simulations: " << simulations << '\n'
            << "fst player exploration: " << exploration << '\n'
            << "fst player move stack capacity: " << data1.move_stack.capacity() << '\n'
            << "fst player playouts: " << data1.playout_count << '\n'
            << "fst player duration: " << match.fst_player_duration << '\n'
            << "snd player depth: " << depth << '\n'
            << "snd player move stack capacity: " << data2.move_stack.capacity() << '\n'
            << "snd player eval calls: " << data2.eval_calls << '\n' 
            << "snd player duration: " << match.snd_player_duration << '\n' 
            << "fst/snd player duration ratio: " 
            << double( chrono::duration_cast< std::chrono::microseconds >( 
                    match.fst_player_duration ).count()) / 
               chrono::duration_cast< std::chrono::microseconds >( 
                    match.snd_player_duration ).count() << '\n';

    assert (match.draws > 0);
}

void uttt_match_mm_vs_tree_mm()
{
    if (extensive)
        cout << __func__ << endl;
    else
    {
        cout << __func__ << " (extensive mode off)" << endl;
        return;
    }

    mt19937 g( seed );
    uttt::minimax::tree::NodeAllocator allocator;

    uttt::minimax::Data data1( g );
    uttt::minimax::tree::Data data2( g, allocator );

    const size_t fst_depth = 2; // 2
    const size_t snd_depth = 2; // 2

    uttt::Game game( Player1, uttt::empty_state );

    MultiMatch< uttt::Move, uttt::State > match;
    match.play_match( 
        game, 
        [&game, &data1]() { return new uttt::minimax::Player( game, 9.0, fst_depth, data1 ); }, 
        [&game, &data2]() { return new uttt::minimax::tree::Player( game, 9.0, snd_depth, data2 ); }, 
        100 );

    if (verbose)
        cout 
            << "fst player wins: " << match.fst_player_wins << '\n'
            << "snd player wins: " << match.snd_player_wins << '\n'
            << "draws: " << match.draws << '\n'
            << "fst player depth: " << fst_depth << '\n'
            << "fst player duration: " << match.fst_player_duration << '\n'
            << "fst player move stack capacity: " << data1.move_stack.capacity() << '\n'
            << "fst player eval calls: " << data1.eval_calls << '\n'
            << "snd player depth: " << snd_depth << '\n'
            << "snd player duration: " << match.snd_player_duration << '\n'
            << "snd player move stack capacity: " << data2.move_stack.capacity() << '\n'
            << "snd player eval calls: " << data2.eval_calls << '\n'
            << "fst/snd player duration ratio: "
            << double( chrono::duration_cast< std::chrono::microseconds >( 
                    match.fst_player_duration ).count()) / 
               chrono::duration_cast< std::chrono::microseconds >( 
                    match.snd_player_duration ).count() << '\n' << endl;

    assert (match.fst_player_wins != 0);
    assert (match.snd_player_wins != 0);
    assert (match.draws > 0);
    assert (data1.move_stack.capacity() == 81);
    assert (data2.move_stack.capacity() == 81);
    assert (data1.eval_calls > 800000 && data1.eval_calls < 1000000);
    assert (data2.eval_calls > 400000 && data2.eval_calls < 600000);
}

vector< ttt::alphazero::training::Position > selfplay_worker( 
    libtorch::InferenceManager& inference_manager, libtorch::Hyperparameters const& hp, 
    size_t runs_per_thread )
{
    mt19937 g = mt19937( random_device{}());
    ttt::alphazero::NodeAllocator node_allocator;
    vector< ttt::alphazero::training::Position > positions;
    PlayerIndex player_index = PlayerIndex::Player1;
    for (; runs_per_thread; --runs_per_thread) 
    {
        ttt::alphazero::libtorch::async::Player player( ttt::Game( player_index, ttt::empty_state ), 
            hp.c_base, hp.c_init, hp.simulations, node_allocator, inference_manager );
        alphazero::training::SelfPlay self_play(
            player, hp.dirichlet_alpha, hp.dirichlet_epsilon,
            hp.opening_moves, g, positions );
        self_play.run();
        player_index = toggle(player_index);
    }

    return positions;
}

void alphazero_training()
{
    if (extensive)
        cout << __func__ << endl;
    else
    {
        cout << __func__ << " (extensive mode off)" << endl;
        return;
    }

    const char* const model_path = "runs/models/ttt_alphazero_experiment_6/final_model.pt"; // Adjust if needed
    auto [model, hp] =  libtorch::load_model( model_path );
    libtorch::InferenceManager inference_manager( 
        std::move( model ), hp, ttt::alphazero::G, ttt::alphazero::P );

    vector< future< vector< ttt::alphazero::training::Position >>> thread_pool( 8 );
    cout << "start " << thread_pool.size() << " worker threads"  << endl;
    for (auto& future : thread_pool)
        future = async( selfplay_worker, ref(inference_manager), hp, 500 );

    cout << "wait for all threads to finish..." << endl;
    size_t total_positions = 0;
    for (auto& future : thread_pool) 
    {
        auto positions = future.get();
        total_positions += positions.size();
    }
    cout << "total positions: " << total_positions << endl;
}

void ttt_alphazero_nn_vs_minimax()
{
    if (extensive)
        cout << __func__ << endl;
    else
    {
        cout << __func__ << " (extensive mode off)" << endl;
        return;
    }

    const char* const model_path = "models/ttt_alphazero_experiment_1/final_model.pt"; // Adjust if needed
    cout << "load model " << model_path << endl;
    auto [model, hp] = libtorch::load_model( model_path ); 

    ttt::Game game( Player1, ttt::empty_state );
    ttt::alphazero::NodeAllocator allocator;
    mt19937 g( seed );
    ttt::minimax::Data data( g );

    MultiMatch< ttt::Move, ttt::State > match;
    const size_t rounds = 100;

    match.play_match( 
        game, 
        [&]() { return new ttt::alphazero::libtorch::sync::Player( 
            game, hp.c_base, hp.c_init, hp.simulations, allocator, *model ); }, 
        [&game, &data]() { return new ttt::minimax::Player( game, 3, data ); }, 
        rounds );

    if (verbose)
        cout 
            << "fst player wins: " << match.fst_player_wins << '\n'
            << "fst player duration: " << match.fst_player_duration << '\n'
            << "snd player wins: " << match.snd_player_wins << '\n'
            << "snd player duration: " << match.snd_player_duration << '\n'
            << "draws: " << match.draws << '\n'
            << "snd player move stack capacity: " << data.move_stack.capacity() << '\n'
            << "snd player eval calls: " << data.eval_calls << '\n'
            << "fst/snd player duration ratio: "
            << double( chrono::duration_cast< std::chrono::microseconds >( 
                    match.fst_player_duration ).count()) / 
               chrono::duration_cast< std::chrono::microseconds >( 
                    match.snd_player_duration ).count() << '\n' << endl;

    assert (match.draws > 0);
}

} // namespace test {

int main()
{
    try
    {
        random_device rd;
        seed = rd();
        cout << "run tests with seed " << seed << endl << endl;
/*
        test::toggle_player();
        test::build_game();
        test::eval_won_game();
        test::eval_drawn_game();
        test::eval_undecided_game();
        test::nim_game();
        test::nim_match();
        test::ttt_game();
        test::ttt_human();
        test::tic_tac_toe_match();
        test::uttt_game();
        test::uttt_human();
        test::uttt_match();
        test::montecarlo_node();
        test::montecarlo_player();
        test::montecarlo_ttt_human();
        test::montecarlo_ttt_match();
        test::montecarlo_minimax_ttt_match();
        test::montecarlo_minimax_uttt_match();
        test::uttt_match_mm_vs_tree_mm();
        test::alphazero_ttt_match();
        test::ttt_multimatch_alphazero_vs_minimax();
        test::montecarlo_minimax_uttt_match();
        test::uttt_multimatch_alphazero_vs_minimax();
        test::alphazero_training();
        test::ttt_alphazero_nn_vs_minimax();
        */
        test::ttt_alphazero_nn_vs_minimax();

        cout << "\neverything ok" << endl;    
        return 0;
    }
    catch( exception const& e )
    {
        cout << "exception caught: " << e.what() << endl;
        return -1;
    }
}