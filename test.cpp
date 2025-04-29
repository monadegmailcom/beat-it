#include <iostream>
#include <exception>
#include <cassert>

#include "games/nim.h"
#include "games/ultimate_ttt.h"
#include "match.h"

using namespace std;

template<>
struct GameState< char, GameResult >
{
    static void append_valid_moves( 
        vector< char >& move_stack, PlayerIndex, GameResult const& state )
    {
        move_stack.push_back( 'a' );
        move_stack.push_back( 'b' );
    }

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

struct TestPlayer : public Player< char, GameResult >
{
    char choose( TestGame const& ) override
    {
        return 0;
    }
};

void build_game()
{
    cout << __func__ << endl;

    TestGame game( Player2, GameResult::Undecided );

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
    if (minimax::eval< char >( TestGame( Player2, GameResult::Player2Win ), score, 0, move_stack, 0, 1, g, calls ) != INFINITY)
        assert( !"wrong score for won game" );
    if (minimax::eval< char >( TestGame( Player2, GameResult::Player1Win ), score, 0, move_stack, 0, 1, g, calls ) != -INFINITY)
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
    if (minimax::eval< char >( TestGame( Player2, GameResult::Draw ), score, 0, move_stack, 0, 1, g, calls ) != 0.0)
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

    if (minimax::eval( undecided_game, score, 0, move_stack, 0, 1, g, calls ) != 42.0)
        assert( !"wrong score for undecided game" );

}

struct TestNimPlayer : public Player< nim::Move, nim::State< 1 > >
{
    nim::Move choose( Game< nim::Move, nim::State< 1 > > const& game ) override
    {
        return next_move;
    }
    nim::Move next_move;
};

random_device rd;
unsigned seed = 0;

void nim_game()
{
    cout << __func__ << endl;

    TestNimPlayer player1;
    TestNimPlayer player2;
    nim::Game< 2 > game( Player1, array< size_t, 2 >{ 1, 2 } );
    auto moves = vector{ nim::Move{ 0, 1 }, nim::Move{ 1, 1 }, nim::Move{ 1, 2 } };
    vector< nim::Move > move_stack;
    game.append_valid_moves( move_stack );
    assert (std::is_permutation(moves.begin(), moves.end(), move_stack.begin()));
    player1.next_move = nim::Move{ 1, 1 };
    auto next_game = game.apply( player1.next_move );    
    assert( next_game.current_player_index() == Player2 );
    moves = vector{ nim::Move{ 0, 1 }, nim::Move{ 1, 1 }};
    move_stack.clear();
    next_game.append_valid_moves( move_stack );
    assert (std::is_permutation(moves.begin(), moves.end(), move_stack.begin()));
    player2.next_move = nim::Move{ 1, 1 };
    auto next_game2 = next_game.apply( player2.next_move );
    assert( next_game2.current_player_index() == Player1 );
    moves = vector{ nim::Move{ 0, 1 }};
    move_stack.clear();
    next_game2.append_valid_moves( move_stack );
    assert( ranges::equal( move_stack, moves));
    
    player1.next_move = nim::Move{ 0, 1 };
    auto next_game3 = next_game2.apply( player1.next_move );
    assert( next_game3.current_player_index() == Player2 );
    assert (next_game3.result() == GameResult::Player2Win);
}

void nim_match()
{
    cout << __func__ << endl;

    mt19937 g( seed );
    const size_t HEAPS = 5;
    nim::Game< HEAPS > game( Player1, { 1, 2, 3, 4, 5 } );

    minimax::Player< nim::Move, nim::State< HEAPS > > fst_player( 2, g );
    minimax::Player< nim::Move, nim::State< HEAPS > > snd_player( 3, g );
    MultiMatch< nim::Move, nim::State< HEAPS > > match;
    match.play_match( game, fst_player, snd_player, 100 );

    cout 
        << "fst player wins: " << match.fst_player_wins << '\n'
        << "snd player wins: " << match.snd_player_wins << '\n'
        << "draws: " << match.draws << '\n'
        << "fst player move stack size: " << fst_player.get_move_stack().size() << '\n'
        << "fst player move stack capacity: " << fst_player.get_move_stack().capacity() << '\n'
        << "snd player move stack size: " << snd_player.get_move_stack().size() << '\n'
        << "snd player move stack capacity: " << snd_player.get_move_stack().capacity() << '\n'
        << "fst player eval calls: " << fst_player.get_eval_calls() << '\n'
        << "snd player eval calls: " << snd_player.get_eval_calls() << endl;
}

struct TicTacToeMatch : public Match< ttt::Move, ttt::State >
{
    TicTacToeMatch( minimax::Player< ttt::Move, ttt::State > const& minimax_player)
        : minimax_player( minimax_player ) {}

    minimax::Player< ttt::Move, ttt::State > const& minimax_player;

    void report( ttt::Game const& game, ttt::Move const& move ) override
    {
        cout 
            << "player " << game.current_player_index() << " (" 
            << (int)move << ")\n"
            << "board before move:\n" << game << '\n'
            << "score: " << minimax_player.score( game ) << endl;
    }
    void draw( ttt::Game const& game ) override
    {
        cout << "draw\n" << game << endl;
    }
    void player1_win( ttt::Game const& game ) override
    {
        cout << "player 1 win\n" << game << endl;;
    }
    void player2_win( ttt::Game const& game ) override
    {
        cout << "player 2 win\n" << game << endl;;
    }
};

void ttt_human()
{
    cout << __func__ << endl;

    ttt::Game game( Player1, ttt::empty_state );

    ttt::console::HumanPlayer human;

    mt19937 g( seed );
    ttt::minimax::Player player( 0, g );

    TicTacToeMatch match( player);
    match.play( game, human, player );
    cout << '\n'
        << "player move stack capacity: " << player.get_move_stack().capacity() << '\n'
        << "player eval calls: " << player.get_eval_calls() << endl;
}

void tic_tac_toe_match()
{
    cout << __func__ << endl;

    mt19937 g( seed );

    ttt::Game game( Player1, ttt::empty_state );

    minimax::Player< ttt::Move, ttt::State > fst_player( 0, g );
    minimax::Player< ttt::Move, ttt::State > snd_player( 5, g );
    MultiMatch< ttt::Move, ttt::State > match;
    match.play_match( game, fst_player, snd_player, 100 );

    cout 
        << "fst player wins: " << match.fst_player_wins << '\n'
        << "snd player wins: " << match.snd_player_wins << '\n'
        << "draws: " << match.draws << '\n'
        << "fst player move stack size: " << fst_player.get_move_stack().size() << '\n'
        << "fst player move stack capacity: " << fst_player.get_move_stack().capacity() << '\n'
        << "snd player move stack size: " << snd_player.get_move_stack().size() << '\n'
        << "snd player move stack capacity: " << snd_player.get_move_stack().capacity() << '\n'
        << "fst player eval calls: " << fst_player.get_eval_calls() << '\n'
        << "snd player eval calls: " << snd_player.get_eval_calls() << endl;
}

struct UltimateTicTacToeMatch : public Match< uttt::Move, uttt::State >
{
    UltimateTicTacToeMatch( minimax::Player< uttt::Move, uttt::State > const& minimax_player)
        : minimax_player( minimax_player ) {}

    minimax::Player< uttt::Move, uttt::State > const& minimax_player;

    void report( uttt::Game const& game, uttt::Move const& move ) override
    {
        cout 
            << "player " << game.current_player_index() << " (" 
            << (int)move.big_move << "," << (int)move.small_move << ")\n"
            << "board before move:\n" << game << '\n'
            << "score: " << minimax_player.score( game ) << '\n'
            << "best score: " << minimax_player.get_best_score() << endl;
    }
    void draw( uttt::Game const& game ) override
    {
        cout << "draw\n" << game << endl;
    }
    void player1_win( uttt::Game const& game ) override
    {
        cout << "player 1 win\n" << game << endl;;
    }
    void player2_win( uttt::Game const& game ) override
    {
        cout << "player 2 win\n" << game << endl;;
    }
};


void uttt_human()
{
    cout << __func__ << endl;
    uttt::Game game( Player1, uttt::empty_state );

    uttt::console::HumanPlayer human;

    mt19937 g( seed );
    uttt::minimax::Player player( 9.0, 5, g );

    UltimateTicTacToeMatch match( player );
    match.play( game, human, player );
    cout << '\n'
        << "player move stack capacity: " << player.get_move_stack().capacity() << '\n'
        << "player eval calls: " << player.get_eval_calls() << endl;
}

void uttt_match()
{
    cout << __func__ << endl;

    mt19937 g( seed );

    uttt::Game game( Player1, uttt::empty_state );

    uttt::minimax::Player fst_player( 9.0, 1, g );
    //minimax::Player< uttt::Move, uttt::State > snd_player( 0, g );
    uttt::minimax::Player snd_player( 9.0, 4, g );
    MultiMatch< uttt::Move, uttt::State > match;
    match.play_match( game, fst_player, snd_player, 100 );

    cout 
        << "fst player wins: " << match.fst_player_wins << '\n'
        << "snd player wins: " << match.snd_player_wins << '\n'
        << "draws: " << match.draws << '\n'
        << "fst player move stack size: " << fst_player.get_move_stack().size() << '\n'
        << "fst player move stack capacity: " << fst_player.get_move_stack().capacity() << '\n'
        << "snd player move stack size: " << snd_player.get_move_stack().size() << '\n'
        << "snd player move stack capacity: " << snd_player.get_move_stack().capacity() << '\n'
        << "fst player eval calls: " << fst_player.get_eval_calls() << '\n'
        << "snd player eval calls: " << snd_player.get_eval_calls() << endl;
}

} // namespace test {

int main()
{
    try
    {
        test::seed = test::rd();
        cout << "run tests with seed " << test::seed << endl << endl;

        test::toggle_player();
        test::build_game();
        test::eval_won_game();
        test::eval_drawn_game();
        test::eval_undecided_game();
        test::nim_game();
        //test::nim_match();
        //test::ttt_human();
        //test::tic_tac_toe_match();
        //test::uttt_human();
        test::uttt_match();

        cout << "\neverything ok" << endl;    
        return 0;
    }
    catch( exception const& e )
    {
        cout << "exception caught: " << e.what() << endl;
        return -1;
    }
}