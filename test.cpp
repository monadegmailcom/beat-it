#include <iostream>
#include <exception>
#include <cassert>
#include <algorithm>

#include "minimax.h"
#include "nim.h"
#include "match.h"

using namespace std;

namespace test {

void toggle_player()
{
    cout << __func__ << endl;
        
    if( toggle( Player1 ) != Player2 ||
        toggle( Player2 ) != Player1 )
        assert( false );
}

struct TestGame : public Game< char >
{
    TestGame( PlayerIndex player_index, GameResult game_result ) : Game( player_index ), game_result( game_result )
    {
    }
    
    void append_valid_moves( vector< char >& move_stack ) const override
    {
        move_stack.push_back( 'a' );
        move_stack.push_back( 'b' );
    }

    virtual std::unique_ptr< Game > apply( const char& ) const override
    {
        return std::unique_ptr< Game >( new TestGame( player_index, game_result ));
    }

    GameResult result() const override
    {
        return game_result;
    }

    GameResult game_result;
};

struct TestPlayer : public Player< char >
{
    char choose( Game< char > const& ) override
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
    minimax::ScoreFunction< char > score = []( Game< char > const& ) 
        { return 0.0; };
    if (minimax::eval< char >( TestGame( Player2, GameResult::Player2Win ), score, 0, move_stack, g ) != INFINITY)
        assert( !"wrong score for won game" );
    if (minimax::eval< char >( TestGame( Player2, GameResult::Player1Win ), score, 0, move_stack, g ) != -INFINITY)
        assert( !"wrong score for won game" );
}

void eval_drawn_game()
{
    cout << __func__ << endl;
    mt19937 g;
    vector< char > move_stack;
    
    minimax::ScoreFunction< char > score = []( Game< char > const& ) 
        { return 0.0; };
    if (minimax::eval< char >( TestGame( Player2, GameResult::Draw ), score, 0, move_stack, g ) != 0.0)
        assert( !"wrong score for drawn game" );
}

void eval_undecided_game()
{
    cout << __func__ << endl;

    TestGame undecided_game( Player2, GameResult::Undecided );

    minimax::ScoreFunction< char > score = []( Game< char > const& ) 
        { return 42.0; };
    mt19937 g;
    vector< char > move_stack;
    if (minimax::eval( undecided_game, score, 0, move_stack, g ) != 42.0)
        assert( !"wrong score for undecided game" );

}

struct TestNimPlayer : public Player< nim::Move >
{
    nim::Move choose( Game< nim::Move > const& game ) override
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
    mt19937 g( seed );
    vector< nim::Move > moves_stack;
    vector< size_t > heap_stack{ 1, 2 };
    nim::Game game( Player1, 0, heap_stack );
    auto moves = vector{ nim::Move{ 0, 1 }, nim::Move{ 1, 1 }, nim::Move{ 1, 2 } };
    vector< nim::Move > move_stack;
    game.append_valid_moves( move_stack );
    assert (std::is_permutation(moves.begin(), moves.end(), move_stack.begin()));
    player1.next_move = nim::Move{ 1, 1 };
    auto next_game = game.apply( player1.next_move );    
    assert( next_game->current_player_index() == Player2 );
    auto nim = dynamic_cast< nim::Game* >( next_game.get() );
    assert (nim);
    moves = vector{ nim::Move{ 0, 1 }, nim::Move{ 1, 1 }};
    move_stack.clear();
    nim->append_valid_moves( move_stack );
    assert (std::is_permutation(moves.begin(), moves.end(), move_stack.begin()));
    player2.next_move = nim::Move{ 1, 1 };
    auto next_game2 = nim->apply( player2.next_move );
    assert( next_game2->current_player_index() == Player1 );
    nim = dynamic_cast< nim::Game* >( next_game2.get() );
    assert (nim);
    moves = vector{ nim::Move{ 0, 1 }};
    move_stack.clear();
    nim->append_valid_moves( move_stack );
    assert( ranges::equal( move_stack, moves));
    
    player1.next_move = nim::Move{ 0, 1 };
    auto next_game3 = nim->apply( player1.next_move );
    assert( next_game3->current_player_index() == Player2 );
    assert (next_game3->result() == GameResult::Player2Win);
}

struct MultiMatch : public ::Match< nim::Move >
{
    void report( Game< nim::Move > const& game, nim::Move const& move ) override
    {
        /*
        cout << "player " << toggle( game.current_player_index())
             << ", heap = " << move.heap_index + 1
             << ", count = " << move.count << endl;
        */
    }
    void draw( Game< nim::Move > const& ) override
    {
        ++draws;
        //cout << "drawn" << endl
    }
    void player1_win( Game< nim::Move > const& ) override
    {
        if (fst_player_index == Player1)
            ++fst_player_wins;
        else
            ++snd_player_wins;
    }
    void player2_win( Game< nim::Move > const& ) override
    {
        if (fst_player_index == Player2)
            ++fst_player_wins;
        else
            ++snd_player_wins;
    }
    void play_match( Game< nim::Move > const& game, Player< nim::Move >& fst_player, Player< nim::Move >& snd_player, 
                     size_t rounds )
    {
        draws = 0;
        fst_player_wins = 0;
        snd_player_wins = 0;
        
        Player< nim::Move >* player = &fst_player;
        Player< nim::Move >* opponent = &snd_player;
        
        fst_player_index = game.current_player_index();
        snd_player_index = toggle( fst_player_index );

        for (size_t rounds = 100; rounds > 0; --rounds)
        {
            play( game, *player, *opponent );
            swap( player, opponent );
            swap( fst_player_index, snd_player_index );
        }
    }

    size_t draws = 0;
    size_t fst_player_wins = 0;
    size_t snd_player_wins = 0;
    PlayerIndex fst_player_index;
    PlayerIndex snd_player_index;
};

class MinimaxNimPlayer : public ::minimax::Player< nim::Move >
{
public:
    MinimaxNimPlayer( unsigned depth, mt19937& g ) : minimax::Player< nim::Move >( depth, g ) {}
    double score( Game< nim::Move > const& ) override { return 0; }
    vector< nim::Move > const& get_move_stack() const { return move_stack; };
};

void play_nim()
{
    cout << __func__ << endl;

    mt19937 g( seed );
    vector< size_t > heap_stack{ 1, 2, 3, 4, 5 };
    nim::Game game( Player1, 0, heap_stack );

    MinimaxNimPlayer fst_player( 2, g );
    MinimaxNimPlayer snd_player( 3, g );
    MultiMatch match;
    match.play_match( game, fst_player, snd_player, 100 );

    cout << "fst player wins: " << match.fst_player_wins << endl;
    cout << "snd player wins: " << match.snd_player_wins << endl;
    cout << "draws: " << match.draws << endl;
    cout << "fst player move stack size: " << fst_player.get_move_stack().size() << endl;
    cout << "fst player move stack capacity: " << fst_player.get_move_stack().capacity() << endl;
    cout << "snd player move stack size: " << snd_player.get_move_stack().size() << endl;
    cout << "snd player move stack capacity: " << snd_player.get_move_stack().capacity() << endl;
    cout << "heap stack size: " << heap_stack.size() << endl;
    cout << "heap stack capacity: " << heap_stack.capacity() << endl;
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
        test::play_nim();

        cout << "\neverything ok" << endl;    
        return 0;
    }
    catch( exception const& e )
    {
        cout << "exception caught: " << e.what() << endl;
        return -1;
    }
}