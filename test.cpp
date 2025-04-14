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

struct TestGame : public UndecidedGame< char >
{
    TestGame( PlayerIndex player_index, char state = ' ')
        : UndecidedGame( player_index ), state( state )
    {
        moves.push_back( 'a' );
        moves.push_back( 'b' );
    }
    
    ranges::subrange< typename vector< char >::const_iterator > valid_moves() const override
    {
        return ranges::subrange( moves.cbegin(), moves.cend());
    }
    virtual std::unique_ptr< Game > apply( size_t index ) const override
    {
        return std::unique_ptr< Game >( new TestGame( player_index, *(moves.begin() + index)));
    }

    char state;
    std::vector< char > moves;
};

struct TestPlayer : public Player< char >
{
    size_t choose( Game< char > const& ) override
    {
        return 0;
    }
};

void build_game()
{
    cout << __func__ << endl;
    
    DrawnGame< char > drawn_game( Player1 );
    if (drawn_game.current_player_index() != Player1)
        assert( !"invalid current player index" );

    WonGame< char > won_game( Player2 );
    if (won_game.current_player_index() != Player2)
        assert( !"initial player index not passed on" );
    if (won_game.winner() != Player2)
        assert( !"winner not returned" );
}

void build_undecided_game()
{
    cout << __func__ << endl;

    TestGame undecided_game( Player2 );

    if (undecided_game.current_player_index() != Player2)
        assert( !"invalid current player index" );
    if (!ranges::equal( undecided_game.valid_moves(), vector{'a', 'b'}))
        assert( !"invalid valid moves" );
}

void eval_won_game()
{
    cout << __func__ << endl;
    mt19937 g;
    minimax::ScoreFunction< char > score = []( Game< char > const& ) 
        { return 0.0; };
    if (minimax::eval< char >( WonGame< char >( Player2 ), score, 0 ) != INFINITY)
        assert( !"wrong score for won game" );
    if (minimax::eval< char >( WonGame< char >( Player1 ), score, 0 ) != -INFINITY)
        assert( !"wrong score for won game" );
}

void eval_drawn_game()
{
    cout << __func__ << endl;
    mt19937 g;
    
    minimax::ScoreFunction< char > score = []( Game< char > const& ) 
        { return 0.0; };
    if (minimax::eval< char >( DrawnGame< char >( Player1 ), score, 0 ) != 0.0)
        assert( !"wrong score for drawn game" );
}

void eval_undecided_game()
{
    cout << __func__ << endl;

    TestGame undecided_game( Player2 );

    minimax::ScoreFunction< char > score = []( Game< char > const& ) 
        { return 42.0; };
    mt19937 g;
    if (minimax::eval( undecided_game, score, 0 ) != 42.0)
        assert( !"wrong score for undecided game" );

}

struct TestNimPlayer : public Player< nim::Move >
{
    size_t choose( Game< nim::Move > const& game ) override
    {
        return next_move_index;
    }
    size_t next_move_index;
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
    nim::Game game( Player1, heap_stack.size(), heap_stack, moves_stack, g );
    auto moves = vector{ nim::Move{ 0, 1 }, nim::Move{ 1, 1 }, nim::Move{ 1, 2 } };
    assert (std::is_permutation(moves.begin(), moves.end(), game.valid_moves().begin()));
    player1.next_move_index = find( 
        game.valid_moves().begin(), game.valid_moves().end(), 
        nim::Move{ 1, 1 }) 
        - game.valid_moves().begin();    
    auto next_game = game.apply( player1.next_move_index );    
    assert( next_game->current_player_index() == Player2 );
    auto nim = dynamic_cast< nim::Game* >( next_game.get() );
    assert (nim);
    moves = vector{ nim::Move{ 0, 1 }, nim::Move{ 1, 1 }};
    assert (std::is_permutation(moves.begin(), moves.end(), nim->valid_moves().begin()));
    player2.next_move_index = find( nim->valid_moves().begin(),
                                    nim->valid_moves().end(),
                                    nim::Move{ 1, 1 }) - nim->valid_moves().begin();
    auto next_game2 = nim->apply( player2.next_move_index );
    assert( next_game2->current_player_index() == Player1 );
    nim = dynamic_cast< nim::Game* >( next_game2.get() );
    assert (nim);
    moves = vector{ nim::Move{ 0, 1 }};
    assert( ranges::equal( nim->valid_moves(), moves));
    
    player1.next_move_index = 0;
    auto next_game3 = nim->apply( player1.next_move_index );
    assert( next_game3->current_player_index() == Player2 );
    auto won_game = dynamic_cast< WonGame< nim::Move >* >( next_game3.get() );
    assert (won_game);
    assert (won_game->winner() == Player2);
}

struct NimPlayer : public minimax::Player< nim::Move >
{
    NimPlayer( unsigned depth ) : minimax::Player< nim::Move >( depth ) {}
    double score( Game< nim::Move > const& game ) override
    {
        // don't care about the score
        return 0.0;
    }
};

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
    void drawn( Game< nim::Move > const& ) override
    {
        ++draws;
        //cout << "drawn" << endl;
    }
    void won( Game< nim::Move > const& won_game ) override
    {
        if (won_game.current_player_index() == fst_player_index)
            ++fst_player_wins;
        else
            ++snd_player_wins;
        //cout << "won player " << won_game.winner() << endl;
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
    MinimaxNimPlayer( unsigned depth ) 
        : ::minimax::Player< nim::Move >( depth ) {}
    double score( Game< nim::Move > const& ) override { return 0; }
};

void play_nim()
{
    cout << __func__ << endl;

    mt19937 g( seed );
    vector< nim::Move > move_stack;
    vector< size_t > heap_stack{ 1, 2, 3, 4, 5 };
    nim::Game game( Player1, heap_stack.size(), heap_stack, move_stack, g );

    MinimaxNimPlayer fst_player( 2  );
    MinimaxNimPlayer snd_player( 3 );
    MultiMatch match;
    match.play_match( game, fst_player, snd_player, 100 );

    cout << "fst player wins: " << match.fst_player_wins << endl;
    cout << "snd player wins: " << match.snd_player_wins << endl;
    cout << "draws: " << match.draws << endl;
    cout << "move stack size: " << move_stack.size() << endl;
    cout << "move stack capacity: " << move_stack.capacity() << endl;
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
        test::build_undecided_game();
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