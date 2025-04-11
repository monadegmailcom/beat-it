#include <iostream>
#include <exception>
#include <cassert>

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
    
    std::vector< char > const& valid_moves() const override
    {
        return moves;
    }
    virtual std::unique_ptr< Game > apply( vector< char >::const_iterator itr ) const override
    {
        return std::unique_ptr< Game >( new TestGame( player_index, *itr ) );
    }

    char state;
    std::vector< char > moves;
};

struct TestPlayer : public Player< char >
{
    TestPlayer( PlayerIndex index ) : Player( index ) {}
    vector< char >::const_iterator choose( 
        UndecidedGame< char > const& game ) override
    {
        return game.valid_moves().begin();
    }
};

void build_game()
{
    cout << __func__ << endl;
    
    DrawnGame drawn_game( Player1 );
    if (drawn_game.current_player_index() != Player1)
        assert( !"invalid current player index" );

    WonGame won_game( Player2 );
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
    if (undecided_game.valid_moves() != vector{'a', 'b'})
        assert( !"invalid valid moves" );
}

void eval_won_game()
{
    cout << __func__ << endl;
    mt19937 g;
    minimax::ScoreFunction< char > score = []( UndecidedGame< char > const& ) 
        { return 0.0; };
    if (minimax::eval< char >( WonGame( Player2 ), score, 0, g ) != INFINITY)
        assert( !"wrong score for won game" );
    if (minimax::eval< char >( WonGame( Player1 ), score, 0, g ) != -INFINITY)
        assert( !"wrong score for won game" );
}

void eval_drawn_game()
{
    cout << __func__ << endl;
    mt19937 g;
    
    minimax::ScoreFunction< char > score = []( UndecidedGame< char > const& ) 
        { return 0.0; };
    if (minimax::eval< char >( DrawnGame( Player1 ), score, 0, g ) != 0.0)
        assert( !"wrong score for drawn game" );
}

void eval_undecided_game()
{
    cout << __func__ << endl;

    TestGame undecided_game( Player2 );

    minimax::ScoreFunction< char > score = []( UndecidedGame< char > const& ) 
        { return 42.0; };
    mt19937 g;

    if (minimax::eval( undecided_game, score, 0, g ) != 42.0)
        assert( !"wrong score for undecided game" );

}

struct TestNimPlayer : public Player< nim::Move >
{
    TestNimPlayer( PlayerIndex index ) : Player( index ) {}
    vector< nim::Move >::const_iterator choose( 
        UndecidedGame< nim::Move > const& ) override
    {
        return next_move;
    }
    vector< nim::Move >::const_iterator next_move;
};

random_device rd;
unsigned seed = 0;

void nim_game()
{
    cout << __func__ << endl;

    TestNimPlayer player1( Player1 );
    TestNimPlayer player2( Player2 );
    mt19937 g( seed );
    nim::Game game( Player1, { 1, 2 } );
    auto moves = vector{ nim::Move{ 0, 1 }, nim::Move{ 1, 1 }, 
                         nim::Move{ 1, 2 } };
    assert (std::is_permutation(moves.begin(), moves.end(), game.valid_moves().begin()));
    player1.next_move = find( game.valid_moves().begin(), game.valid_moves().end(),
                              nim::Move{ 1, 1 });    
    auto next_game = game.apply( player1.next_move );    
    assert( next_game->current_player_index() == Player2 );
    auto nim = dynamic_cast< nim::Game* >( next_game.get() );
    assert (nim);
    moves = vector{ nim::Move{ 0, 1 }, 
                    nim::Move{ 1, 1 }};
    assert (std::is_permutation(moves.begin(), moves.end(), nim->valid_moves().begin()));
    player2.next_move = find( nim->valid_moves().begin(),
                              nim->valid_moves().end(),
                              nim::Move{ 1, 1 });
    next_game = nim->apply( player2.next_move );
    assert( next_game->current_player_index() == Player1 );
    nim = dynamic_cast< nim::Game* >( next_game.get() );
    assert (nim);
    moves = vector{ nim::Move{ 0, 1 }};
    assert( nim->valid_moves() == moves);
    
    player1.next_move = nim->valid_moves().begin();
    next_game = nim->apply( player1.next_move );
    assert( next_game->current_player_index() == Player2 );
    auto won_game = dynamic_cast< WonGame* >( next_game.get() );
    assert (won_game);
    assert (won_game->winner() == Player2);
}

void print( nim::Game const& game )
{
    for( auto const& heap : game.get_heaps() )
        cout << heap << " ";
    cout << endl;
}

struct NimPlayer : public minimax::Player< nim::Move >
{
    NimPlayer( PlayerIndex index, unsigned depth, mt19937& g) 
        : minimax::Player< nim::Move >( index, depth, g ) {}
    double score( UndecidedGame< nim::Move > const& game ) override
    {
        // don't care about the score
        return 0.0;
    }
};

struct MultiMatch : public ::Match< nim::Move >
{
    MultiMatch( Player< nim::Move >& player1, Player< nim::Move >& player2 ) 
        : ::Match< nim::Move >(), player1( player1 ), player2( player2 ) {}

    void report( Game const& game, nim::Move const& move ) override
    {
        /*
        cout << "player " << toggle( game.current_player_index())
             << ", heap = " << move.heap_index + 1
             << ", count = " << move.count << endl;
        */
    }
    void drawn( DrawnGame const& ) override
    {
        ++draws;
        //cout << "drawn" << endl;
    }
    void won( WonGame const& won_game ) override
    {
        if (won_game.winner() == player1.get_index())
            ++player1_wins;
        else
            ++player2_wins;
        //cout << "won player " << won_game.winner() << endl;
    }
    Player< nim::Move >& player1;
    Player< nim::Move >& player2;
    size_t draws = 0;
    size_t player1_wins = 0;
    size_t player2_wins = 0;
};

class MinimaxNimPlayer : public ::minimax::Player< nim::Move >
{
public:
    MinimaxNimPlayer( PlayerIndex index, unsigned depth, mt19937& g ) 
        : ::minimax::Player< nim::Move >( index, depth, g ) {}
    double score( UndecidedGame< nim::Move > const& ) override { return 0; }
};

void play_nim()
{
    mt19937 g( seed );
    nim::Game game( Player1, { 1, 2, 3, 4, 5 } );

    MinimaxNimPlayer player1( Player1, 2, g );
    MinimaxNimPlayer player2( Player2, 3, g );
    MultiMatch match( player1, player2 );
    for (size_t rounds = 100; rounds > 0; --rounds)
    {
        match.play( game, player1, player2 );
        player1.set_index( toggle( player1.get_index()));
        player2.set_index( toggle( player2.get_index()));
    }
    cout << "player 1 wins: " << match.player1_wins << endl;
    cout << "player 2 wins: " << match.player2_wins << endl;
    cout << "draws: " << match.draws << endl;
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