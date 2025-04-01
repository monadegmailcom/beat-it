#include <iostream>
#include <exception>
#include <cassert>

#include "minimax.h"

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
    TestGame( Player< char > const& player, Player< char > const& opponent, 
              char state = ' ')
        : UndecidedGame( player, opponent ), state( state )
    {
        moves.push_back( 'a' );
        moves.push_back( 'b' );
    }

    virtual std::unique_ptr< Game > apply( char const& move ) const override
    {
        return std::unique_ptr< Game >( new TestGame( opponent, player, move ) );
    }

    char state;
};

struct TestPlayer : public Player< char >
{
    TestPlayer( PlayerIndex index ) : Player( index ) {}
    vector< char >::const_iterator choose( 
        vector< char > const& valid_moves ) override
    {
        return valid_moves.begin();
    }
};

void build_game()
{
    cout << __func__ << endl;
    
    TestPlayer player1( Player1 );
    TestPlayer player2( Player2 );

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

    TestPlayer player1( Player1 );
    TestPlayer player2( Player2 );

    TestGame undecided_game( player2, player1 );

    if (undecided_game.current_player_index() != Player2)
        assert( !"invalid current player index" );
    if (undecided_game.next_to_make_a_move().get_index() 
        != undecided_game.current_player_index())
        assert( !"game index and player index do not match" );
    if (undecided_game.valid_moves() != vector{'a', 'b'})
        assert( !"invalid valid moves" );
}

void eval_won_game()
{
    cout << __func__ << endl;
    
    minimax::ScoreFunction< char > score = []( UndecidedGame< char > const& ) 
        { return 0.0; };
    if (minimax::eval< char >( WonGame( Player2 ), score, 0 ) != INFINITY)
        assert( !"wrong score for won game" );
    if (minimax::eval< char >( WonGame( Player1 ), score, 0 ) != -INFINITY)
        assert( !"wrong score for won game" );
}

void eval_drawn_game()
{
    cout << __func__ << endl;
    
    minimax::ScoreFunction< char > score = []( UndecidedGame< char > const& ) 
        { return 0.0; };
    if (minimax::eval< char >( DrawnGame( Player1 ), score, 0 ) != 0.0)
        assert( !"wrong score for drawn game" );
}

void eval_undecided_game()
{
    cout << __func__ << endl;

    TestPlayer player1( Player1 );
    TestPlayer player2( Player2 );

    TestGame undecided_game( player2, player1 );

    minimax::ScoreFunction< char > score = []( UndecidedGame< char > const& ) 
        { return 42.0; };

    if (minimax::eval( undecided_game, score, 0 ) != 42.0)
        assert( !"wrong score for undecided game" );

}

} // namespace test {

int main()
{
    try
    {
        cout << "run tests\n" << endl;

        test::toggle_player();
        test::build_game();
        test::build_undecided_game();
        test::eval_won_game();
        test::eval_drawn_game();
        test::eval_undecided_game();

        cout << "\neverything ok" << endl;    
        return 0;
    }
    catch( exception const& e )
    {
        cout << "exception caught: " << e.what() << endl;
        return -1;
    }
}