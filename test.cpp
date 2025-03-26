#include <iostream>
#include <exception>
#include <cassert>

using namespace std;

#include "game.h"

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
    TestGame( Player< char > const& player, Player< char > const& opponent )
        : UndecidedGame( player, opponent ) {}
    virtual std::vector< char > const& valid_moves() const override
    {
        static std::vector< char > moves;
        return moves;
    }
    virtual std::unique_ptr< Game > apply_next_move() const override
    {
        return std::unique_ptr< Game >( new TestGame( opponent, player ) );
    }
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
}

} // namespace test {

int main()
{
    try
    {
        cout << "run tests\n" << endl;

        test::toggle_player();
        test::build_game();

        cout << "\neverything ok" << endl;    
        return 0;
    }
    catch( exception const& e )
    {
        cout << "exception caught: " << e.what() << endl;
        return -1;
    }
}