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

void build_game()
{
    cout << __func__ << endl;
    
    Game game( Player1 );
    
    if( game.current_player_index() != Player1 )
        assert( !"initial player index not returned" );

    UndecidedGame undecided_game( Player2 );
    if (undecided_game.current_player_index() != Player2)
        assert( !"initial player index not passed on" );

    DrawnGame drawn_game( Player1 );
    if (drawn_game.current_player_index() != Player1)
        assert( !"initial player index not passed on" );

    WonGame won_game( Player2 );
    if (won_game.current_player_index() != Player2)
        assert( !"initial player index not passed on" );
    if (won_game.winner() != Player2)
        assert( !"winner not returned" );
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