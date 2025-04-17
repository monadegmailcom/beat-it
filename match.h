#include "player.h"

template< typename MoveT, typename StateT >
class Match
{
public:
    virtual ~Match() {}
    void play( 
        Game< MoveT, StateT > game, Player< MoveT, StateT >& player, 
        Player< MoveT, StateT >& opponent )
    {
        const GameResult result = game.result();
        if (result == GameResult::Draw)
            draw( game );
        else if (result == GameResult::Player1Win)
            player1_win( game );
        else if (result == GameResult::Player2Win)
            player2_win( game );
        else
        {
            const MoveT move = player.choose( game );
            report( game, move );
            play( game.apply( move ), opponent, player );
        }
    }
protected:
    virtual void report( Game< MoveT, StateT > const& game, MoveT const& move ) = 0;
    virtual void player1_win( Game< MoveT, StateT > const& ) = 0;
    virtual void player2_win( Game< MoveT, StateT > const& ) = 0;
    virtual void draw( Game< MoveT, StateT > const& ) = 0;
};
