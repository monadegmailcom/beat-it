#include "game.h"

template< typename MoveT >
class Match
{
public:
    virtual ~Match() {}
    void play( Game< MoveT > const& game, Player< MoveT >& player, Player< MoveT >& opponent )
    {
        if (game.is_drawn())
            drawn( game );
        else if (game.is_won())
            won( game );
        else
        {
            size_t move_index = player.choose( game );
            if (move_index >= game.valid_moves().size())
                throw std::runtime_error( "invalid move index" );
            auto next_game = game.apply( move_index );
            report( *next_game, game.valid_moves()[move_index]);

            play( *next_game, opponent, player );
        }
    }
protected:
    virtual void report( Game< MoveT > const& game, MoveT const& move ) = 0;
    virtual void drawn( Game< MoveT > const& ) = 0;
    virtual void won( Game< MoveT > const& ) = 0;
};
