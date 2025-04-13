#include "game.h"

template< typename MoveT >
class Match
{
public:
    virtual ~Match() {}
    void play( Game const& game, Player< MoveT >& player, Player< MoveT >& opponent )
    {
        if (auto drawn_game = dynamic_cast< DrawnGame const* >( &game ); drawn_game)
            drawn( *drawn_game );
        else if (auto won_game = dynamic_cast< WonGame const* >( &game ); won_game)
            won( *won_game );
        else
        {
            UndecidedGame< MoveT > const& undecided_game = 
                dynamic_cast< UndecidedGame< MoveT > const& >( game );

            size_t move_index = player.choose( undecided_game );
            if (move_index >= undecided_game.valid_moves().size())
                throw std::runtime_error( "invalid move index" );
            auto next_game = undecided_game.apply( move_index );
            report( *next_game, undecided_game.valid_moves()[move_index]);

            play( *next_game, opponent, player );
        }
    }
protected:
    virtual void report( Game const& game, MoveT const& move ) = 0;
    virtual void drawn( DrawnGame const& ) = 0;
    virtual void won( WonGame const& ) = 0;
};
