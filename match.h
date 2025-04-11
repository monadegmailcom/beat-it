#include "game.h"

template< typename MoveT >
class Match
{
public:
    virtual ~Match() {}
    // require: both player has to have different indices, the game's player index chooses the 
    //          player to start
    void play(Game const& game, Player< MoveT >& player1, Player< MoveT >& player2)
    {
        if (player1.get_index() == player2.get_index())
            throw std::invalid_argument( "both players have the same index" );

        Player< MoveT >* player = nullptr;
        Player< MoveT >* opponent = nullptr;

        if (game.current_player_index() == player1.get_index())
        {
            player = &player1;
            opponent = &player2;
        }
        else
        {
            player = &player2;
            opponent = &player1;
        }

        std::unique_ptr< Game > next_game;
        Game const* current_game = &game;
        while (true)
        {
            if (auto drawn_game = dynamic_cast< DrawnGame const* >( current_game ); drawn_game)
            {
                drawn( *drawn_game );
                break;
            }
            else if (auto won_game = dynamic_cast< WonGame const* >( current_game ); won_game)
            {
                won( *won_game );
                break;
            }

            UndecidedGame< MoveT > const& undecided_game = 
                dynamic_cast< UndecidedGame< MoveT > const& >( *current_game );

            auto move_itr = player->choose( undecided_game );
            if (move_itr == undecided_game.valid_moves().end())
                throw std::invalid_argument( "invalid move" );

            // by reassigning next_game, the current game is destroyed, 
            // so copy the move before applying
            const MoveT move = *move_itr;
            next_game = undecided_game.apply( move_itr );
            report( *next_game, move );

            current_game = next_game.get();
            std::swap( player, opponent );
        }
    }
protected:
    virtual void report( Game const& game, MoveT const& move ) = 0;
    virtual void drawn( DrawnGame const& ) = 0;
    virtual void won( WonGame const& ) = 0;
};
