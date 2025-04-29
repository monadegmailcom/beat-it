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

template< typename MoveT, typename StateT >
struct MultiMatch : public ::Match< MoveT, StateT >
{
    void report( Game< MoveT, StateT > const&, MoveT const& ) override
    {}
    void draw( Game< MoveT, StateT > const& ) override
    {
        ++draws;
    }
    void player1_win( Game< MoveT, StateT > const& ) override
    {
        if (fst_player_index == Player1)
            ++fst_player_wins;
        else
            ++snd_player_wins;
    }
    void player2_win( Game< MoveT, StateT > const& ) override
    {
        if (fst_player_index == Player2)
            ++fst_player_wins;
        else
            ++snd_player_wins;
    }
    void play_match( Game< MoveT, StateT > const& game, Player< MoveT, StateT >& fst_player, 
                     Player< MoveT, StateT >& snd_player, size_t rounds )
    {
        draws = 0;
        fst_player_wins = 0;
        snd_player_wins = 0;
        
        Player< MoveT, StateT >* player = &fst_player;
        Player< MoveT, StateT >* opponent = &snd_player;
        
        fst_player_index = game.current_player_index();
        snd_player_index = toggle( fst_player_index );

        for (; rounds > 0; --rounds)
        {
            this->play( game, *player, *opponent );
            std::swap( player, opponent );
            std::swap( fst_player_index, snd_player_index );
        }
    }

    size_t draws = 0;
    size_t fst_player_wins = 0;
    size_t snd_player_wins = 0;
    PlayerIndex fst_player_index;
    PlayerIndex snd_player_index;
};

