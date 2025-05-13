#include "player.h"

template< typename MoveT, typename StateT >
class Match
{
public:
    virtual ~Match() {}
    GameResult play( 
        Game< MoveT, StateT > game, 
        Player< MoveT >& player, 
        Player< MoveT >& opponent )
    {
        const GameResult result = game.result();
        if (result != GameResult::Undecided)
            return result;
        else
        {
            const MoveT move = player.choose_move();
            const Game next_game = game.apply( move );
            opponent.apply_opponent_move( move );
            report( next_game, move );
            return play( next_game, opponent, player );
        }
    }
protected:
    virtual void report( Game< MoveT, StateT > const& game, MoveT const& move ) = 0;
};

template< typename MoveT, typename StateT >
struct MultiMatch : public ::Match< MoveT, StateT >
{
    void report( Game< MoveT, StateT > const&, MoveT const& ) override
    {}

    // we need a new player each round, so we use a factory
    void play_match( 
        Game< MoveT, StateT > const& game, 
        PlayerFactory< MoveT > fst_player_factory, 
        PlayerFactory< MoveT > snd_player_factory, 
        size_t rounds )
    {
        draws = 0;
        fst_player_wins = 0;
        snd_player_wins = 0;
        
        auto player_factory = &fst_player_factory;
        auto opponent_factory = &snd_player_factory;
        
        fst_player_index = game.current_player_index();
        snd_player_index = toggle( fst_player_index );

        for (; rounds > 0; --rounds)
        {
            const GameResult game_result = this->play( 
                game, *(*player_factory)(), *(*opponent_factory)());
            if (game_result == GameResult::Draw)
                ++draws;
            else if (game_result == GameResult::Player1Win)
                fst_player_index == Player1
                    ? ++fst_player_wins
                    : ++snd_player_wins;
            else if (game_result == GameResult::Player2Win)
                 fst_player_index == Player2
                    ? ++fst_player_wins
                    : ++snd_player_wins;
            std::swap( player_factory, opponent_factory );
            std::swap( fst_player_index, snd_player_index );
        }
    }

    size_t draws = 0;
    size_t fst_player_wins = 0;
    size_t snd_player_wins = 0;
    PlayerIndex fst_player_index;
    PlayerIndex snd_player_index;
};

