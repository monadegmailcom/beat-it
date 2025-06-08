#include "player.h"

#include <chrono>

template< typename MoveT, typename StateT >
class Match
{
public:
    virtual ~Match() {}
    GameResult play( 
        Game< MoveT, StateT > game, 
        Player< MoveT >& player,
        std::chrono::microseconds& player_duration, 
        Player< MoveT >& opponent,
        std::chrono::microseconds& opponent_duration )
    {
        Player< MoveT >* pl = &player;
        std::chrono::microseconds* pl_dur = &player_duration;
        Player< MoveT >* op = &opponent;
        std::chrono::microseconds* op_dur = &opponent_duration;
        for (;;)
        {
            const GameResult result = game.result();
            if (result != GameResult::Undecided)
                return result;

            auto start = std::chrono::steady_clock::now();
            const MoveT move = pl->choose_move();
            *pl_dur += std::chrono::duration_cast< std::chrono::microseconds >( 
                   std::chrono::steady_clock::now() - start );
            game = game.apply( move );
            start = std::chrono::steady_clock::now();
            op->apply_opponent_move( move );
            *op_dur += std::chrono::duration_cast< std::chrono::microseconds >( 
                       std::chrono::steady_clock::now() - start );
            report( game, move );
            std::swap( pl, op );
            std::swap( pl_dur, op_dur );
        }
    }
protected:
    virtual void report( Game< MoveT, StateT > const&, MoveT const& ) {};
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
        auto player_duration = &fst_player_duration;

        auto opponent_factory = &snd_player_factory;
        auto opponent_duration = &snd_player_duration;
        
        fst_player_index = game.current_player_index();
        snd_player_index = toggle( fst_player_index );

        for (; rounds > 0; --rounds)
        {
            const GameResult game_result = this->play( 
                game, *(std::unique_ptr< Player< MoveT > >((*player_factory)())), *player_duration, 
                *(std::unique_ptr< Player< MoveT > >((*opponent_factory)())), *opponent_duration );
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
            std::swap( player_duration, opponent_duration );
            std::swap( fst_player_index, snd_player_index );
        }
    }

    size_t draws = 0;
    size_t fst_player_wins = 0;
    size_t snd_player_wins = 0;
    PlayerIndex fst_player_index;
    PlayerIndex snd_player_index;
    std::chrono::microseconds fst_player_duration {0};
    std::chrono::microseconds snd_player_duration {0};
};

