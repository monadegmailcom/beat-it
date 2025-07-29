#include "player.h"

#include <atomic>
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
    // we need a new player each round, so we use factories
    MultiMatch(
        Game< MoveT, StateT > const& game,
        PlayerFactory< MoveT > fst_player_factory,
        PlayerFactory< MoveT > snd_player_factory,
        int rounds,
        size_t number_of_threads )
    : rounds( rounds )
    {
        std::vector< std::thread > threads( number_of_threads );
        for (std::thread& thread : threads)
            thread = std::thread(
                [this, &game, fst_player_factory, snd_player_factory]()
                { worker( game, fst_player_factory, snd_player_factory ); });
        for (auto& thread : threads)
            thread.join();
    }

    void worker(
        Game< MoveT, StateT > const& game,
        PlayerFactory< MoveT > fst_player_factory,
        PlayerFactory< MoveT > snd_player_factory )
    {
        size_t tls_draws = 0;
        size_t tls_fst_player_wins = 0;
        size_t tls_snd_player_wins = 0;

        std::chrono::microseconds tls_fst_player_duration {0};
        std::chrono::microseconds tls_snd_player_duration {0};

        PlayerIndex fst_player_index = game.current_player_index();
        PlayerIndex snd_player_index = toggle( fst_player_index );

        auto player_factory = &fst_player_factory;
        auto player_duration = &fst_player_duration;

        auto opponent_factory = &snd_player_factory;
        auto opponent_duration = &snd_player_duration;

        while (rounds.fetch_sub(1, std::memory_order_relaxed) > 0)
        {
            const GameResult game_result = this->play(
                game, *(std::unique_ptr< Player< MoveT > >((*player_factory)())), *player_duration,
                *(std::unique_ptr< Player< MoveT > >((*opponent_factory)())), *opponent_duration );
            if (game_result == GameResult::Draw)
                ++tls_draws;
            else if (game_result == GameResult::Player1Win)
                fst_player_index == Player1
                    ? ++tls_fst_player_wins
                    : ++tls_snd_player_wins;
            else if (game_result == GameResult::Player2Win)
                 fst_player_index == Player2
                    ? ++tls_fst_player_wins
                    : ++tls_snd_player_wins;

            std::swap( player_factory, opponent_factory );
            std::swap( player_duration, opponent_duration );
            std::swap( fst_player_index, snd_player_index );
        }

        // accumulate results thread safe
        std::lock_guard< std::mutex > lock( mutex );

        draws += tls_draws;
        fst_player_wins += tls_fst_player_wins;
        snd_player_wins += tls_snd_player_wins;
        fst_player_duration += tls_fst_player_duration;
        snd_player_duration += tls_snd_player_duration;
    }

    std::mutex mutex;
    // has to be a signed type because we test for positiv
    std::atomic< int > rounds;
    size_t draws = 0;
    size_t fst_player_wins = 0;
    size_t snd_player_wins = 0;
    std::chrono::microseconds fst_player_duration {0};
    std::chrono::microseconds snd_player_duration {0};
};

