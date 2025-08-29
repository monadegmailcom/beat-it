#include "player.h"

#include <atomic>
#include <chrono>
#include <random>

template< typename MoveT, typename StateT >
class Match
{
public:
    virtual ~Match() = default;
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
            std::swap( pl, op );
            std::swap( pl_dur, op_dur );
            // report the game move after the move is applied
            report( game, move );
        }
    }
protected:
    virtual void report( Game< MoveT, StateT > const&, MoveT const& )
    { /* do nothing on default */};
};

template< typename MoveT, typename StateT >
class MultiMatch : public ::Match< MoveT, StateT >
{
public:
    // we need a new player each round, so we use factories
    MultiMatch(
        Game< MoveT, StateT > const& game,
        PlayerFactory< MoveT > fst_player_factory,
        PlayerFactory< MoveT > snd_player_factory,
        int rounds,
        size_t number_of_threads,
        unsigned seed)
    : rounds(rounds),
      g(seed),
      game(game),
      fst_player_factory(fst_player_factory),
      snd_player_factory(snd_player_factory),
      number_of_threads(number_of_threads)
    {}

    void run()
    {
        std::vector< std::thread > threads( number_of_threads );
        for (std::thread& thread : threads)
            thread = std::thread(
                [this]() { worker(); });
        for (auto& thread : threads)
            thread.join();
    }

    size_t get_draws() const
    {
        std::lock_guard< std::mutex > lock( mutex );
        return draws;
    }
    size_t get_fst_player_wins() const
    {
        std::lock_guard< std::mutex > lock( mutex );
        return fst_player_wins;
    }
    size_t get_snd_player_wins() const
    {
        std::lock_guard< std::mutex > lock( mutex );
        return snd_player_wins;
    }
    std::chrono::microseconds get_fst_player_duration() const
    {
        std::lock_guard< std::mutex > lock( mutex );
        return fst_player_duration;
    }
    std::chrono::microseconds get_snd_player_duration() const
    {
        std::lock_guard< std::mutex > lock( mutex );
        return snd_player_duration;
    }
private:
    void worker()
    {
        size_t tls_draws = 0;
        size_t tls_fst_player_wins = 0;
        size_t tls_snd_player_wins = 0;

        std::chrono::microseconds tls_fst_player_duration {0};
        std::chrono::microseconds tls_snd_player_duration {0};

        /* Pointers to the factories and their corresponding TLS win/duration
            counters. These will be swapped each round to alternate which
            factory goes first. */
        auto* p_fst_factory = &fst_player_factory;
        auto* p_snd_factory = &snd_player_factory;
        auto* p_fst_wins = &tls_fst_player_wins;
        auto* p_snd_wins = &tls_snd_player_wins;
        auto* p_fst_duration = &tls_fst_player_duration;
        auto* p_snd_duration = &tls_snd_player_duration;

        // The starting player index also alternates each round.
        PlayerIndex current_starter_index = game.current_player_index();

        while (rounds.fetch_sub(1, std::memory_order_relaxed) > 0)
        {
            // For each game, create a new Game instance with the correct
            // starting player, but with the original game's initial state.
            Game<MoveT, StateT> round_game(
                current_starter_index, game.get_state());

            const unsigned round_seed = g();
            const GameResult game_result = this->play(
                round_game,
                *(std::unique_ptr< Player< MoveT > >((*p_fst_factory)(
                    round_seed ))), *p_fst_duration,
                *(std::unique_ptr< Player< MoveT > >((*p_snd_factory)(
                    round_seed ))), *p_snd_duration );

            if (game_result == GameResult::Draw)
                ++tls_draws;
            // Check if the winner matches the starting player of the round.
            else if (game_result == (current_starter_index == Player1
                        ? GameResult::Player1Win
                        : GameResult::Player2Win))
                (*p_fst_wins)++; // The first player for this round won.
            else
                (*p_snd_wins)++; // The second player for this round won.

            // Swap the roles for the next round.
            std::swap(p_fst_factory, p_snd_factory);
            std::swap(p_fst_wins, p_snd_wins);
            std::swap(p_fst_duration, p_snd_duration);
            current_starter_index = toggle(current_starter_index);
        }

        // accumulate results thread safe
        std::lock_guard< std::mutex > lock( mutex );

        draws += tls_draws;
        fst_player_wins += tls_fst_player_wins;
        snd_player_wins += tls_snd_player_wins;
        fst_player_duration += tls_fst_player_duration;
        snd_player_duration += tls_snd_player_duration;
    }

    mutable std::mutex mutex;
    // has to be a signed type because we test for positivity
    std::atomic< int > rounds;
    std::mt19937 g;
    size_t draws = 0;
    size_t fst_player_wins = 0;
    size_t snd_player_wins = 0;
    std::chrono::microseconds fst_player_duration {0};
    std::chrono::microseconds snd_player_duration {0};

    Game< MoveT, StateT > const& game;
    PlayerFactory< MoveT > fst_player_factory;
    PlayerFactory< MoveT > snd_player_factory;
    size_t number_of_threads;
};
