#pragma once

#include "game.h"
#include "player.h"

#include <algorithm>
#include <functional>
#include <random>
#include <source_location>

namespace minimax
{

template < typename MoveT, typename StateT >
using ScoreFunction = std::function< double( Game< MoveT, StateT > const & ) >;

double max_value( PlayerIndex );
std::function< bool( double, double ) > cmp( PlayerIndex );

template < typename MoveT, typename StateT >
double eval( Game< MoveT, StateT > game, ScoreFunction< MoveT, StateT > score,
             unsigned depth, double alpha, double beta, std::mt19937 &g,
             size_t &calls )
{
    ++calls;

    if ( GameResult result = game.result(); result == GameResult::Draw )
        return 0.0;
    else if ( result == GameResult::Player1Win )
        return max_value( PlayerIndex::Player1 );
    else if ( result == GameResult::Player2Win )
        return max_value( PlayerIndex::Player2 );
    else if ( depth == 0 )
        return score( game );

    double best_score;
    std::function< bool( double, double ) > compare;
    double *palpha;
    double const *pbeta;
    // minimizing player
    if ( game.current_player_index() == PlayerIndex::Player1 )
    {
        best_score = INFINITY;
        compare = std::less< double >();
        palpha = &beta;
        pbeta = &alpha;
    }
    else // maximizing player
    {
        best_score = -INFINITY;
        compare = std::greater< double >();
        palpha = &alpha;
        pbeta = &beta;
    }

    for ( MoveT const &move : game )
    {
        if ( auto next_score = eval( game.apply( move ), score, depth - 1,
                                     alpha, beta, g, calls );
             compare( next_score, best_score ) )
            best_score = next_score;
        if ( compare( best_score, *palpha ) )
            *palpha = best_score;
        if ( !compare( *pbeta, best_score ) )
            break;
    }

    return best_score;
}

template < typename MoveT, typename StateT >
class Player : public ::Player< MoveT >
{
  public:
    Player( Game< MoveT, StateT > const &game, unsigned depth, unsigned seed )
        : game( game ), depth( depth ), g( seed )
    {
    }
    virtual double score( Game< MoveT, StateT > const & ) const { return 0.0; };

  private:
    Game< MoveT, StateT > game;
    unsigned depth;
    std::mt19937 g;
    std::vector< MoveT > move_stack;
    size_t eval_calls = 0;
    double best_score = 0.0;

    void apply_opponent_move( MoveT const &move ) override
    {
        game = game.apply( move );
    }

    MoveT choose_move() override
    {
        const auto compare = cmp( game.current_player_index() );
        const ScoreFunction< MoveT, StateT > score_function =
            [this]( Game< MoveT, StateT > const &ga )
        { return this->score( ga ); };

        if ( game.result() != GameResult::Undecided )
            throw std::source_location::current();

        GameState< MoveT, StateT >::get_valid_moves(
            move_stack, game.current_player_index(), game.get_state() );

        // shuffle order of moves to avoid the same order every time
        std::ranges::shuffle( move_stack, g );

        best_score = max_value( toggle( game.current_player_index() ) );
        if ( move_stack.empty() )
            throw std::invalid_argument( "no valid moves" );
        MoveT const *best_move = &move_stack.front();

        for ( MoveT const &move : move_stack )
        {
            const double score =
                eval( game.apply( move ), score_function, depth, -INFINITY,
                      INFINITY, g, eval_calls );
            if ( compare( score, best_score ) )
            {
                best_score = score;
                best_move = &move;
            }
        }

        game = game.apply( *best_move );
        return *best_move;
    }
};

} // namespace minimax
