#pragma once

#include "player.h"
#include "game.h"

#include <functional>
#include <algorithm>
#include <random>

namespace minimax
{

template< typename MoveT, typename StateT >
using ScoreFunction = std::function< double (Game< MoveT, StateT > const&) >;  

double max_value( PlayerIndex );
std::function< bool (double, double) > cmp( PlayerIndex );

template< typename MoveT, typename StateT >
double eval( 
    Game< MoveT, StateT > game, ScoreFunction< MoveT, StateT > score, unsigned depth, 
    double alpha, double beta, std::mt19937& g, size_t& calls )
{
    ++calls;

    const GameResult result = game.result();
    if (result == GameResult::Draw)
        return 0.0;
    else if (result == GameResult::Player1Win)
        return max_value( Player1 );
    else if (result == GameResult::Player2Win)
        return max_value( Player2 );
    else if (depth == 0)
        return score( game );
        
    double best_score;
    std::function< bool (double, double) > compare;
    double* palpha;
    double* pbeta;
    if (game.current_player_index() == Player1) // minimizing player
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

    for (MoveT const& move : game)
    {
        auto next_score = eval( 
            game.apply( move ), score, depth - 1, alpha, beta, g, calls );
        if (compare( next_score, best_score ))
            best_score = next_score;
        if (compare( best_score, *palpha ))
            *palpha = best_score;
        if (!compare( *pbeta, best_score ))
            break;
    }

    return best_score;
}

template< typename MoveT >
struct Data
{
    Data( std::mt19937& g ) : g( g ) {}
    std::mt19937& g;
    std::vector< MoveT > move_stack;
    size_t eval_calls = 0;
    double best_score = 0.0;
};

template< typename MoveT, typename StateT >
class Player : public ::Player< MoveT >
{   
public:
    Player( Game< MoveT, StateT > const& game, unsigned depth, Data< MoveT >& data ) 
    : game( game ), depth( depth ), data( data ) {}
    virtual double score( Game< MoveT, StateT > const&) const { return 0.0; };
protected:
    Game< MoveT, StateT > game;
    unsigned depth;
    Data< MoveT >& data;

    void apply_opponent_move( MoveT const& move ) override
    {
        game = game.apply( move );
    }

    MoveT choose_move() override
    {
        const auto compare = cmp( game.current_player_index());
        const ScoreFunction< MoveT, StateT > score_function = 
            [this](Game< MoveT, StateT > const& game) { return this->score( game ); };

        if (game.result() != GameResult::Undecided)
            throw std::runtime_error( "game already finished" );
            
        GameState< MoveT, StateT >::get_valid_moves( 
            data.move_stack, game.current_player_index(), game.get_state());
        
        // shuffle order of moves to avoid the same order every time
        std::ranges::shuffle( data.move_stack, data.g );

        data.best_score = max_value( toggle( game.current_player_index()));
        if (data.move_stack.empty())
            throw std::invalid_argument( "no valid moves" );
        MoveT const* best_move = &data.move_stack.front();

        for (MoveT const& move : data.move_stack)
        {
            const double score = eval( 
                game.apply( move ), score_function, depth, 
                -INFINITY, INFINITY, data.g, data.eval_calls );
            if (compare( score, data.best_score ))
            {
                data.best_score = score;
                best_move = &move;
            }
        }
            
        game = game.apply( *best_move );
        return *best_move;
    }
};

template< typename MoveT, typename StateT >
using Buffer = char[sizeof( Player< MoveT, StateT > )];

template< typename MoveT, typename StateT >
PlayerFactory< MoveT > player_factory(
    Game< MoveT, StateT > const& game, unsigned depth, Data< MoveT >& data, Buffer< MoveT, StateT > raw_data )
{
    return [&game, &data, depth, raw_data]()
    { 
        return std::unique_ptr< ::Player< MoveT >, void(*)( ::Player< MoveT >*) >( 
            new (raw_data) Player< MoveT, StateT >( game, depth, data ), 
            [](::Player< MoveT >* p){p->~Player(); }); 
    };
}

} // namespace minimax
