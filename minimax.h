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
    std::vector< MoveT >& move_stack, double alpha, double beta, std::mt19937& g, size_t& calls )
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

    const size_t prev_move_stack_size = move_stack.size();
    game.append_valid_moves( move_stack );

    for (size_t i = prev_move_stack_size, end = move_stack.size(); i != end; ++i)
    {
        auto next_score = eval( 
            game.apply( move_stack[i] ), score, depth - 1, move_stack, alpha, beta, g, calls );
        if (compare( next_score, best_score ))
            best_score = next_score;
        if (compare( best_score, *palpha ))
            *palpha = best_score;
        if (!compare( *pbeta, best_score ))
            break;
    }

    move_stack.erase( move_stack.begin() + prev_move_stack_size, move_stack.end());

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
        const size_t prev_move_stack_size = data.move_stack.size();
        game.append_valid_moves( data.move_stack );
        if (prev_move_stack_size == data.move_stack.size())
            throw std::invalid_argument( "no valid moves" );

        const auto compare = cmp( game.current_player_index());
        const ScoreFunction< MoveT, StateT > score_function = 
            [this](Game< MoveT, StateT > const& game) { return this->score( game ); };

        // shuffle order of moves to avoid the same order every time
        std::shuffle( 
            data.move_stack.begin() + prev_move_stack_size, data.move_stack.end(), 
            data.g );

        data.best_score = max_value( toggle( game.current_player_index()));
        size_t best_move_index = 0;

        for (size_t index = prev_move_stack_size, end = data.move_stack.size(); index != end; ++index)
        {
            const double score = eval( 
                game.apply( data.move_stack[index] ), score_function, depth, data.move_stack, 
                -INFINITY, INFINITY, data.g, data.eval_calls );
            if (compare( score, data.best_score ))
            {
                data.best_score = score;
                best_move_index = index;
            }
        }
            
        const MoveT best_move = data.move_stack[best_move_index];
        data.move_stack.erase( 
            data.move_stack.begin() + prev_move_stack_size, data.move_stack.end());

        game = game.apply( best_move );
        return best_move;
    }
};

template< typename MoveT, typename StateT >
using Buffer = char[sizeof( Player< MoveT, StateT > )];

template< typename MoveT, typename StateT >
PlayerFactory< MoveT > player_factory(
    Game< MoveT, StateT > const&, unsigned depth, Data< MoveT >& data, Buffer< MoveT, StateT > );

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
