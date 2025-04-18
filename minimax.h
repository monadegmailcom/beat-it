#include "player.h"

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

    const PlayerIndex index = game.current_player_index();
    if (game.result() == GameResult::Draw)
        return 0.0;
    else if (game.result() == GameResult::Player1Win)
        return max_value( Player1 );
    else if (game.result() == GameResult::Player2Win)
        return max_value( Player2 );
    else if (depth == 0)
        return score( game );
        
    double best_score = max_value( toggle( index ));
    const auto compare = cmp( index );
    double* palpha = nullptr;
    double* pbeta = nullptr;
    if (index == Player1) // minimizing player
    {
        palpha = &beta;
        pbeta = &alpha;
    }
    else // maximizing player
    {
        palpha = &alpha;
        pbeta = &beta;
    }

    const size_t prev_move_stack_size = move_stack.size();
    game.append_valid_moves( move_stack );
    // shuffle order of moves to avoid the same order every time
    std::shuffle( move_stack.begin() + prev_move_stack_size, move_stack.end(), g );

    for (size_t index = prev_move_stack_size, end = move_stack.size(); index != end; ++index)
    {
        auto next_score = eval( 
            game.apply( move_stack[index] ), score, depth - 1, move_stack, alpha, beta, g, calls );
        if (compare( next_score, best_score ))
            best_score = next_score;
        if (!compare( *pbeta, best_score ))
            break;
        if (compare( best_score, *palpha ))
            *palpha = best_score;
    }

    move_stack.erase( move_stack.begin() + prev_move_stack_size, move_stack.end());

    return best_score;
}

template< typename MoveT, typename StateT >
class Player : public ::Player< MoveT, StateT >
{   
public:
    Player( unsigned depth, std::mt19937& g ) : depth( depth ), g( g ) {}
    virtual ~Player() {}
    virtual double score( Game< MoveT, StateT > const& ) const { return 0.0; };
    std::vector< MoveT > const& get_move_stack() { return move_stack; }
    size_t get_eval_calls() const { return eval_calls; }
protected:
    unsigned depth;
    std::mt19937& g;
    std::vector< MoveT > move_stack;
    size_t eval_calls = 0;

    MoveT choose( Game< MoveT, StateT > const& game ) override
    {
        const size_t prev_move_stack_size = move_stack.size();
        game.append_valid_moves( move_stack );
        if (prev_move_stack_size == move_stack.size())
            throw std::invalid_argument( "no valid moves" );

        // shuffle order of moves to avoid the same order every time
        std::shuffle( move_stack.begin() + prev_move_stack_size, move_stack.end(), g );

        const auto compare = cmp( game.current_player_index());
        double best_score = max_value( toggle( game.current_player_index()));
        size_t best_move_index = 0;
        const ScoreFunction< MoveT, StateT > score_function = 
            [this](Game< MoveT, StateT > const& game) { return this->score( game ); };

        for (size_t index = prev_move_stack_size, end = move_stack.size(); index != end; ++index)
        {
            const double score = eval( 
                game.apply( move_stack[index] ), score_function, depth, move_stack, 
                -INFINITY, INFINITY, g, eval_calls );
            if (compare( score, best_score ))
            {
                best_score = score;
                best_move_index = index;
            }
        }
            
        const MoveT best_move = move_stack[best_move_index];
        move_stack.erase( move_stack.begin() + prev_move_stack_size, move_stack.end());
        return best_move;
    }
};

} // namespace minimax
