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

template< typename MoveT, typename StateT >
class Player : public ::Player< MoveT, StateT >
{   
public:
    Player( unsigned depth, std::mt19937& g ) : depth( depth ), g( g ) {}
    virtual ~Player() {}
    virtual double score( Game< MoveT, StateT > const& ) const { return 0.0; };
    std::vector< MoveT > const& get_move_stack() { return move_stack; }
    size_t get_eval_calls() const { return eval_calls; }
    double get_best_score() const { return best_score; }
protected:
    unsigned depth;
    std::mt19937& g;
    std::vector< MoveT > move_stack;
    size_t eval_calls = 0;
    double best_score;

    MoveT choose( Game< MoveT, StateT > const& game ) override
    {
        const size_t prev_move_stack_size = move_stack.size();
        game.append_valid_moves( move_stack );
        if (prev_move_stack_size == move_stack.size())
            throw std::invalid_argument( "no valid moves" );

        const auto compare = cmp( game.current_player_index());
        const ScoreFunction< MoveT, StateT > score_function = 
            [this](Game< MoveT, StateT > const& game) { return this->score( game ); };

        // shuffle order of moves to avoid the same order every time
        std::shuffle( move_stack.begin() + prev_move_stack_size, move_stack.end(), g );

        best_score = max_value( toggle( game.current_player_index()));
        size_t best_move_index = 0;

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
