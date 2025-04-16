#include "game.h"

#include <functional>
#include <algorithm>
#include <random>

namespace minimax
{

template< typename MoveT >
using ScoreFunction = std::function< double (Game< MoveT > const&) >;  

double max_value( PlayerIndex );
std::function< bool (double, double) > cmp( PlayerIndex );

template< typename MoveT >
double eval( 
    Game< MoveT > const& game, ScoreFunction< MoveT > score, unsigned depth, 
    std::vector< MoveT >& move_stack, std::mt19937& g )
{
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

    const size_t prev_move_stack_size = move_stack.size();
    game.append_valid_moves( move_stack );
    // shuffle order of moves to avoid the same order every time
    std::shuffle( move_stack.begin() + prev_move_stack_size, move_stack.end(), g );

    for (size_t index = prev_move_stack_size, end = move_stack.size(); index != end; ++index)
    {
        auto next_score = eval( *game.apply( move_stack[index] ), score, depth - 1, move_stack, g );
        if (compare( next_score, best_score ))
            best_score = next_score;
    }

    move_stack.erase( move_stack.begin() + prev_move_stack_size, move_stack.end());

    return best_score;
}

template< typename MoveT >
class Player : public ::Player< MoveT >
{   
public:
    Player( unsigned depth, std::mt19937& g ) : depth( depth ), g( g ) {}
    virtual ~Player() {}
    virtual double score( Game< MoveT > const& ) = 0;
protected:
    unsigned depth;
    std::mt19937& g;
    std::vector< MoveT > move_stack;

    MoveT choose( Game< MoveT > const& game ) override
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
        const ScoreFunction< MoveT > score_function = [this](Game< MoveT > const& game)
            { return this->score( game ); };

        for (size_t index = prev_move_stack_size, end = move_stack.size(); index != end; ++index)
        {
            const double score = eval( 
                *game.apply( move_stack[index] ), score_function, depth, move_stack, g );
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
