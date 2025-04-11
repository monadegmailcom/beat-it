#include "game.h"

#include <functional>
#include <algorithm>
#include <random>

namespace minimax
{

template< typename MoveT >
using ScoreFunction = std::function< double (UndecidedGame< MoveT > const&) >;  

double max_value( PlayerIndex );
std::function< bool (double, double) > cmp( PlayerIndex );

template< typename MoveT >
double eval( Game const& game, ScoreFunction< MoveT > score, unsigned depth, std::mt19937& g )
{
    const PlayerIndex index = game.current_player_index();
    if (dynamic_cast< DrawnGame const* >( &game ))
        return 0.0;
    else if (dynamic_cast< WonGame const* >( &game ))
        return max_value( index );
    else if (depth == 0)
        return score( dynamic_cast< UndecidedGame< MoveT > const& >( game ));
    auto const& undecided_game = 
        dynamic_cast< UndecidedGame< MoveT > const& >( game );
    double best_score = max_value( toggle( index ));
    const auto compare = cmp( index );

    // shuffle iterators to the moves to avoid the same order every time
    std::vector< typename std::vector< MoveT >::const_iterator > moves( 
        undecided_game.valid_moves().size());
    auto dest = moves.begin();
    for (auto itr = undecided_game.valid_moves().begin();
            itr != undecided_game.valid_moves().end(); ++itr, ++dest)
        *dest = itr;
    std::shuffle( moves.begin(), moves.end(), g );

    for (auto move_itr : moves)
    {
        auto next_game = undecided_game.apply( move_itr );
        auto next_score = eval( *next_game, score, depth - 1, g );
        if (compare( next_score, best_score ))
            best_score = next_score;
    }
    return best_score;
}

template< typename MoveT >
class Player : public ::Player< MoveT >
{   
public:
    Player( PlayerIndex index, unsigned depth, std::mt19937& g ) 
        : ::Player< MoveT >( index ), depth( depth ), g( g ) {}
    virtual ~Player() {}
    virtual double score( UndecidedGame< MoveT > const& ) = 0;
protected:
    unsigned depth;
    std::mt19937& g;

    std::vector< MoveT >::const_iterator choose( 
        UndecidedGame< MoveT > const& game ) override
    {
        const auto compare = cmp( this->get_index());
        const ScoreFunction< MoveT > score_function = [this](UndecidedGame< MoveT > const& game)
            { return this->score( game ); };
        double best_score = max_value( toggle( this->get_index()));
        auto best_move = game.valid_moves().begin();

        // shuffle iterators to the moves to avoid the same order every time
        std::vector< typename std::vector< MoveT >::const_iterator > moves( game.valid_moves().size());
        auto dest = moves.begin();
        for (auto itr = game.valid_moves().begin();
             itr != game.valid_moves().end(); ++itr, ++dest)
            *dest = itr;
        std::shuffle( moves.begin(), moves.end(), g );

        for (auto move_itr : moves)
        {
            const double score = eval( *game.apply( move_itr ), score_function, depth, g );
            if (compare( score, best_score ))
            {
                best_score = score;
                best_move = move_itr;
            }
        }
            
        return best_move;
    }
};

} // namespace minimax
