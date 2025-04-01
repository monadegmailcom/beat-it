#include <functional>

#include "game.h"

namespace minimax
{

template< typename MoveT >
using ScoreFunction = double (*)(UndecidedGame< MoveT > const&);  

double max_value( PlayerIndex );
std::function< bool (double, double) > cmp( PlayerIndex );

template< typename MoveT >
double eval( Game const& game, ScoreFunction< MoveT > score, 
             unsigned depth )
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
    for (auto const& move : undecided_game.valid_moves())
    {
        auto next_game = undecided_game.apply( move );
        auto next_score = eval( *next_game, score, depth - 1 );
        if (compare( next_score, best_score ))
            best_score = next_score;
    }
    return best_score;
}

} // namespace minimax
