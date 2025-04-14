#include "game.h"

#include <functional>
#include <algorithm>

namespace minimax
{

template< typename MoveT >
using ScoreFunction = std::function< double (Game< MoveT > const&) >;  

double max_value( PlayerIndex );
std::function< bool (double, double) > cmp( PlayerIndex );

template< typename MoveT >
double eval( 
    Game< MoveT > const& game, ScoreFunction< MoveT > score, unsigned depth )
{
    const PlayerIndex index = game.current_player_index();
    if (game.is_drawn())
        return 0.0;
    else if (game.is_won())
        return max_value( index );
    else if (depth == 0)
        return score( game );
    double best_score = max_value( toggle( index ));
    const auto compare = cmp( index );

    for (size_t index = 0, end = game.valid_moves().size(); index != end; ++index)
    {
        auto next_game = game.apply( index );
        auto next_score = eval( *next_game, score, depth - 1 );
        if (compare( next_score, best_score ))
            best_score = next_score;
    }

    return best_score;
}

template< typename MoveT >
class Player : public ::Player< MoveT >
{   
public:
    Player( unsigned depth ) : depth( depth ) {}
    virtual ~Player() {}
    virtual double score( Game< MoveT > const& ) = 0;
protected:
    unsigned depth;

    size_t choose( Game< MoveT > const& game ) override
    {
        const auto compare = cmp( game.current_player_index());
        double best_score = max_value( toggle( game.current_player_index()));
        const ScoreFunction< MoveT > score_function = [this](Game< MoveT > const& game)
            { return this->score( game ); };
        size_t best_move_index = 0;

        for (size_t index = 0, end = game.valid_moves().size(); index != end; ++index)
        {
            auto next_game = game.apply( index );
            const double score = eval( *next_game, score_function, depth );
            if (compare( score, best_score ))
            {
                best_score = score;
                best_move_index = index;
            }
        }
            
        return best_move_index;
    }
};

} // namespace minimax
