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

    // shuffle order of moves to avoid the same order every time
    std::vector< size_t > move_indices( undecided_game.valid_moves().size());
    std::generate( move_indices.begin(), move_indices.end(), [n = 0]() mutable { return n++; });
    std::shuffle( move_indices.begin(), move_indices.end(), g );

    for (auto move_index : move_indices)
    {
        auto next_game = undecided_game.apply( move_index );
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
    Player( unsigned depth, std::mt19937& g ) : depth( depth ), g( g ) {}
    virtual ~Player() {}
    virtual double score( UndecidedGame< MoveT > const& ) = 0;
protected:
    unsigned depth;
    std::mt19937& g;

    size_t choose( UndecidedGame< MoveT > const& game ) override
    {
        const auto compare = cmp( game.current_player_index());
        double best_score = max_value( toggle( game.current_player_index()));
        const ScoreFunction< MoveT > score_function = [this](UndecidedGame< MoveT > const& game)
            { return this->score( game ); };
        size_t best_move_index = 0;

        // shuffle order of moves to avoid the same order every time
        std::vector< size_t > move_indices( game.valid_moves().size());
        std::generate( move_indices.begin(), move_indices.end(), [n = 0]() mutable { return n++; });
        std::shuffle( move_indices.begin(), move_indices.end(), g );

        for (auto move_index : move_indices)
        {
            auto next_game = game.apply( move_index );
            const double score = eval( *next_game, score_function, depth, g );
            if (compare( score, best_score ))
            {
                best_score = score;
                best_move_index = move_index;
            }
        }
            
        return best_move_index;
    }
};

} // namespace minimax
