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
double eval( 
    Game const& game, ScoreFunction< MoveT > score, unsigned depth, std::mt19937& g, 
    std::vector< size_t >& move_indices )
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
    const size_t prev_size = move_indices.size();
    move_indices.resize( prev_size + undecided_game.valid_moves().size());
    std::generate( move_indices.begin() + prev_size, move_indices.end(), [n = 0]() mutable { return n++; });
    std::shuffle( move_indices.begin() + prev_size, move_indices.end(), g );

    for (size_t i = prev_size, size = move_indices.size(); i != size; ++i)
    {
        auto next_game = undecided_game.apply( move_indices[i]);
        auto next_score = eval( *next_game, score, depth - 1, g, move_indices );
        if (compare( next_score, best_score ))
            best_score = next_score;
    }
    move_indices.resize( prev_size );
    return best_score;
}

template< typename MoveT >
class Player : public ::Player< MoveT >
{   
public:
    Player( unsigned depth, std::mt19937& g ) : depth( depth ), g( g ) {}
    virtual ~Player() {}
    virtual double score( UndecidedGame< MoveT > const& ) = 0;
    std::vector< size_t > const& get_move_indices() const { return move_indices; }
protected:
    unsigned depth;
    std::mt19937& g;
    std::vector< size_t > move_indices;

    size_t choose( UndecidedGame< MoveT > const& game ) override
    {
        const auto compare = cmp( game.current_player_index());
        double best_score = max_value( toggle( game.current_player_index()));
        const ScoreFunction< MoveT > score_function = [this](UndecidedGame< MoveT > const& game)
            { return this->score( game ); };
        size_t best_move_index = 0;

        // shuffle order of moves to avoid the same order every time
        move_indices.resize( game.valid_moves().size());
        std::generate( move_indices.begin(), move_indices.end(), [n = 0]() mutable { return n++; });
        std::shuffle( move_indices.begin(), move_indices.end(), g );

        for (size_t i = 0, size = move_indices.size(); i != size; ++i)
        {
            const size_t move_index = move_indices[i];
            auto next_game = game.apply( move_index );
            const double score = eval( *next_game, score_function, depth, g, move_indices );
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
