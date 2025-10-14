#pragma once

#include "minimax.h"
#include "node.h"

#include <random>
#include <algorithm>

namespace minimax::tree {

template< typename MoveT, typename StateT >
struct Value
{
    Value( Game< MoveT, StateT > const& game, MoveT const& move )
    : game( game ), move( move ), game_result( game.result()) {}

    Value( Value&& other ) noexcept
        : game(std::move(other.game)), // Game is truly moved
          move(std::move(other.move)),
          game_result(other.game_result)
    {}

    Game< MoveT, StateT > game;
    MoveT move; // the previous move resulting in this game
    const GameResult game_result; // the cached game result
    double evaluation = 0.0;
};

template< typename MoveT, typename StateT >
using NodeAllocator = ::NodeAllocator< Value< MoveT, StateT > >;

template< typename MoveT, typename StateT >
class Player : public ::Player< MoveT >
{
public:
    using value_type = Value< MoveT, StateT >;
    using node_type = Node< value_type >; 
    Player( Game< MoveT, StateT > const& game, unsigned max_depth, unsigned seed,
        NodeAllocator< MoveT, StateT >& allocator )
    : max_depth( max_depth ), g( seed ), allocator( allocator )
    {
        allocator.rebase( allocator.allocate( value_type( game, MoveT())));
    }

    virtual double score( Game< MoveT, StateT > const&) const
    { return 0; };
private:
    unsigned max_depth;
    std::mt19937 g;
    NodeAllocator< MoveT, StateT >& allocator;
    std::vector< MoveT > move_stack;
    size_t eval_calls = 0;

    void apply_opponent_move( MoveT const& move ) override
    {
        auto root = allocator.get_root();
        auto itr =
            std::ranges::find_if(
                root->get_children(),
                [move](auto const& node)
                { return node.get_value().move == move; } );
        node_type& new_root = (itr == root->get_children().end())
            ? allocator.allocate( value_type(
                   root->get_value().game.apply( move ), move))
            : *itr;

        allocator.rebase( new_root );
    }

    double eval( node_type& node, unsigned depth, double alpha, double beta )
    {
        ++eval_calls;
        auto& value = node.get_value();

    if (GameResult result = value.game_result; result == GameResult::Draw)
            return 0.0;
        else if (result == GameResult::Player1Win)
            return max_value( PlayerIndex::Player1 );
        else if (result == GameResult::Player2Win)
            return max_value( PlayerIndex::Player2 );
        else if (!depth)
            return score( value.game );

        double best_score;
        std::function< bool (double, double) > compare;
        double* palpha;
        double const* pbeta;
        // minimizing player
        if (value.game.current_player_index() == PlayerIndex::Player1) 
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

        // push_front child nodes on first visit
        if (node.get_children().empty())
        {
            move_stack.clear();
            GameState< MoveT, StateT >::get_valid_moves(
                move_stack, value.game.current_player_index(), 
                value.game.get_state());
            std::ranges::shuffle( move_stack, g );

            for (MoveT const& move : move_stack)
                node.get_children().push_front( 
                    allocator.allocate(
                        value_type( value.game.apply( move ), move )));
        }
        // evaluate child nodes recursively until pruning
        auto child_itr = node.get_children().begin();
        for (;child_itr != node.get_children().end(); ++child_itr) // NOSONAR
        {
            child_itr->get_value().evaluation =
                 eval( *child_itr, depth - 1, alpha, beta );
            if (compare( child_itr->get_value().evaluation, best_score ))
                best_score = child_itr->get_value().evaluation;
            if (compare( best_score, *palpha ))
                *palpha = best_score;
            if (!compare( *pbeta, best_score ))
            {
                ++child_itr;
                break;
            }
        }

        // stable sort child nodes up to pruned child itr
        // for earlier pruning opportunities on next visit
        boost::intrusive::list< node_type > prefix;
        prefix.splice(
            prefix.begin(),
            node.get_children(),
            node.get_children().begin(),
            child_itr           // ...up to (but not including) child_itr
        );

        prefix.sort(
            [compare](auto const& a, auto const& b)
            { 
                return compare( 
                    a.get_value().evaluation, b.get_value().evaluation); 
            });

        node.get_children().splice(node.get_children().begin(), prefix);

        return best_score;
    }

    MoveT choose_move() override
    {
        auto root = allocator.get_root();

        if (root->get_value().game.result() != GameResult::Undecided)
            throw std::source_location::current();

        // eval with increasing depth
        // evaluation will benefit from better pruning from ordering of previous step
        // always start from level 0 because pruning my be different from last
        // time due to initialized alpha/beta start values
        for (size_t d = 0; d <= max_depth + 1; ++d)
            root->get_value().evaluation = eval( *root, d, -INFINITY, INFINITY );

        if (root->get_children().empty())
            throw std::source_location::current();

        allocator.rebase( *root->get_children().begin());

        return root->get_value().move;
    }
};

} // namespace minimax::tree
