#pragma once

#include "minimax.h"
#include "node.h"
#include "exception.h"

#include <random>
#include <algorithm>
#include <atomic>

namespace minimax::tree {

struct Payload
{
    Payload( Payload const& payload ) 
    : evaluation( payload.evaluation.load()) {}

    Payload() = default;
    std::atomic< double > evaluation { 0.0 };
};

template< typename MoveT, typename StateT >
class Player : public ::Player< MoveT >
{
public:
    using game_type = Game< MoveT, StateT >;
    using node_type = Node< MoveT, StateT, Payload >; 
    using pre_node_type = PreNode< MoveT, StateT, Payload >;
    using allocator_type = GenerationalArenaAllocator;

    Player( game_type const& game, unsigned max_depth, unsigned seed,
        allocator_type& allocator )
    : max_depth( max_depth ), g( seed ), allocator( allocator ),
      root( new (allocator.allocate< pre_node_type >()) 
                pre_node_type( game )) {}

    virtual double score( Game< MoveT, StateT > const&) const
    { return 0; };
private:
    unsigned max_depth;
    std::mt19937 g;
    allocator_type& allocator;
    std::vector< MoveT > move_stack;
    size_t eval_calls = 0;
    node_type* root;

    node_type* build_node( game_type const& game, MoveT const& move )
    {
        return new (allocator.allocate< node_type >()) 
            node_type( value_type( game, move ));
    }

    void apply_opponent_move( MoveT const& move ) override
    {
        auto itr =
            std::ranges::find_if(
                root->get_children(),
                [move](auto const& node)
                { return node.get_move() == move; } );
        if (itr == root->get_children().end())
            throw beat_it::Exception( "Invalid move.");

        node_type& new_root = *itr;

        allocator.reset();
        root = &new_root.copy_tree( allocator );
    }

    double eval( node_type& node, unsigned depth, double alpha, double beta )
    {
        ++eval_calls;
        pre_node_type& pre_node = static_cast< pre_node_type& >( node );

        if (GameResult result = node.get_game_result(); result == GameResult::Draw)
            return 0.0;
        else if (result == GameResult::Player1Win)
            return max_value( PlayerIndex::Player1 );
        else if (result == GameResult::Player2Win)
            return max_value( PlayerIndex::Player2 );
        else if (!depth)
            return score( pre_node.get_game());

        double best_score;
        std::function< bool (double, double) > compare;
        double* palpha;
        double const* pbeta;
        // minimizing player
        if (node.get_current_player_index() == PlayerIndex::Player1) 
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
                move_stack, node.get_current_player_index(), 
                pre_node.get_game().get_state());
            std::ranges::shuffle( move_stack, g );

            for (MoveT const& move : move_stack)
                node.get_children().push_front( 
                    *(new (allocator.allocate< pre_node_type >())
                        pre_node_type( 
                            pre_node.get_game().apply( move ), move )));
        }
        // evaluate child nodes recursively until pruning
        auto child_itr = node.get_children().begin();
        for (;child_itr != node.get_children().end(); ++child_itr) // NOSONAR
        {
            child_itr->get_payload().evaluation =
                 eval( *child_itr, depth - 1, alpha, beta );
            if (compare( child_itr->get_payload().evaluation, best_score ))
                best_score = child_itr->get_payload().evaluation;
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
                    a.get_payload().evaluation, b.get_payload().evaluation); 
            });

        node.get_children().splice(node.get_children().begin(), prefix);

        return best_score;
    }

    MoveT choose_move() override
    {
        if (root->get_game_result() != GameResult::Undecided)
            throw std::source_location::current();

        // eval with increasing depth
        // evaluation will benefit from better pruning from ordering of previous step
        // always start from level 0 because pruning my be different from last
        // time due to initialized alpha/beta start values
        for (size_t d = 0; d <= max_depth + 1; ++d)
            root->get_payload().evaluation = 
                eval( *root, d, -INFINITY, INFINITY );

        if (root->get_children().empty())
            throw beat_it::Exception( "no move choosen" );

        allocator.reset(); 
        root = &root->get_children().begin()->copy_tree( allocator );

        return root->get_move();
    }
};

} // namespace minimax::tree
