#pragma once

#include "minimax.h"
#include "node.h"

#include <random>
#include <algorithm>

namespace minimax {
namespace tree {
namespace detail {

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

} // namespace detail {

template< typename MoveT, typename StateT >
using NodeAllocator = ::NodeAllocator< detail::Value< MoveT, StateT > >;

template< typename MoveT, typename StateT >
class Player : public ::Player< MoveT >
{
public:
    Player( Game< MoveT, StateT > const& game, unsigned depth, unsigned seed,
        NodeAllocator< MoveT, StateT >& allocator )
    : depth( depth ), g( seed ), allocator( allocator ),
      root( new (allocator.allocate(1))
            Node< detail::Value< MoveT, StateT >>(
                detail::Value< MoveT, StateT >( game, MoveT()), allocator ),
            [&allocator = allocator](auto* ptr) {
                if (ptr) { ptr->~Node(); allocator.deallocate(ptr, 1); }
            }
          ) {}

    virtual double score( Game< MoveT, StateT > const&) const
    { return 0; };
protected:
    unsigned depth;
    std::mt19937 g;
    NodeAllocator< MoveT, StateT >& allocator;
    std::vector< MoveT > move_stack;
    size_t eval_calls = 0;
    double best_score = 0.0;

    NodePtr< detail::Value< MoveT, StateT > > root;

    void apply_opponent_move( MoveT const& move ) override
    {
        auto itr =
            std::ranges::find_if(
                root->get_children(),
                [move](auto const& node)
                { return node.get_value().move == move; } );
        Node< detail::Value< MoveT, StateT > >* new_root = nullptr;

        if (itr == root->get_children().end())
            new_root = new (this->allocator.allocate(1))
                   Node< detail::Value< MoveT, StateT >>(
                        detail::Value< MoveT, StateT >(
                            root->get_value().game.apply( move ), move),
                        this->allocator );
        else
        {
            new_root = &*itr;
            root->get_children().erase( itr );
        }

        root.reset( new_root );
    }

    double eval(
        Node< detail::Value< MoveT, StateT > >& node,
        unsigned depth, double alpha, double beta )
    {
        ++eval_calls;
        auto& value = node.get_value();

        const GameResult result = value.game_result;
        if (result == GameResult::Draw)
            return 0.0;
        else if (result == GameResult::Player1Win)
            return max_value( Player1 );
        else if (result == GameResult::Player2Win)
            return max_value( Player2 );
        else if (!depth)
            return score( value.game );

        double best_score;
        std::function< bool (double, double) > compare;
        double* palpha;
        double* pbeta;
        if (value.game.current_player_index() == Player1) // minimizing player
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
                move_stack, value.game.current_player_index(), value.game.get_state());
            std::shuffle( move_stack.begin(), move_stack.end(), g );

            for (MoveT const& move : move_stack)
                node.get_children().push_front( *(new
                    (allocator.allocate(1))
                    Node(
                        detail::Value< MoveT, StateT >( value.game.apply( move ), move ),
                        allocator )));
        }
        // evaluate child nodes recursivly until pruning
        auto child_itr = node.get_children().begin();
        for (;child_itr != node.get_children().end(); ++child_itr)
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
        boost::intrusive::list<Node<detail::Value<MoveT, StateT>>> prefix;
        prefix.splice(
            prefix.begin(),
            node.get_children(),
            node.get_children().begin(),
            child_itr           // ...up to (but not including) child_itr
        );

        prefix.sort(
            [compare](auto const& a, auto const& b)
            { return compare( a.get_value().evaluation, b.get_value().evaluation); });

        node.get_children().splice(node.get_children().begin(), prefix);

        return best_score;
    }

    MoveT choose_move() override
    {
        if (root->get_value().game.result() != GameResult::Undecided)
            throw std::runtime_error( "game already finished" );

        // eval with increasing depth
        // evaluation will benefit from better pruning from ordering of previous step
        // always start from level 0 because pruning my be different from last
        // time due to initialized alpha/beta start values
        for (size_t d = 0; d <= depth + 1; ++d)
            root->get_value().evaluation = eval( *root, d, -INFINITY, INFINITY );

        if (root->get_children().empty())
            throw std::runtime_error( "no move choosen");
        auto chosen = root->get_children().begin();

        auto new_root = &*chosen;
        root->get_children().erase( chosen );
        root.reset( new_root );

        return root->get_value().move;
    }
};

} // namespace tree {
} // namespace minimax {