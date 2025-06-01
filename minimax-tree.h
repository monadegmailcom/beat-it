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
struct Data
{
    Data( std::mt19937& g, NodeAllocator< MoveT, StateT >& allocator ) 
    : g( g ), allocator( allocator ) {}

    std::mt19937& g;
    NodeAllocator< MoveT, StateT >& allocator;
    std::vector< Node< detail::Value< MoveT, StateT > >* > stack;
    std::vector< MoveT > move_stack;
    size_t eval_calls = 0;
    double best_score = 0.0;
};

template< typename MoveT, typename StateT >
class Player : public ::Player< MoveT >
{   
public:
    Player( 
        Game< MoveT, StateT > const& game, unsigned depth, Data< MoveT, StateT >& data ) 
    : depth( depth ), data( data ),
      root( new (data.allocator.allocate()) 
            Node< detail::Value< MoveT, StateT >>( 
                detail::Value< MoveT, StateT >( game, MoveT()), data.allocator ),
            [&data](auto ptr) { deallocate( data.allocator, ptr ); }
          ) {}

    virtual double score( Game< MoveT, StateT > const&) const 
    { return 0; };
protected:
    unsigned depth;
    Data< MoveT, StateT >& data;
    NodePtr< detail::Value< MoveT, StateT > > root;

    void apply_opponent_move( MoveT const& move ) override
    {
        auto itr = 
            std::find_if( root->begin(), root->end(), 
                [move](auto const& node)
                { return node.get_value().move == move; } );
        auto node = itr == root->end() 
            ? new (this->data.allocator.allocate()) 
                   Node< detail::Value< MoveT, StateT >>( 
                        detail::Value< MoveT, StateT >(
                            root->get_value().game.apply( move ), move), 
                        this->data.allocator ) 
            : &*itr;

        root->remove_child( itr );
        root.reset( node );
    }

    double eval( 
        Node< detail::Value< MoveT, StateT > >& node, 
        unsigned depth, double alpha, double beta )
    {
        ++data.eval_calls;
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
        if (node.begin() == node.end())
        {
            data.move_stack.clear();
            GameState< MoveT, StateT >::get_valid_moves( 
                data.move_stack, value.game.current_player_index(), value.game.get_state());
            std::shuffle( data.move_stack.begin(), data.move_stack.end(), data.g );

            for (MoveT const& move : data.move_stack)
                node.push_front_child( detail::Value< MoveT, StateT >( 
                    value.game.apply( move ), move));
        }
        // evaluate child nodes recursivly until pruning
        auto child_itr = node.begin();
        for (;child_itr != node.end(); ++child_itr)
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

        // sort relevant child nodes for earlier pruning opportunities on next visit
        node.sort_prefix( 
            child_itr, data.stack,
            [compare](auto const& a, auto const& b) 
            { return compare( a.evaluation, b.evaluation); });

        return best_score;
    }

    MoveT choose_move() override
    {
        if (root->get_value().game.result() != GameResult::Undecided)
            throw std::runtime_error( "game already finished" );

        // eval with increasing depth
        // evaluation will benefit from better pruning from ordering of previous step
        // always start from level 0 because pruning my be different from last 
        //   time due to initialized alpha/beta start values
        for (size_t d = 0; d <= depth + 1; ++d)
            root->get_value().evaluation = eval( *root, d, -INFINITY, INFINITY );
    
        auto chosen = root->begin();
        if (chosen == root->end())
            throw std::runtime_error( "no move choosen");

        root->remove_child( chosen );
        root.reset( &*chosen );

        return root->get_value().move;
    }
};

} // namespace tree {
} // namespace minimax {