#pragma once

#include "player.h"

#include <boost/pool/pool_alloc.hpp>

namespace montecarlo {

template< typename MoveT, typename StateT >
class Node
{
public:
    Node( 
        Game< MoveT, StateT > const& game, 
        MoveT const& move,
        boost::fast_pool_allocator< 
            Node,
            boost::default_user_allocator_new_delete,
            boost::details::pool::null_mutex >& allocator ) 
    : game( game ), move( move ), allocator( allocator ) {}

    Node( Node const& ) = delete;
    Node& operator=( Node const& ) = delete;

    ~Node() 
    {
        if (first_child)
            allocator.deallocate( first_child );
        if (next_sibling)
            allocator.deallocate( next_sibling );
    }

    Node& push_front_sibling( Game< MoveT, StateT > const& game, MoveT const& move )
    {
        Node* const node = new (this->allocator.allocate()) Node( game, move, allocator );
        node->next_sibling = next_sibling;
        next_sibling = node;
        return *node;
    }

    Node& push_front_child( Game< MoveT, StateT > const& game, MoveT const& move )
    {
        Node* const node = new (this->allocator.allocate()) Node( game, move, allocator );
        node->next_sibling = first_child;
        first_child = node;
        return *node;
    }

    Node* remove_child( MoveT const& move )
    {
        Node* found_node = nullptr;
        Node* prev_node = nullptr;
        for (Node* node = first_child; node != nullptr; 
             prev_node = node, node = node->next_sibling)
            if (node->move == move)
            {   
                found_node = node;
                
                if (prev_node)
                    prev_node->next_sibling = node->next_sibling;
                else
                    first_child = node->next_sibling;

                found_node->next_sibling = nullptr;
                break;
            }

        return found_node;
    }

private:
    Node* next_sibling = nullptr;
    Node* first_child = nullptr;

    Game< MoveT, StateT > game;
    MoveT move;
    double numerator = 0.0;
    size_t denominator = 0;
    GameResult game_result = GameResult::Undecided;

    boost::fast_pool_allocator< 
        Node,
        boost::default_user_allocator_new_delete,
        boost::details::pool::null_mutex >& allocator;
};

template< typename MoveT, typename StateT >
using NodeAllocator = boost::fast_pool_allocator< 
    Node< MoveT, StateT >,
    boost::default_user_allocator_new_delete,
    boost::details::pool::null_mutex >;

template< typename MoveT, typename StateT >
class Player : public ::Player< MoveT >
{
public:
    Player( 
        Game< MoveT, StateT > const& game, 
        double exploration,
        size_t simulations,
        std::vector< MoveT >& move_stack,
        NodeAllocator< MoveT, StateT >& allocator )
    : move_stack( move_stack ), exploration( exploration ), simulations( simulations ) 
    {
        root = new (allocator.allocate()) Node< MoveT, StateT >( game, MoveT(), allocator );

        const size_t prev_size = move_stack.size();
        game.append_valid_moves( move_stack );
        for (MoveT const& child_move: std::ranges::subrange( move_stack + prev_size, move_stack.end()))
            root->push_front_child( game.apply( child_move ), child_move );

        move_stack.erase( move_stack.begin() + prev_size, move_stack.end());
    }

    ~Player() override
    {
        if (root)
            this->allocator.deallocate( root );
    }

    MoveT choose_move( Game< MoveT, StateT > const& game ) override
    {
        return MoveT{};
    }

    void apply_opponent_move( MoveT const& move ) override
    {
        Node< MoveT, StateT >* node = root->remove_child( move );
        if (!node)
            throw std::invalid_argument( "cannot apply invalid opponent move" );

        this->allocator.deallocate( root );
        root = node;
    }
private:
    std::vector< MoveT >& move_stack;
    Node< MoveT, StateT >* root = nullptr;
    double exploration;
    size_t simulations;
};

} // namespace montecarlo

