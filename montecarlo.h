#pragma once

#include "player.h"

#include <boost/pool/pool_alloc.hpp>
#include <cmath>

namespace montecarlo {

namespace detail {

template< typename MoveT, typename StateT >
class Node;

} // namespace detail {

template< typename MoveT, typename StateT >
using NodeAllocator = boost::fast_pool_allocator< 
    detail::Node< MoveT, StateT >,
    boost::default_user_allocator_new_delete,
    boost::details::pool::null_mutex >;

template< typename MoveT, typename StateT >
struct Data
{
    Data( std::mt19937& g, NodeAllocator< MoveT, StateT >& allocator ) 
    : g( g ), allocator( allocator ) {}

    std::mt19937& g;
    NodeAllocator< MoveT, StateT >& allocator;
    std::vector< MoveT > move_stack;
    size_t playout_count = 0;
};

namespace detail {
    
template< typename MoveT, typename StateT >
GameResult playout( Game< MoveT, StateT > game, Data< MoveT, StateT >& data )
{
    GameResult result;
    for (result = GameResult::Undecided; result == GameResult::Undecided;)
    {
        const size_t prev_size = data.move_stack.size();
        game.append_valid_moves( data.move_stack );
        const size_t move_count = data.move_stack.size() - prev_size;
        if (move_count == 0)
            throw std::runtime_error( "no valid moves to playout" );

        game = game.apply( 
            data.move_stack[prev_size + data.g() % move_count] );
        result = game.result();
        data.move_stack.resize( prev_size );
    }
    return result;
}

template< typename MoveT, typename StateT >
class Node
{
public:
    Node( Game< MoveT, StateT > const& game, NodeAllocator< MoveT, StateT >& allocator ) 
    : game( game ), game_result( game.result()), allocator( allocator ) 
    {}

    Node( Node const& ) = delete;
    Node& operator=( Node const& ) = delete;

    ~Node() 
    {
        if (first_child)
        {
            first_child->~Node();
            allocator.deallocate( first_child );
        }
        if (next_sibling)
        {
            next_sibling->~Node();
            allocator.deallocate( next_sibling );
        }
    }

    GameResult get_game_result() const { return game_result; }
    MoveT const& get_move() const { return move; }

    // debug interface ->
    size_t children_count() const 
    {
        size_t count = 0;
        for (Node* node = first_child; node != nullptr; node = node->next_sibling)
            ++count;
        return count;
    }
    size_t node_count() const
    {
        size_t count = 1;
        for (Node* node = first_child; node != nullptr; node = node->next_sibling)
            count += node->node_count();
        return count;
    }
    // <- debug interface

    size_t get_visits() const { return visits; }

    void add_children( std::vector< MoveT >& move_stack )
    {
        if (first_child)
            throw std::runtime_error( "children already added" );
        const size_t prev_size = move_stack.size();
        game.append_valid_moves( move_stack );
        for (MoveT const& child_move : std::ranges::subrange( move_stack.begin() + prev_size, move_stack.end()))
            this->push_front_child( game.apply( child_move ), child_move );

        move_stack.resize( prev_size );
    }

    Node* remove_child_by_move( MoveT const& move )
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

    Node* remove_child_by_most_visits()
    {
        Node* best_child = first_child;
        Node* prev_child = nullptr;
        Node* child2 = nullptr;
        size_t most_visits = 0;

        for (Node* child = first_child; child; 
             child2 = child, child = child->next_sibling)
        {
            if (child->visits > most_visits)
            {
                most_visits = child->visits;
                best_child = child;
                prev_child = child2;
            }
        }
        if (best_child)
        {    
            if (prev_child)
                prev_child->next_sibling = best_child->next_sibling;
            else
                first_child = best_child->next_sibling;

            best_child->next_sibling = nullptr;
        }

        return best_child;
    }

    void update_points( GameResult game_result )
    {
        const GameResult player_to_game_result[] = 
            { GameResult::Player1Win, GameResult::Player2Win };
        if (game_result == GameResult::Draw)
            points += 0.5;
        else if (game_result == player_to_game_result[game.current_player_index()])
            points += 1.0;
    }

    GameResult simulation( Data< MoveT, StateT >& data, double exploration)
    {
        ++visits;

        GameResult backpropagation;
        if (game_result != GameResult::Undecided)
            backpropagation = game_result;
        else if (visits == 1)
        {
            backpropagation = playout< MoveT, StateT >( game, data );
            ++data.playout_count;
        }
        else 
        {
            if (visits == 2)
                add_children( data.move_stack );

            Node* const node = selection( exploration );
            if (!node)
                throw std::runtime_error( "no node selected" );

            backpropagation = node->simulation( data, exploration );
        }

        update_points( backpropagation );
        return backpropagation;
    }

    Node* selection( double exploration )
    {
        Node* selected_node = nullptr;
        double best_uct = -INFINITY;
        for (Node* node = first_child; node != nullptr; 
             node = node->next_sibling)
        {
            if (!node->visits)
                return node;

            const double uct = 
                1 - node->points / node->visits 
                + exploration * std::sqrt( std::log( visits ) / node->visits );
            if (uct > best_uct)
            {
                best_uct = uct;
                selected_node = node;
            }
        }

        return selected_node;
    }
private:
    Node* next_sibling = nullptr;
    Node* first_child = nullptr;

    Game< MoveT, StateT > game;
    MoveT move;
    double points = 0.0; // 1 for win, 0.5 for draw, 0 for loss
    size_t visits = 0;
    const GameResult game_result;

    NodeAllocator< MoveT, StateT >& allocator;

    void push_front_child( Game< MoveT, StateT > const& game, MoveT const& move )
    {
        Node* const node = new (this->allocator.allocate()) Node( game, allocator );
        node->move = move;
        node->next_sibling = first_child;
        first_child = node;
    }
};

template< typename MoveT, typename StateT >
using NodePtr = std::unique_ptr< Node< MoveT, StateT >, 
                                 std::function< void (Node< MoveT, StateT >*) > 
                               >;

template< typename MoveT, typename StateT >                               
void deallocate( NodeAllocator< MoveT, StateT >& allocator, Node< MoveT, StateT >* ptr )
{
    if (ptr)
    {
        ptr->~Node< MoveT, StateT >(); // Call the destructor
        allocator.deallocate( ptr ); // Deallocate memory
    }
}

} // namespace detail

template< typename MoveT, typename StateT >
class Player : public ::Player< MoveT >
{
public:
    Player( 
        Game< MoveT, StateT > const& game, 
        double exploration,
        size_t simulations,
        Data< MoveT, StateT >& data )
    : data( data ), 
      root( new (data.allocator.allocate()) 
            detail::Node< MoveT, StateT >( game, data.allocator ),
            [&data](detail::Node< MoveT, StateT >* ptr) { deallocate( data.allocator, ptr ); }
          ),
      exploration( exploration ), simulations( simulations ) 
    {}

    MoveT choose_move() override
    {
        for (size_t i = simulations; i != 0; --i)
            root->simulation( data, exploration );

        detail::Node< MoveT, StateT >* const node = root->remove_child_by_most_visits();
        if (!node)
            throw std::runtime_error( "no move choosen" );

        root.reset( node );

        return root->get_move();
    }

    void apply_opponent_move( MoveT const& move ) override
    {
        if (root->get_visits() <= 1)
            root->add_children( data.move_stack );

        detail::Node< MoveT, StateT >* node = root->remove_child_by_move( move );
        if (!node)
            throw std::runtime_error( "no move found" );

        root.reset( node );
    }

    // debug interface ->
    detail::Node< MoveT, StateT > const& root_node() const { return *root; }
    // <- debug interface
private:
    Data< MoveT, StateT >& data;
    detail::NodePtr< MoveT, StateT > root;
    double exploration;
    size_t simulations;
};

template< typename MoveT, typename StateT >
using Buffer = char[sizeof( Player< MoveT, StateT > )];

template< typename MoveT, typename StateT >
PlayerFactory< MoveT > player_factory(
    Game< MoveT, StateT > const& game, 
    double exploration,
    size_t simulations,
    Data< MoveT, StateT >& data,
    Buffer< MoveT, StateT > raw_data )
{
    return [&game, &data, exploration, simulations, raw_data]()
    { 
        return std::unique_ptr< ::Player< MoveT >, void(*)( ::Player< MoveT >*) >( 
            new (raw_data) Player< MoveT, StateT >( game, exploration, simulations, data ), 
            [](::Player< MoveT >* p){p->~Player(); }); 
    };
}

} // namespace montecarlo

