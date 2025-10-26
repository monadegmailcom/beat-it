#pragma once

#include "game.h"
#include "allocator.h"

#include <boost/intrusive/list.hpp>
#include <boost/intrusive/list_hook.hpp>

#include <shared_mutex>

template< typename MoveT, typename StateT, typename PayloadT >
class PreNode;
template< typename MoveT, typename StateT, typename PayloadT >
class FixNode;

template< typename MoveT, typename StateT, typename PayloadT >
struct NodeVisitor
{
    virtual ~NodeVisitor() = default;
    virtual void visit( PreNode< MoveT, StateT, PayloadT >& ) {};
    virtual void visit( FixNode< MoveT, StateT, PayloadT >& ) {};
};

template< typename MoveT, typename StateT, typename PayloadT >
class Node : public boost::intrusive::list_base_hook<>
{
public:
    Node( 
        MoveT const& move, GameResult game_result, 
        PlayerIndex current_player_index, PayloadT const& payload ) : 
        move( move ), game_result( game_result ), 
        current_player_index( current_player_index ), payload( payload ) {}

    virtual ~Node() = default;
    Node( Node const& ) = delete;
    Node& operator=( Node const& ) = delete;

    virtual void accept( NodeVisitor< MoveT, StateT, PayloadT >& ) = 0;

    MoveT const& get_move() const { return move; }
    GameResult get_game_result() const { return game_result; }
    PlayerIndex get_current_player_index() const 
    { return current_player_index; }
    PayloadT const& get_payload() const { return payload; }
    PayloadT& get_payload() { return payload; }

    boost::intrusive::list< Node >& get_children() 
    { return children; }
    boost::intrusive::list< Node > const& get_children() const
    { return children; }
    // not thread-safe.
    virtual Node& copy_tree( GenerationalArenaAllocator& ) = 0;
protected:
    // not thread-safe.
    void copy_children_to( Node& node, GenerationalArenaAllocator& allocator )
    {
        for (Node& child : children)
            node.get_children().push_back( child.copy_tree( allocator ));
    }
private:
    const MoveT move; // the previous move resulting in this game
    const GameResult game_result;
    const PlayerIndex current_player_index;

    PayloadT payload;
    boost::intrusive::list< Node > children;
};

// a fix node is already expanded and its children list can be accessed without
// locking.
template< typename MoveT, typename StateT, typename PayloadT >
class FixNode : public Node< MoveT, StateT, PayloadT >
{
private:
    using base_node_type = Node< MoveT, StateT, PayloadT >;
    using base_node_type::base_node_type;

    void accept( 
        NodeVisitor< MoveT, StateT, PayloadT >& visitor) override
    { visitor.visit( *this ); }

    // not thread-safe.
    base_node_type& copy_tree( 
        GenerationalArenaAllocator& allocator ) override
    {
        auto& new_node = *(new (allocator.allocate< FixNode >()) 
            FixNode( 
                this->get_move(), this->get_game_result(), 
                this->get_current_player_index(), this->get_payload()));
       
        this->copy_children_to( new_node, allocator );
        return new_node;
    }
};

// a pre-node is not yet expanded and needs an additional game attribute for
// later expansion. its children list to be accessed with a locked read-only
// mutex.
template< typename MoveT, typename StateT, typename PayloadT >
class PreNode : public Node< MoveT, StateT, PayloadT >
{
public:
    using base_node_type = Node< MoveT, StateT, PayloadT >;
    using fix_node_type = FixNode< MoveT, StateT, PayloadT >;
    using game_type = Game< MoveT, StateT >;
    using visitor_type = NodeVisitor< MoveT, StateT, PayloadT >;

    PreNode( 
        MoveT const& move, PayloadT const& payload, 
        Game< MoveT, StateT > const& game ) : 
            base_node_type( 
                move, game.result(), game.current_player_index(), payload ),
            game( game ) {}
    
    void accept( visitor_type& visitor) override
    { visitor.visit( *this ); }

    game_type const& get_game() const noexcept { return game; }
    std::shared_mutex& get_mutex() noexcept { return node_mutex; }
private:
    game_type game;
    std::shared_mutex node_mutex;
  
    // not thread-safe.
    base_node_type& copy_tree( GenerationalArenaAllocator& allocator ) override
    {
        // a pre node may be reduced to a fix node on copy.
        base_node_type& new_node = 
            (this->get_game_result() == GameResult::Undecided 
                && this->get_children().empty())
            ? *static_cast< base_node_type* >(
                new (allocator.allocate< PreNode >()) PreNode( 
                    this->get_move(), this->get_payload(), this->get_game()))
            : *static_cast< base_node_type* >(
                new (allocator.allocate< fix_node_type >()) fix_node_type( 
                    this->get_move(), this->get_game_result(), 
                    this->get_current_player_index(), 
                    this->get_payload()));
        
        this->copy_children_to( new_node, allocator );
        return new_node;
    }
};

template< typename MoveT, typename StateT, typename PayloadT>
size_t node_count( Node< MoveT, StateT, PayloadT > const& node )
{
    size_t count = 1;
    for (Node<  MoveT, StateT, PayloadT > const& child : node.get_children())
        count += node_count( child );
    return count;
}
