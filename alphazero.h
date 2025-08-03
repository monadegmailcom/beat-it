#pragma once

#include "player.h"
#include "game.h"
#include "node.h"

#include <random>

namespace alphazero
{

namespace detail
{
    template< typename MoveT, typename StateT >
    struct Value
    {
        Value( Game< MoveT, StateT > const& game, MoveT const& move )
        : game( game ), move( move ), game_result( game.result()) {}

        Game< MoveT, StateT > game;
        MoveT move; // the previous move resulting in this game
        const GameResult game_result; // the cached game result

        size_t visits = 0;
        // policy probalities (priors) of choosing this move
        float nn_policy = 0.0;
        // 1 for win, 0.5 for draw, 0 for loss
        // value sum of the game outcomes: -1 loss, 0 draw, 1 win from the perspective
        // of the current player
        float nn_value_sum = 0.0;
    };

    float game_result_2_score( GameResult, PlayerIndex );

} // namespace detail {

template< typename MoveT, typename StateT >
using NodeAllocator = ::NodeAllocator< detail::Value< MoveT, StateT > >;

template< typename MoveT, typename StateT, size_t G, size_t P >
class Player : public ::Player< MoveT >
{
public:
    using game_type = Game< MoveT, StateT >;
    using value_type = detail::Value< MoveT, StateT >;
    using node_type = Node< value_type >;
    static constexpr std::size_t game_size = G;
    static constexpr std::size_t policy_size = P;

    Player(
        Game< MoveT, StateT > const& game,
        float c_base,
        float c_init,
        size_t simulations,
        NodeAllocator< MoveT, StateT >& allocator )
    : root( new (allocator.allocate(1))
            node_type( value_type( game, MoveT()), allocator ),
            // The custom deleter now only needs to destruct and deallocate the single
            // node pointer it is given. The Node's destructor is responsible for
            // recursively cleaning up its children.
            [&allocator](node_type* ptr) { if (ptr) { ptr->~node_type(); allocator.deallocate(ptr, 1); } }
          ),
      c_base( c_base ), c_init( c_init ), simulations( simulations ), allocator( allocator )
    {
        if (!simulations)
            throw std::invalid_argument( "simulations must be > 0" );
    }

    MoveT choose_move() override
    {
        for (size_t i = simulations; i != 0; --i)
            simulation( *root );
        auto itr = choose_best_move();

        if (itr == root->get_children().end())
            throw std::runtime_error( "no move choosen" );

        auto new_root = &*itr;
        root->get_children().erase( itr );
        root.reset( new_root );

        return root->get_value().move;
    }

    NodePtr< value_type >& get_root() { return root; }
    size_t get_simulations() const { return simulations; }

    void apply_opponent_move( MoveT const& move ) override
    {
        auto itr = std::ranges::find_if(
            root->get_children(),
            [move](auto const& node) { return node.get_value().move == move; } );

        node_type* new_root = nullptr;
        if (itr == root->get_children().end())
            new_root = new
                (allocator.allocate(1))
                node_type( value_type( root->get_value().game.apply( move ), move), allocator );
        else
        {
            new_root = &*itr;
            root->get_children().erase( itr );
        }

        root.reset( new_root );
    }

    // promise: return index of move in policy_vector
    virtual size_t move_to_policy_index( MoveT const& ) const = 0;

    // promise: node is expanded
    float nn_eval( node_type& node )
    {
        auto& value = node.get_value();

        // if node not expanded, do so
        if (node.get_children().empty())
            for (MoveT const& move : value.game)
            {
                auto child = new
                    (allocator.allocate(1))
                    Node( value_type( value.game.apply( move ), move ), allocator );

                node.get_children().push_front( *child );
            }

        auto [nn_value, policies] = predict( serialize_state( value.game ));

        // convert logits of legal moves to probabilities with softmax
        // and save in children
        float policy_sum = 0.0f;
        for (auto& child : node.get_children())
        {
            const float p = std::exp( policies[
                move_to_policy_index( child.get_value().move )]);
            child.get_value().nn_policy = p;
            policy_sum += p;
        }
        // normalize
        for (auto& child : node.get_children())
            child.get_value().nn_policy /= policy_sum;

        return nn_value;
    }

    float simulation( node_type& node )
    {
        auto& value = node.get_value();
        ++value.visits;

        // set target value
        float nn_value;
        // from game result if game is decided
        if (value.game_result != GameResult::Undecided)
            nn_value = detail::game_result_2_score(
                value.game_result, value.game.current_player_index());
        // from nn if node is not yet expanded
        else if (node.get_children().empty())
            nn_value = nn_eval( node );
        // or recursivly otherwise
        else
            // recursively simulate the selected child node
            // negate sign of return value because its from the opponent's perspective
            nn_value = -simulation( select( node ));

        value.nn_value_sum += nn_value;

        return nn_value;
    }

    // Apple clang version 17.0.0 (clang-1700.0.13.3) produces strange errors
    // when i specify the return value iterator type explicitly:
    // Node< ValueT > member boost::intrusive::list< Node > children has incomplete type
    auto choose_best_move()
    {
        // choose child with most visits
        return std::ranges::max_element( root->get_children(),
            [](auto const& a, auto const& b)
            { return a.get_value().visits < b.get_value().visits; } );
    }

    virtual std::array< float, G > serialize_state( Game< MoveT, StateT > const& ) const = 0;
protected:
    // upper confidence bound
    float ucb( value_type const& value, size_t parent_visits )
    {
        const float c =
            std::log( (parent_visits + c_base + 1) / c_base) + c_init;
        float q = 0.0;
        if (value.visits != 0)
            // switch sign because value is from the child's perspective
            q = -value.nn_value_sum / value.visits;
        const float p = value.nn_policy;

        return q + c * p * std::sqrt( parent_visits ) / (value.visits + 1);
    }

    // require: node has children
    node_type& select( node_type& node )
    {
        return *std::ranges::max_element(
            node.get_children(),
            [this, parent_visits = node.get_value().visits]
            (auto const& a, auto const& b)
            {
                return   ucb( a.get_value(), parent_visits )
                       < ucb( b.get_value(), parent_visits );
            });
    }

    // predict game state value and policy vector from nn
    // promise: returned policies contain probability distribution of moves
    virtual std::pair< float, std::array< float, P > > predict( std::array< float, G > const& ) = 0;

    NodePtr< value_type > root;
    float c_base;
    float c_init;
    const size_t simulations;

    NodeAllocator< MoveT, StateT >& allocator;
};

namespace training {

template< size_t G, size_t P >
struct Position
{
    std::array< float, G > game_state_players;
    std::array< float, P > target_policy;
    float target_value = 0.0f;
    PlayerIndex current_player;
};

template< typename MoveT, typename StateT, size_t G, size_t P >
class SelfPlay
{
public:
    SelfPlay(
        Player< MoveT, StateT, G, P >& player,
        float dirichlet_alpha,
        float dirichlet_epsilon,
        size_t opening_moves,
        std::mt19937& g,
        std::vector< Position< G, P >>& positions )
    : player( player ), opening_moves( opening_moves ), dirichlet_epsilon( dirichlet_epsilon ),
      g( g ), gamma_dist( dirichlet_alpha, 1.0f ), positions( positions ) {}

    void run()
    {
        const size_t prev_size = positions.size();
        GameResult game_result;

        for (game_result = player.get_root()->get_value().game_result;
             game_result == GameResult::Undecided;
             game_result = player.get_root()->get_value().game_result)
            choose_move();

        for (auto itr = positions.begin() + prev_size; itr != positions.end(); ++itr)
            itr->target_value = detail::game_result_2_score( game_result, itr->current_player );
    }
protected:
    // promise: root node is expanded
    void add_dirichlet_noise()
    {
        auto& root = *player.get_root();

        // expand node if not expanded
        if (root.get_children().empty())
            player.nn_eval( root );

        // add noise for root node
        for (auto& child : root.get_children())
        {
            float& policy = child.get_value().nn_policy;
            policy *= 1.0 - dirichlet_epsilon;
            policy += gamma_dist( g ) * dirichlet_epsilon;
        }
    }

    MoveT choose_move()
    {
        // add dirichlet noise to root node before first simulation
        add_dirichlet_noise();

        auto& root = *player.get_root();

        // root node is expanded now
        for (size_t i = player.get_simulations(); i != 0; --i)
            player.simulation( root );

        auto itr = root.get_children().begin();
        if (++move_count <= opening_moves)
            itr = choose_opening_move();
        else
            itr = player.choose_best_move();

        if (itr == root.get_children().end())
            throw std::runtime_error( "no move choosen" );

        append_training_data();

        auto new_root = &*itr;
        root.get_children().erase( itr );
        player.get_root().reset( new_root );

        return new_root->get_value().move;
    }

    void append_training_data()
    {
        positions.emplace_back();
        auto& position = positions.back();
        auto& root = *player.get_root();
        auto& value = root.get_value();

        position.current_player = value.game.current_player_index();

        // append serialized game state
        position.game_state_players = player.serialize_state( value.game );

        size_t sum_visits = 0;
        for (auto const& child : root.get_children())
            sum_visits += child.get_value().visits;

        // append new target policies
        for (size_t policy_index = 0; policy_index < P; ++policy_index)
        {
            auto child_itr = std::ranges::find_if(
                root.get_children(),
                [this, policy_index](auto const& child)
                { return policy_index == player.move_to_policy_index( child.get_value().move ); } );
            if (child_itr == root.get_children().end())
                position.target_policy[policy_index] = 0.0f;
            else
                position.target_policy[policy_index] = static_cast< float >(
                    child_itr->get_value().visits ) / sum_visits;
        }
    }

    auto choose_opening_move()
    {
        auto& children = player.get_root()->get_children();

        // sample from children in opening phase by visit distribution
        // so we are more versatile in the opening
        size_t total_visits = 0;
        for (auto const& child : children)
            total_visits += child.get_value().visits;

        if (!total_visits)
            throw std::runtime_error( "no opening move choosen");

        std::uniform_int_distribution< size_t > dist(0, total_visits - 1);
        size_t r = dist( g );

        for (auto itr = children.begin(); itr != children.end(); ++itr)
        {
            if (r < itr->get_value().visits)
                return itr;
            r -= itr->get_value().visits;
        }

        return children.begin(); // Fallback, should ideally not be reached
    }

    Player< MoveT, StateT, G, P >& player;
    size_t opening_moves;
    float dirichlet_epsilon;
    std::mt19937& g;
    std::gamma_distribution< float > gamma_dist;
    std::vector< Position< G, P >>& positions;
    size_t move_count = 0;
};

} // namespace training
} // namespace alphazero