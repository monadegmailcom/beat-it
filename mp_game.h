#pragma once

#include <boost/iterator/iterator_facade.hpp>

#include <vector>
#include <optional>
#include <variant>
#include <iostream>

namespace multiplayer {

// -> possible game results
struct Draw {};
struct Undecided {};
template< typename PlayerIndexT >
struct Winner { PlayerIndexT player_index; };
// <-

template< typename PlayerIndexT >
using GameResult = std::variant< Draw, Undecided, Winner< PlayerIndexT >>;

template< typename PlayerIndexT >
std::ostream& operator<<(
    std::ostream& stream, GameResult< PlayerIndexT > game_result)
{
    auto visitor = [&stream]( auto&& arg )
    {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v< T, Draw >)
            stream << "Draw";
        else if constexpr (std::is_same_v< T, Undecided >)
            stream << "Undecided";
        else
        {
            auto [ v ] = arg;
            stream << "Winner player " << v;
        }
    };

    visit( visitor, game_result);

    return stream;
}

// helper base class to provide a default get_valid_moves implementation
template< typename DerivedRuleSetT, typename MoveT, typename StateT >
struct DefaultGetValidMovesProvider
{
    static void get_valid_moves(
        std::vector< MoveT >& moves, StateT const& state )
    {
        moves.clear();
        std::optional< MoveT > move;
        for (DerivedRuleSetT::next_valid_move( move, state );
             move; next_valid_move( move, state ))
            moves.push_back( *move );
    }
};

// for every game provide this rule set
// using crtp (Curiously recurring template pattern) to provide a default
//  get_valid_moves implementation
template< typename MoveT, typename PlayerIndexT, typename StateT >
struct RuleSet : public DefaultGetValidMovesProvider< RuleSet<
    MoveT, PlayerIndexT, StateT >, MoveT, StateT >
{
    using move_type = MoveT;
    using state_type = StateT;
    using game_result_type = GameResult< PlayerIndexT >;

    // each game is required to implement these
    static PlayerIndexT next_player_index( StateT const& );
    static game_result_type result( StateT const& );
    // require: move is valid
    static StateT apply( MoveT const&, StateT const& );
    // promise: return next valid move if available
    static void next_valid_move( std::optional< MoveT >&, StateT const& );

    // get_valid_moves is inherited from DefaultGetValidMovesProvider
    // specializations can override their own get_valid_moves
};

template< typename MoveT, typename PlayerIndexT, typename StateT >
class Game
{
public:
    using move_type = MoveT;
    using state_type = StateT;
    using rule_set_type = RuleSet< MoveT, PlayerIndexT, StateT >;
    using game_result_type = GameResult< PlayerIndexT >;

    Game( StateT const& state ) : state_( state) {}

    Game apply( MoveT const& move )
    { return Game( rule_set_type::apply( move, state_ )); }

    game_result_type result() const
    { return rule_set_type::result( state_ );}

    StateT const& state() const { return state_; }

    PlayerIndexT next_player_index() const
    { return rule_set_type::next_player_index( state_ ); }

    class MoveItr : public boost::iterator_facade<
        MoveItr, // Derived class
        MoveT, // Value type
        boost::single_pass_traversal_tag, // Iterator category
        MoveT const& > // Reference type
    {
    public:
        explicit MoveItr( Game const& game ) : game( game ) {}

        // require: move is valid
        const MoveT& dereference() const { return *move; }

        // pre-increment operator
        void increment()
        {
            move = rule_set_type::next_valid_move( game.state());
        }

        // equality comparison
        bool equal(const MoveItr& other) const { return move == other.move; }
    private:
        friend class boost::iterator_core_access;
        // unset optional represents end of iteration
        std::optional< MoveT > move;
        Game const& game;
    };

    MoveItr begin() const
    {
        MoveItr itr( *this );
        return ++itr;
    }

    MoveItr end() const
    {
        return MoveItr( *this);
    }
private:
    StateT state_;
};

} // namespace multiplayer
