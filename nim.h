#include "game.h"

namespace nim {

struct Move
{
    size_t heap_index;
    size_t count;
};

bool operator==( Move const& lhs, Move const& rhs );

// the nim game, the player who takes the last object looses
class Game : public UndecidedGame< Move >
{
public:
    // require: at least one heap and heaps are not empty
    // promise: heaps argument is moved
    Game( PlayerIndex player_index, std::vector< size_t >&& heaps, std::vector< Move >& move_stack );
    ~Game() override;
    std::vector< size_t > const& get_heaps() const { return heaps; }
    std::ranges::subrange< typename std::vector< Move >::const_iterator > valid_moves() const override
    {
        const auto begin = move_stack.cbegin() + moves_begin_index;
        return std::ranges::subrange( begin, begin + moves_count );
    }
    std::unique_ptr< ::Game > apply( size_t ) const override;
private:
    std::vector< size_t > heaps;

    std::vector< Move >& move_stack;
    size_t moves_begin_index;
    size_t moves_count = 0;
};

namespace console {

class HumanPlayer : public Player< Move >
{
public:
    size_t choose( UndecidedGame< Move > const& game ) override;
};

} // namespace console {

} // namespace nim {
