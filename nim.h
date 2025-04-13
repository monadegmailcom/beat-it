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
    using HeapRange = std::ranges::subrange< typename std::vector< size_t >::const_iterator >;
    // require: at least one heap and heaps are not empty
    // promise: heaps argument is moved
    Game( PlayerIndex, size_t heap_count, std::vector< size_t >& heap_stack, std::vector< Move >& move_stack );
    ~Game() override;
    HeapRange get_heaps() const;
    MoveRange valid_moves() const override;
    std::unique_ptr< ::Game > apply( size_t ) const override;
private:
    std::vector< size_t >& heap_stack;
    size_t heap_begin_index;
    size_t heap_count = 0;
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
