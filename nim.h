#include "game.h"

namespace nim {

struct Move
{
    size_t heap_index;
    size_t count;
};

bool operator==( Move const& lhs, Move const& rhs );

using HeapRange = std::ranges::subrange< typename std::vector< size_t >::const_iterator >;

// the nim game, the player who takes the last object looses
class Game : public ::Game< Move >
{
public:
    // require: at least one heap and heaps are not empty
    // promise: heaps argument is moved
    Game( PlayerIndex, size_t heap_begin_index, std::vector< size_t >& heap_stack );
    ~Game() override;
    HeapRange get_heaps() const;
    void append_valid_moves( std::vector< Move >& move_stack ) const override;
    std::unique_ptr< ::Game< Move > > apply( Move const& ) const override;
    GameResult result() const override;
private:
    std::vector< size_t >& heap_stack;
    size_t heap_begin_index;
};

namespace console {

class HumanPlayer : public Player< Move >
{
public:
    Move choose( ::Game< Move > const& game ) override;
};

} // namespace console {

} // namespace nim {
