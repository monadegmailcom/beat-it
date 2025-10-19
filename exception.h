#pragma once

#include <stdexcept>
#include <source_location>
#include <string>
#include <format>

namespace beat_it {

class Exception : public std::runtime_error {
public:
    explicit Exception(
        std::string const& message,
        std::source_location const& location = std::source_location::current()
    ) :
        std::runtime_error( std::format( 
            "{} in {} at {}: {}", message, location.function_name(), 
            location.file_name(), std::to_string(location.line())))
    {}
};

} // namespace beat_it
