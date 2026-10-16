#pragma once

#include <format>
#include <iostream>
#include <string>
#include <utility>


template <typename... Args>
void println(std::format_string<Args...> fmt, Args&&... args)
{
    std::string formatted = std::format(fmt, std::forward<Args>(args)...);
    std::cout << formatted << std::endl;
}