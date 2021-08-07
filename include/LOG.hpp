/**===============代替ROS—STREAM的模块==============
 * @author hqy
 * @date 2021.2.13
 * @note 输出彩色 + header + 时间显示
 */
#ifndef __Log_HPP
#define __Log_HPP

#include <chrono>
#include <iostream>
#include <sstream>
#undef Log

// #define LOG_DEBUG
#ifdef LOG_DEBUG
    #define log_debug Log::LogInfo
    #define log_printf Log::printc
#else
    #define log_debug(...)
    #define log_printf(...)
#endif

// No arguments
#define LOG_INFO_STREAM(format) log_debug("INFO", 0, format)                // Characters have no color
#define LOG_ERROR_STREAM(format) log_debug("ERROR", 31, format)             // Characters are printed in red
#define LOG_MARK_STREAM(format) log_debug("MARK", 33, format)               // Characters are printed in yellow
#define LOG_CHECK_STREAM(format) log_debug("CHECK", 32, format)             // Characters are printed in green
#define LOG_SHELL_STREAM(format) log_debug("SHELL", 34, format)             // Characters are printed in blue
#define LOG_GAY_STREAM(format) log_debug("GAY ", 35, format)                // Characters are printed in magenta

#define LOG_INFO(format, args...) log_debug("INFO", 0, format, args)        // Characters have no color
#define LOG_ERROR(format, args...) log_debug("ERROR", 31, format, args)     // Characters are printed in red
#define LOG_MARK(format, args...) log_debug("MARK", 33, format, args)       // Characters are printed in yellow
#define LOG_CHECK(format, args...) log_debug("CHECK", 32, format, args)     // Characters are printed in green
#define LOG_SHELL(format, args...) log_debug("SHELL", 34, format, args)     // Characters are printed in blue
#define LOG_GAY(format, args...) log_debug("GAY ", 35, format, args)        // Characters are printed in magenta

class Log{
public:
    Log(){}
    ~Log(){}
public:
    static void LogInfo(std::string header, int color, std::string format){
        uint64_t now = std::chrono::system_clock::now().time_since_epoch().count();
        std::stringstream ss;
        if (header.length() < 5){
            header = std::string(" ") + header;
        }
        ss << "[" << header << "] [" << std::to_string(now) << "] " << format;
        format = ss.str();
        ss.str("");
        if (color != 0){
            ss << "\033[" << std::to_string(color) << "m" << format << "\n\033[0m";
        }
        else {
            ss << format << std::endl;
        }
        std::cout << ss.str();
    }

    template<typename... types>
    static void printc(int color, std::string format, const types&... args){
        format = std::string("\033[") + std::to_string(color) + "m" + format + std::string("\n\033[0m");
        printf(format.c_str(), args...);
    }

    template<typename... types>
    static void LogInfo(std::string header, int color, std::string format, const types&... args){
        uint64_t now = std::chrono::system_clock::now().time_since_epoch().count();
        std::stringstream ss;
        if (header.length() < 5){
            header = std::string(" ") + header;
        }
        ss << "[" << header << "] [" << std::to_string(now) << "] " << format;
        format = ss.str();
        ss.str("");
        if (color != 0){
            ss << "\033[" << std::to_string(color) << "m" << format << "\n\033[0m";
        }
        else {
            ss << format << std::endl;
        }
        format = ss.str();
        printf(format.c_str(), args...);
    }
};

#endif //__Log_HPP