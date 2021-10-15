/// @author (Enigmatisms:https://github.com/Enigmatisms) @copyright Qianyue He
#pragma once
#include <unistd.h>
#include <fcntl.h>
#include <linux/input.h>
#include <array>
#include <atomic>

class KeyCtrl {
public:
    KeyCtrl(std::string dev_name, std::atomic_char& stat);
    ~KeyCtrl();

    void onKeyThread();
private:
    void handler(int sig) {
        printf("\nexiting...(%d)\n", sig);
        exit(0);
    }

    void perror_exit(char *error) {
        perror(error);
        handler(9);
    }
private:
    std::atomic_char& status;
    int fd;
};
