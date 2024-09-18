#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <algorithm>
using namespace std;

struct word_info {
    uint64_t count; // total number of times the word was found
    uint32_t in_books; // in how many books
    int32_t last_book; // number of last book it was found in
};
// "the"  last_book: -1 in_book = 1  count = 1 last_book = 1
// "the"  book: 2  last_book = 1, count = 2 last_book = 2
class Dict {
private:
    unordered_map<string, word_info> dict;
public:
    Dict() {}
    void add_word(const string& word, int book) {
        if (dict.find(word) == dict.end()) {
            dict[word] = { 1, 1, book };
        }
        else {
            dict[word].count++;
            if (book > dict[word].last_book)
                dict[word].last_book = book;        }
    }
};

#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// open a single book
Dict d;

void openfile(const fs::path& path, int book_num) {

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return;
    }
    // for each word in the file, lower case it and add it to the dictionary

    string word;
    while (file >> word) {
        // hyph-enated
        // Chrises'
        //  132nd 
        transform(word.begin(), word.end(), word.begin(), [](unsigned char c) { return std::tolower(c); });

        d.add_word(word, book_num);
    }
}
int main(int argc, char* argv[]) {
    string path = argv[1];

    int book_num = 0;
    try {
        for (const auto& entry : fs::directory_iterator(path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                std::cout << "Found .txt file: " << entry.path().filename() << std::endl;
                openfile(entry.path(), ++book_num);
            }
        }
    } catch (const fs::filesystem_error& err) {
        std::cerr << "Filesystem error: " << err.what() << std::endl;
    }

    return 0;
}
