#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using namespace filesystem;

struct word_info {
	unsigned long count; // total number of times the word was found
	unsigned int in_books; // in how many books
	int last_book; // number of last book it was found in
};
// "the" last_book: -1 in_book = 1 count = 1 last_book = 1
// "the" book: 2 last_book = 1, count = 2 last_book = 2
class Dict {
	unordered_map<string, word_info> dict;

public:
	Dict() = default;
	void add_word(const string& word, const int book) {
		if (!dict.contains(word)) {
			dict[word] = {1, 1, book};
		} else {
			dict[word].count++;
			if (book > dict[word].last_book)
				dict[word].last_book = book;
		}
	}
};

// open a single book
Dict d;

void openfile(const path& path, const int book_num) {
	ifstream file(path);
	if (!file.is_open()) {
		cerr << "Failed to open file: " << path << endl;
		return;
	}
	// for each word in the file, lower case it and add it to the dictionary

	string word;
	while (file >> word) {
		// hyph-enated
		// Chrises'
		// 132nd
		ranges::transform(word, word.begin(), [](const unsigned char c) { return tolower(c); });

		d.add_word(word, book_num);
	}
}
int main(int, char* argv[]) {
	const string path = argv[1];
	try {
		auto book_num = 0;
		for (const auto& entry : directory_iterator(path)) {
			if (entry.is_regular_file() && entry.path().extension() == ".txt") {
				cout << "Found .txt file: " << entry.path().filename() << endl;
				openfile(entry.path(), ++book_num);
			}
		}
	} catch (const filesystem_error& err) {
		cerr << "Filesystem error: " << err.what() << endl;
	}
	return 0;
}
