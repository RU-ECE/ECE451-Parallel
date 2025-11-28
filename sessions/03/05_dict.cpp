#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;

int main() {
	unordered_map<string, int> dict1, dict2;

	dict1["hello"] = 1;
	dict1["world"] = 1;
	dict1["the"] = 3;

	dict2["hello"] = 1;
	dict2["and"] = 19;


	for (auto it = dict1.begin(); it != dict1.end(); ++it)
		cout << it->first << " " << it->second << '\t';
	cout << "\n\ndict2:";
	for (auto it = dict2.begin(); it != dict2.end(); ++it)
		cout << it->first << " " << it->second << '\t';

	for (auto it = dict1.begin(); it != dict1.end(); ++it) {
		!dict2.contains(it->first) ? dict2[it->first] = it->second
								   : dict2[it->first] += it->second; // add it to the dictionary
	}

	return 0;
}
