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
	for (auto& [fst, snd] : dict1)
		cout << fst << " " << snd << '\t';
	cout << "\n\ndict2:";
	for (auto& [fst, snd] : dict2)
		cout << fst << " " << snd << '\t';
	for (auto& [fst, snd] : dict1)
		!dict2.contains(fst) ? dict2[fst] = snd : dict2[fst] += snd; // add it to the dictionary
	return 0;
}
