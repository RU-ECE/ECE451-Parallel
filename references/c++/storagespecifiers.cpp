int x; // x is global

void f() {
	auto x = 2; // x is on the stack (auto)
	static auto y = 2; // y is in global space, but not visible to other functions
	x++;
	y++;
	const auto p = new int[100];
	delete[] p;
}

int main() {
	constexpr auto pi = 3.14159265358979323846;
	f();
	f();
	int x; // x is on the stack (auto)
	static int y; // y is in global space, but not visible to other functions
	const auto p = new int[100];
	delete[] p;
}
