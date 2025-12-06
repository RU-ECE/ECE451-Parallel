auto x = 1; // global

int main() {
	auto x = 2; // auto (on stack)
	::x = 3; // global
	x = 5;
	if constexpr (2 < 3)
		auto x = 6;
}
