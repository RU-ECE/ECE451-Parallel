#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iostream>

using namespace std;
using namespace chrono;

// 0x7e92ed659b4b3490
// last 4 digits  0011 0100 1001 0000
// first number 1..127
// seconds 128..255
// third 256..38
// 3 * 5 * 7 = 105 * 64
constexpr unsigned long
	wheel[] = {0x7e92ed659b4b3490, 0xde697df279b5b3cd, 0x5b769edfa5fbcb36, 0xf7cb66decdbdbfcf, 0xf7cb66decdbdbfcf,
			   0xeb75b77bbddaf9b4, 0xf4bf4bf793cfbdfa, 0x7fedbb697fb3cf67, 0x7fedbb697fb3cf67, 0x3fdefb36beedbdde,
			   0xf7fbb6ffff67d2cd, 0x6ffbd9a6bf5d76b7, 0x6ffbd9a6bf5d76b7, 0x96ff6dbbcfbcbff9, 0xedbef25bf7f77def,
			   0xfeef2cfe7de4fb7b, 0xfeef2cfe7de4fb7b, 0xeb6e9fdd3fbb5ffd, 0xf6df6b2dfb59bff3, 0x5b74f7efaefe7dff,
			   0x5b74f7efaefe7dff, 0xfcb3ef7ebbcf65da, 0xdf7bb6f77faddafd, 0xf5be5ffefe7bfed7, 0xf5be5ffefe7bfed7,
			   0xd6d9fff7c9ffbaed, 0xefeffb7f3f9e7d7c, 0xbadfa7dfdb76b7ef, 0xbadfa7dfdb76b7ef, 0xef3fdafde6ffdb77,
			   0x7fd7cd35feebf797, 0xcdeffeff2fda7f7d, 0xcdeffeff2fda7f7d, 0xbcf7ffeede79bcf3, 0xff6d6ff3efb7fbdb,
			   0xacff7df6f6cd3fbf, 0xacff7df6f6cd3fbf, 0xbfff7ebe697ff77f, 0xdfa7b769f7b7efb7, 0xfa5fefbbdd7edfff,
			   0xfa5fefbbdd7edfff, 0xfff5fbdfb6fe6fbf, 0x3fd37fecb7cb76db, 0xfb2ffb7badfbc967, 0xfb2ffb7badfbc967,
			   0xe79acd7dbb6f75ff, 0xd7fdbffefdf5ff6f, 0x36f7fbadd3ffbdf7, 0x36f7fbadd3ffbdf7, 0xbbdb66f6edb79f7b,
			   0xefffdeebfcffdbff, 0x9fcffebf7b3cf6f9, 0x9fcffebf7b3cf6f9, 0x5dfdf36fee97fde7, 0x7ddbeb76d679fffb,
			   0xfdfffb5bbd97edef, 0xfdfffb5bbd97edef, 0xedfeffeff3f9ef9e, 0x9aedbdde6bfcdfef, 0xfebfff2fbf5bf5d6,
			   0xfebfff2fbf5bf5d6, 0xb77dbfd2dfbefb6f, 0xdbf7fecfefffdf3e, 0xb7e97eb7fda79eff, 0xb7e97eb7fda79eff,
			   0x5f35fff9eedeffef, 0xbdff4f6efbfffdbf, 0x79f5ff5feedadf6d, 0x79f5ff5feedadf6d, 0xf7badfffb7fd3ef3,
			   0xffddecfbf967ffcd, 0x2fd77fbcbfefe7db, 0x2fd77fbcbfefe7db, 0xd3ff279bdb7edffb, 0xfdffff5dbef7edf6,
			   0xdf6beddf5be7fb5b, 0xdf6beddf5be7fb5b, 0xfb7fbbcdfdbf5f7c, 0x759f79bff3ffe7ff, 0xfbf69eef3efe5db5,
			   0xfbf69eef3efe5db5, 0xf7bf4bfe9fff77fe, 0xfb6bbff77ffcfef9, 0xfd9b7b7fff7bbdfe, 0xfd9b7b7fff7bbdfe,
			   0xdeddbfbbcfffb7cd, 0xdf2ffb5f3cf7fdff, 0xf7efe7feeb7df76f, 0xf7efe7feeb7df76f, 0x7f6cdf5ff5bf7df6,
			   0xfefaffaddfff3ff7, 0x6f7fd7ff65bb6fb4, 0x6f7fd7ff65bb6fb4, 0xffd7fdbcf6f9b5fb, 0xbb7ff69affe7fecb,
			   0xaef7cbffd7edffbe, 0xaef7cbffd7edffbe, 0x9f7b76d76fedfafb, 0x7db6bfd9efb6efed, 0xd35daffffbefb3ef,
			   0xd35daffffbefb3ef, 0xedaffe4ff5b76fbf, 0x6dd75ff6fb5bfff7, 0xffadf3fbf7ffcd7e, 0xffadf3fbf7ffcd7e},
	wheelupto11[] = {
		0x7e92ed659b4b3490, 0xde697df279b5b3cd, 0x5b769edfa5fbcb36, 0xf7cb66decdbdbfcf, 0xf7cb66decdbdbfcf,
		0xeb75b77bbddaf9b4, 0xf4bf4bf793cfbdfa, 0x7fedbb697fb3cf67, 0x7fedbb697fb3cf67, 0x3fdefb36beedbdde,
		0xf7fbb6ffff67d2cd, 0x6ffbd9a6bf5d76b7, 0x6ffbd9a6bf5d76b7, 0x96ff6dbbcfbcbff9, 0xedbef25bf7f77def,
		0xfeef2cfe7de4fb7b, 0xfeef2cfe7de4fb7b, 0xeb6e9fdd3fbb5ffd, 0xf6df6b2dfb59bff3, 0x5b74f7efaefe7dff,
		0x5b74f7efaefe7dff, 0xfcb3ef7ebbcf65da, 0xdf7bb6f77faddafd, 0xf5be5ffefe7bfed7, 0xf5be5ffefe7bfed7,
		0xd6d9fff7c9ffbaed, 0xefeffb7f3f9e7d7c, 0xbadfa7dfdb76b7ef, 0xbadfa7dfdb76b7ef, 0xef3fdafde6ffdb77,
		0x7fd7cd35feebf797, 0xcdeffeff2fda7f7d, 0xcdeffeff2fda7f7d, 0xbcf7ffeede79bcf3, 0xff6d6ff3efb7fbdb,
		0xacff7df6f6cd3fbf, 0xacff7df6f6cd3fbf, 0xbfff7ebe697ff77f, 0xdfa7b769f7b7efb7, 0xfa5fefbbdd7edfff,
		0xfa5fefbbdd7edfff, 0xfff5fbdfb6fe6fbf, 0x3fd37fecb7cb76db, 0xfb2ffb7badfbc967, 0xfb2ffb7badfbc967,
		0xe79acd7dbb6f75ff, 0xd7fdbffefdf5ff6f, 0x36f7fbadd3ffbdf7, 0x36f7fbadd3ffbdf7, 0xbbdb66f6edb79f7b,
		0xefffdeebfcffdbff, 0x9fcffebf7b3cf6f9, 0x9fcffebf7b3cf6f9, 0x5dfdf36fee97fde7, 0x7ddbeb76d679fffb,
		0xfdfffb5bbd97edef, 0xfdfffb5bbd97edef, 0xedfeffeff3f9ef9e, 0x9aedbdde6bfcdfef, 0xfebfff2fbf5bf5d6,
		0xfebfff2fbf5bf5d6, 0xb77dbfd2dfbefb6f, 0xdbf7fecfefffdf3e, 0xb7e97eb7fda79eff, 0xb7e97eb7fda79eff,
		0x5f35fff9eedeffef, 0xbdff4f6efbfffdbf, 0x79f5ff5feedadf6d, 0x79f5ff5feedadf6d, 0xf7badfffb7fd3ef3,
		0xffddecfbf967ffcd, 0x2fd77fbcbfefe7db, 0x2fd77fbcbfefe7db, 0xd3ff279bdb7edffb, 0xfdffff5dbef7edf6,
		0xdf6beddf5be7fb5b, 0xdf6beddf5be7fb5b, 0xfb7fbbcdfdbf5f7c, 0x759f79bff3ffe7ff, 0xfbf69eef3efe5db5,
		0xfbf69eef3efe5db5, 0xf7bf4bfe9fff77fe, 0xfb6bbff77ffcfef9, 0xfd9b7b7fff7bbdfe, 0xfd9b7b7fff7bbdfe,
		0xdeddbfbbcfffb7cd, 0xdf2ffb5f3cf7fdff, 0xf7efe7feeb7df76f, 0xf7efe7feeb7df76f, 0x7f6cdf5ff5bf7df6,
		0xfefaffaddfff3ff7, 0x6f7fd7ff65bb6fb4, 0x6f7fd7ff65bb6fb4, 0xffd7fdbcf6f9b5fb, 0xbb7ff69affe7fecb,
		0xaef7cbffd7edffbe, 0xaef7cbffd7edffbe, 0x9f7b76d76fedfafb, 0x7db6bfd9efb6efed, 0xd35daffffbefb3ef,
		0xd35daffffbefb3ef, 0xedaffe4ff5b76fbf, 0x6dd75ff6fb5bfff7, 0xffadf3fbf7ffcd7e, 0xffadf3fbf7ffcd7e,
		0x7efefdefdf7bb6de, 0x967bfefeddbfb7ef, 0xff9eed3fd2fdfef3, 0xff9eed3fd2fdfef3, 0xf7edfebfcf3fff5f,
		0x7bfebf6b7fffdff4, 0xbe7b75d6f9feffdb, 0xbe7b75d6f9feffdb, 0x7ffcb36bf7f7ef7f, 0xafdfdfbebf69bffe,
		0xefb5dedffed779fd, 0xefb5dedffed779fd, 0xbddafbbff7ebfff7, 0xdeff6dfbefbcdefd, 0xfedbdd7dfbef76be,
		0xfedbdd7dfbef76be, 0xf6ff3ef77bf7f7eb, 0xc97ffffd75be4fbd, 0xf7f9f7fefd6dbe6b, 0xf7f9f7fefd6dbe6b,
		0x7f76d7e92ffefff6, 0xaff36ffe97ffff9f, 0xffb6bff97edadfff, 0xffb6bff97edadfff, 0xe7df5fffb6edfef7,
		0xd7dbefff59fefacf, 0xbdfed9fdff6d6ff3, 0xbdfed9fdff6d6ff3, 0xfefdbddbfbffb6e9, 0xffeff77fe5f37f77,
		0xff7becdb5fedf7cf, 0xff7becdb5fedf7cf, 0xcfffd7ef3f9aff7d, 0x37fef9eefedffcfb, 0x4fffbff9adf3fdec,
		0x4fffbff9adf3fdec, 0xeffb7967feefbf9e, 0xbe7fbe9feb6df77b, 0xef9fcbf7beffadd6, 0xef9fcbf7beffadd6,
		0xf77bffbfeffed7ff, 0xfdadfbfbb7df6fbd, 0xbfdfb7faeffdbf7d, 0xbfdfb7faeffdbf7d, 0xff3fdbf9feffff66,
		0xf7bbefeddefb7d97, 0x6df6f3dffdbedb36, 0x6df6f3dffdbedb36, 0xbcf6ffbfffffedbf, 0xffdbeefefd67df7b,
		0xeff77d7fb3dfa7db, 0xeff77d7fb3dfa7db, 0xdfeff6bef97edadb, 0xdbb5ffef7fbbefe7, 0xf6fffdf37bef9ffd,
		0xf6fffdf37bef9ffd, 0xdf3dfbeff7fe7dfd, 0xeeffffe7fbf9f697, 0xfdbeffd9fff749ef, 0xfdbeffd9fff749ef,
		0x7fdbdfe5dedbb5fe, 0xfffdbfd2fffdbbff, 0xfe9e79eed3ffecbf, 0xfe9e79eed3ffecbf, 0xb7cf7efbfdefbe7f,
		0xef75de7fafd7dfb7, 0xfadf3f9f6d7df35f, 0xfadf3f9f6d7df35f, 0x7df7fbfffebfdf2d, 0xb7be7fb7fffd7dd2,
		0xdd3f9bffff97fb3d, 0xdd3f9bffff97fb3d, 0x6eff79f6bbcdf7bf, 0xdbffb79bfffed7fd, 0x77f7dfbfdf6f74be,
		0x77f7dfbfdf6f74be, 0x9f7bfdffdfecbbfb, 0xeff7ffcfbfdb6f7e, 0xb37ffe97ed6fbf4b, 0xb37ffe97ed6fbf4b,
		0xefbfb6ed3ff7ffe7, 0xefb3db76beffe5db, 0xf9eff7ebee96ddb7, 0xf9eff7ebee96ddb7, 0xbfffefffd77b7fdb,
		0xd7f9b7f7d9f7feef, 0xeddb7dfff77dfeff, 0xeddb7dfff77dfeff, 0xfbef6fbf5fbfbffb, 0xef7fd3dfefbfdd6f,
		0xfffb7cdedff6b7ef, 0xfffb7cdedff6b7ef, 0x5fef93ffffbb5bbd, 0x36b77f7cff7fb7bb, 0xfbf7bf6defd6fbec,
		0xfbf7bf6defd6fbec, 0xe6ffdfe6ffdd7dbb, 0xbeff759f6f7efedf, 0xbdfedfb6deefbff6, 0xbdfedfb6deefbff6,
		0xfbddfebbebffd3cf, 0xef7fdefb76f76dfd, 0xfecda79b4fbff6ed, 0xfecda79b4fbff6ed, 0x7f3ef6fbbeff6ff7,
		0xfedbff65ff6ff79e, 0xdbe7ffdfb7dffff7, 0xdbe7ffdfb7dffff7, 0xfff6ffeefffdfcb7, 0xfbff67feefb5fe7b,
		0xb5f34bfffacfeffb, 0xb5f34bfffacfeffb, 0xbbdfffd7fd6ed67d, 0x5dfebf7f6ed3ff3f, 0xd3fffffbddeff3fd,
		0xd3fffffbddeff3fd, 0xcdedffff3fff79ee, 0xaffafdbfbf7f76bf, 0xebfdf67fedf76f7f, 0xebfdf67fedf76f7f,
		0x6fbbff75dfdbfeb7, 0xffed2efffdfefbfb, 0x76f7fbfddf7be6f7, 0x76f7fbfddf7be6f7, 0xbb7de797fdfddb7f,
		0x7ffed7ffbfdeffbd, 0xdfdfbffe6fbfdbff, 0xdfdfbffe6fbfdbff, 0xdff6bfffe6d7dfaf, 0xf7df7fbf9ffb7ff7,
		0xfd6dfbff3dbeff7d, 0xfd6dfbff3dbeff7d, 0xfeff5ffdf3ff7fdb, 0xfeff2fdefbbe9ff9, 0xe7beedffbbdbfff7,
		0xe7beedffbbdbfff7, 0xdffbadff7fbef3d9, 0xfde7b7df6fdbfff6, 0xbfeb67dfffb5da7b, 0xbfeb67dfffb5da7b,
		0xebbffffffffa5dff, 0xf6ff7b7fd3dfa7fe, 0xfde7ffeff6ffcff7, 0xfde7ffeff6ffcff7, 0xf7bbdbfffffdfdde,
		0xf3fdedbb4dff9fed, 0xacd6fbf4fb4ffffa, 0xacd6fbf4fb4ffffa, 0xd6df7dffcfb79eff, 0xefbcdfddacffe9ff,
		0xf7fbffd67ffcbb5d, 0xf7fbffd67ffcbb5d, 0x7bf6d7cfbf9bfb3f, 0x3fff6dfdde7fafbb, 0xdf7eff697cfb7be7,
		0xdf7eff697cfb7be7, 0xaeff7b77bafffdfa, 0xff6bb4be79aedfdf, 0xf7faff76dff9fcfb, 0xf7faff76dff9fcfb,
		0xf67dffbfff6ffadd, 0xdf3dbe5fffffffaf, 0xbbfdffbe6f7ef76f, 0xbbfdffbe6f7ef76f, 0xfb6ff679f5f3eff7,
		0x76dfcff7fb5f37ff, 0xdff6fefdb7fe4ffd, 0xdff6fefdb7fe4ffd, 0xbd9ffd7ffffbfcfb, 0xfffdf6bbefb5decf,
		0xe7bf5f6efbfdefdf, 0xe7bf5f6efbfdefdf, 0xfafff696f9fdfffb, 0xffffb7fffffefda7, 0xdf7fbdf7cff7b7df,
		0xdf7fbdf7cff7b7df, 0xfdedbffbfe976f6e, 0xeeff5ffef7edff9e, 0x69edfefffffb7bfe, 0x69edfefffffb7bfe,
		0x77d7dd77be5ffebf, 0xfeebeddb7fbdbbff, 0xfdbffd7cd6f9f7b7, 0xfdbffd7cd6f9f7b7, 0xbfef7ffaef7f9bdb,
		0xdff6fe7bbdf7fdfd, 0xbadff7f6ff7efbfd, 0xbadff7f6ff7efbfd, 0xfdf7f3d96f9eefb5, 0xfdfffbfd9fff6ede,
		0xdfa5ff5b7ffffdee, 0xdfa5ff5b7ffffdee, 0xfff6fffdfbfb77ff, 0xfbff37dfebb6f7eb, 0xfffbffff9a7f7fff,
		0xfffbffff9a7f7fff, 0x9f7bfcdeffe6bf7f, 0x7f7f9eede5dbfb3f, 0xff5ffefedf6ffbcb, 0xff5ffefedf6ffbcb,
		0xcbfdbf6d6ef7fff4, 0xbffffde69eef77bf, 0x7bfef35ffffbdd7f, 0x7bfef35ffffbdd7f, 0x75dfff75f7ffbedf,
		0xf77deefbfb77b7ff, 0x7ff37faffb5dfefe, 0x7ff37faffb5dfefe, 0xbeede7fe5f7db7ff, 0xfffedbffbeb7fbfe,
		0xff7f7cf7ddb5f7df, 0xff7f7cf7ddb5f7df, 0xcbffffff3dfefb76, 0xb5befdfdd77dfffb, 0xebfed77f6dfeffbe,
		0xebfed77f6dfeffbe, 0xbcfb4fefb3dfafff, 0xbbfffdfefb6fde5b, 0xafdfffb5fe6dfdf7, 0xafdfffb5fe6dfdf7,
		0xf7fff6fbdd6fdacd, 0xcdfffaeffede7fee, 0xfbede7bfdff4bf7f, 0xfbede7bfdff4bf7f, 0x7d3efad9efbfddff,
		0xf7beddfddbcffeff, 0xfdffbecfbffbffbd, 0xfdffbecfbffbffbd, 0x76f77b2edbdfecff, 0xff5be7bbffadfb7f,
		0xbfbf7bffd7cfbfdb, 0xbfbf7bffd7cfbfdb, 0xbffbf7f6fbeff77f, 0x7ffffffde6ffddf7, 0xfedfa7b7cbfff3ff,
		0xfedfa7b7cbfff3ff, 0xff6fffeb7fdffbbc, 0xfcffdfbfffffee92, 0x7ffcd77be7f7dbef, 0x7ffcd77be7f7dbef,
		0xe696efffbf4fbddf, 0xfffb3dfb7ffffffb, 0xfed6ef2fffddfdfb, 0xfed6ef2fffddfdfb, 0xbbe9fefbeff79f5f,
		0x7bbfb77dfdd7fdef, 0x9feffcd67fffdbfb, 0x9feffcd67fffdbfb, 0xdfaeff5f7effffb7, 0x6dfbdfbefeff2ef3,
		0xdfaddfef7fbf79fe, 0xdfaddfef7fbf79fe, 0x2ffffdf5b7efffde, 0xb7fffffb7fb7beed, 0xf7d3dfedbbfff59f,
		0xf7d3dfedbbfff59f, 0xf7f9bff67bffbbdf, 0xed67ffdf7fbfff3e, 0xff7d67f7ed6fdfef, 0xff7d67f7ed6fdfef,
		0xff3cd7fffcfb5ff5, 0xffbffbf7fbefbdbb, 0xfbf7fbffef96ed77, 0xfbf7fbffef96ed77, 0xe7dffbb6b77bfdfb,
		0xfffdfef7cf7f9bfd, 0x2ffa79feffcbf7b7, 0x2ffa79feffcbf7b7, 0xbefdbfde7b7cffff, 0xffaedf5dbfb77fef,
		0xf77fbffaddfdbffd, 0xf77fbffaddfdbffd, 0x7dfefaffe7dfef7d, 0xfed66f7efbffa7bf, 0x6b7f9ffffff2fbe7,
		0x6b7f9ffffff2fbe7, 0xedf3cf7ff2ef7d9f, 0xdedf3fbfedffffdf, 0x3fbbfff7d66fecfb, 0x3fbbfff7d66fecfb,
		0xfe7fe5ffd9ffdeff, 0xcdffdfebf5dfebec, 0x97ffbd9f7f7efef9, 0x97ffbd9f7f7efef9, 0x7ffdfb7fbdf77d6f,
		0x7ef3ff67fffbf4f6, 0xfd7ef3fff5bf7ffe, 0xfd7ef3fff5bf7ffe, 0xf6dffd6fd7fbaef7, 0xf35f7ef6ffff9fef,
		0xfffbeb7efbfdfdff, 0xfffbeb7efbfdfdff, 0xfedfbf9fedeef6fb, 0x79f5f7ffeebfcfb7, 0xffdff5ff6fe7befd,
		0xffdff5ff6fe7befd, 0xef359b7ff7fef93e, 0xafdf7dfeb77f66fb, 0x79efdbdfe7fbffe6, 0x79efdbdfe7fbffe6,
		0xffbbefeddbff3dbf, 0xfef9edffdbeff7e9, 0xfffeefbcffddafbf, 0xfffeefbcffddafbf, 0xffebfefffff5decf,
		0xfff5d6ef3efb7bef, 0x9ffbfeb7ff7efbdf, 0x9ffbfeb7ff7efbdf, 0xfbb5b7eff7f3ed7f, 0xafba7bbfd7ffefd7,
		0xff6fff5fbdfe79ff, 0xff6fff5fbdfe79ff, 0x7ff7ffbebfdd6ffa, 0xb6cfa7feff3ff7fd, 0x6ffaeff7dbff76bf,
		0x6ffaeff7dbff76bf, 0x9eedfff77df5bfcd, 0xdbeffbddffdefbff, 0xffed6fdfdd77fb7b, 0xffed6fdfdd77fb7b,
		0xebf4bffbfedbf9ac, 0xf7f36feff2fd7d9f, 0xdffebbfdfebfff7d, 0xdffebbfdfebfff7d, 0xffbeeff5bf7bbff3,
		0xdfdff7bb7ff6facf, 0xfdf7fbbff749feff, 0xfdf7fbbff749feff, 0xdbeff7bf7fbefe7f, 0xed3ff679edfbdfff,
		0x96ff2eff7fedbb4b, 0x96ff2eff7fedbb4b, 0xfbe7ffedbdff7fbf, 0xfdd7fbfddef9b6ff, 0xdb7efeef6ef7ffbf,
		0xdb7efeef6ef7ffbf, 0xefff7977b3ef2dfb, 0xbf6f75ffebfdf37f, 0xb7becf3e9fefefdf, 0xb7becf3e9fefefdf,
		0xfafdf6bffff6b7fd, 0xfde7dbdfb5df7dee, 0xdefffdfbfbb7b7f9, 0xdefffdfbfbb7b7f9, 0xedfdf3ffe7ff7b7f,
		0x76beef7dfaebfffe, 0xebffd7fdbf9feffd, 0xebffd7fdbf9feffd, 0x3dffedaedf5fbff7, 0xb3ed76bfcfe5feff,
		0xfebff9e7dedde7de, 0xfebff9e7dedde7de, 0xfbcffdb7fb6fff7f, 0xfbb7f7db77fafff7, 0xfb7df5ff6fee9aef,
		0xfb7df5ff6fee9aef, 0xfd6dfbdbb79ffb3c, 0x7fd6f9e7bf7f77bb, 0xebeffe5fb4ffcf76, 0xebeffe5fb4ffcf76,
		0xff9adfffbf6bfdff, 0xdfed7edfddbff35d, 0xbebf7fbdfb7feeff, 0xbebf7fbdfb7feeff, 0xfbdb7fd3cdbfffeb,
		0xff7cdfff6effffe6, 0xdf6fbebfffeedfdf, 0xdf6fbebfffeedfdf, 0xf9e7b76d6fbfffff, 0xefdbcbfeb77f7ef7,
		0xddf5ff6ffffeef6f, 0xddf5ff6ffffeef6f, 0xfefe7beeffddfffe, 0xfeddffbbff77dfff, 0xe6d7ff75fb5f37fe,
		0xe6d7ff75fb5f37fe, 0xf6ededdb5fadff7f, 0xdbfffbffb7fedbf7, 0xf7eb77feddf59f6f, 0xf7eb77feddf59f6f,
		0x6f3fdfff2cf7fdfe, 0xf5f74ffeffffefdb, 0x7bfdfbdbffdafdad, 0x7bfdfbdbffdafdad, 0xffbf5ffdbf6defdb,
		0xffddfef7fbeef7dd, 0x6ff6ddfff3ffff93, 0x6ff6ddfff3ffff93, 0xffdf3fdfffbfd7ef, 0xededfa7dfebfff6f,
		0xd679eff7fbb6ffdb, 0xd679eff7fbb6ffdb, 0x6f77b6cff5de7fbe, 0xf6be7f7ed6fbbeff, 0xdfbfdefd7ffafdb7,
		0xdfbfdefd7ffafdb7, 0xffb76fff9fdfa7ff, 0xbbffffb77f3fd77f, 0xedbaef7cf7effffe, 0xedbaef7cf7effffe,
		0xdfdfffbbfbf7d3ff, 0xdf37fafffedfef3f, 0xdbed779eff77bfe9, 0xdbed779eff77bfe9, 0xf9eff7ffbdff597f,
		0x7efbdfafff5fbfb7, 0xed7ebeef7d9fdb74, 0xed7ebeef7d9fdb74, 0xbefefffcf2d9f7bf, 0xffdfefdfdd35feff,
		0xaeffefff97edf5ff, 0xaeffefff97edf5ff, 0xbbff7fff7fbfff7d, 0x7fbdff6be7d6ffff, 0xdfffbdfbefeedfcf,
		0xdfffbdfbefeedfcf, 0xefb7ffdf3dbeefaf, 0x7dd7ffedbf79effb, 0xf9befaf9eeff7ff6, 0xf9befaf9eeff7ff6,
		0xffdfefe7bf7f7c9f, 0xb7e9bcf77feffb5f, 0xbdff7ffdff7df6fb, 0xbdff7ffdff7df6fb, 0xfbdf6fb6fd37bbdf,
		0x6bf597ffbddfdfbd, 0xda7fffdefdbffe7d, 0xda7fffdefdbffe7d, 0x7feff7ffe7fbff7d, 0xaffe5bffff7bfdff,
		0xefe7df5fbdff6ffd, 0xefe7df5fbdff6ffd, 0x7ffff9bff7ed7fb6, 0xfffdfdff7bb4de7f, 0x76fbfda7ffdf37f7,
		0x76fbfda7ffdf37f7, 0xdfffaffaffffff4f, 0xc9eed6ff75ffcfb5, 0xffcfffbedfbfbbff, 0xffcfffbedfbfbbff,
		0xfbfcf7e9befffdf4, 0xecb7fff7fbcff7df, 0x7feebb7f67f2fdbf, 0x7feebb7f67f2fdbf, 0xef9bff7ff6ebeeff,
		0xffdbe6ffcdf7b7ff, 0x7efeffefbbff7fd6, 0x7efeffefbbff7fd6, 0xfbdf3dfacff7ffef, 0x7f7df379efb3efe6,
		0xb7ff2efaffeff7fd, 0xb7ff2efaffeff7fd, 0xd9f79ffd6d9a7bff, 0x3ef7ff7fd67fbdbb, 0x7fbefeffbff65dbf,
		0x7fbefeffbff65dbf, 0xbfbbdf67dbedbfbe, 0xfb5f7dff6d7ddfdb, 0xedff7b3f9ff93dff, 0xedff7b3f9ff93dff,
		0xff7bffff5f7efaff, 0xff75fbfff6bef9bf, 0xf6ffbfff5ffdb77d, 0xf6ffbfff5ffdb77d, 0x6beddf5badf3dfff,
		0xf7f3fd3fdafbfebf, 0xfd7fb7dfbfdfdbff, 0xfd7fb7dfbfdfdbff, 0x7dbfefafffddecf7, 0xbb7deedfedbfffdb,
		0xe7ff5ffff7dfb7ff, 0xe7ff5ffff7dfb7ff, 0xfbef75fffdefdbdd, 0xf9bfb7fd7fdaff7f, 0xdafffdb7eb77feff,
		0xdafffdb7eb77feff, 0xeff7faeffd9fff2f, 0x7cfbdfb6f7fd6f9f, 0xebafdf5bffbfdde7, 0xebafdf5bffbfdde7,
		0xffffff7fbadff6d7, 0xbffbffd7fffebbe9, 0x3ef7edeedfdfefbb, 0x3ef7edeedfdfefbb, 0xf77ff7ffffb7ff4f,
		0xefbcdfff6dfaf9b6, 0xffdfbfbfefbffadb, 0xffdfbfbfefbffadb, 0x7ffcb35f7ffadfef, 0x3dfedff5de7ffff7,
		0xfdfdde7ff7bfff7e, 0xfdfdde7ff7bfff7e, 0x2ff75fb6f77feff3, 0xdfcff7ff5bfdfeff, 0xf6dffde7ff7b3dfe,
		0xf6dffde7ff7b3dfe, 0xf6fb6fdf7ffcfb7f, 0xfd7efaffffbbefbf, 0xf7cffff6ef75dfdb, 0xf7cffff6ef75dfdb,
		0xdb37f6ebfcf37ff7, 0xe6ffdf6ffadff5fa, 0x7fafffdbffd7ef7f, 0x7fafffdbffd7ef7f, 0x7fdeef7ef77f6eff,
		0xde7db5bfcfeffbff, 0xfff3dffeffff7fdb, 0xfff3dffeffff7fdb, 0xf6cfef9ffb3ef77d, 0xededffffadff7f7e,
		0xffffbef7fba7bfeb, 0xffffbef7fba7bfeb, 0x7f7ff7fd77ffffbc, 0x3ffeedbff65dbff7, 0xefffdfed7df2fdf7,
		0xefffdfed7df2fdf7, 0xaefbdfe7bedffffb, 0xfffffc97fbeffe5b, 0x3ddb7ff5fffd3efe, 0x3ddb7ff5fffd3efe,
		0xff5fffb7fdf7fbff, 0xdf67be7f7eb7e9bf, 0xffcfe7fe7fbff7fd, 0xffcfe7fe7fbff7fd, 0xf9bdd7fff7bfef7e,
		0xef9effefbf4b3edf, 0x4befbffdeffadb3e, 0x4befbffdeffadb3e, 0xffdf7fafdb7fe4f7, 0xfffff6f7fd3fbbdb,
		0xb7bbfffff6ff7fbe, 0xb7bbfffff6ff7fbe, 0xfe7bbddffffed65f, 0xffaefbef7fd3efb5, 0xf37ba5fff97ef6cf,
		0xf37ba5fff97ef6cf, 0xddef9effffffedfd, 0xbff7dbfeb74f7eb7, 0x7bfcffddb5f36fff, 0x7bfcffddb5f36fff,
		0xfffedfefdaef7fd6, 0x9f6dbfd3ffe4ffdf, 0x75f7ff7edbfbfdff, 0x75f7ff7edbfbfdff, 0xffeffffeffbfbafb,
		0xff7dffe9efde59f5, 0xdfcbfeffffeffb7d, 0xdfcbfeffffeffb7d, 0xfbefff7be7faefb7, 0xeffbff7ebeffeff3,
		0xeffd9b7ffedffd7e, 0xeffd9b7ffedffd7e, 0x3df779fdff6d6fff, 0x9fff2ddffff7f77b, 0xffbfff77fbcfb5fe,
		0xffbfff77fbcfb5fe, 0xdf6fadfefdaff3ff, 0xcfffdfcfedffef7e, 0xfb796ffbefbdbeff, 0xfb796ffbefbdbeff,
		0x5bb7be6ffff7dfbe, 0xecfff9fef6ffffdf, 0xd9f6bfedffbfddaf, 0xd9f6bfedffbfddaf, 0xf7fbfb77dff9bfd7,
		0xffffb5b3cffefbff, 0x6dfa7ffdbbffff9e, 0x6dfa7ffdbbffff9e, 0xb6ed7ffeebf5deff, 0x7bbcfa7bfdf7e97f,
		0xfff9fff7ddbeffd9, 0xfff9fff7ddbeffd9, 0x7d7ef7ef77fafbff, 0xfffffbaddbffbfff, 0xff3fbe7d7fdf7fee,
		0xff3fbe7d7fdf7fee, 0xbebfe9fff3dffd9f, 0xbf5ff7ff7fefdafd, 0x77fecf37bff97ff2, 0x77fecf37bff97ff2,
		0xdafff7f75fe7fadd, 0xdfb7ff6bbfbfeffd, 0xfeeffffefbffbf7b, 0xfeeffffefbffbf7b, 0xffbefff9febf7dff,
		0xffd7dfbfdffffcf6, 0xffefd2ddfffe7f3f, 0xffefd2ddfffe7f3f, 0xfc966bfdffdbeeff, 0xf7eff7f6fdedbbdf,
		0xadfbfffefefd6dbf, 0xadfbfffefefd6dbf, 0xfefbf7bfebbffb5f, 0xfdedf7ebf6dbddff, 0xffffeef3cbfedfdd,
		0xffffeef3cbfedfdd, 0xef6f9efbf5f76fff, 0x6fd7fdfdbbcdffdf, 0x7defd35fbdfbff6f, 0x7defd35fbdfbff6f,
		0xf6ffffeffeef77f7, 0xfeebfff3fff4bfd9, 0xbfdffb6ff3fdf6f7, 0xbfdffb6ff3fdf6f7, 0xff7f6ef3fdedffff,
		0xfbbedffdfdfb5bf7, 0xfadbbfff6f7fd679, 0xfadbbfff6f7fd679, 0xdba7fffdffb6ddff, 0xefdbfff4bfef7dfa,
		0xdfaffbff77dfefff, 0xdfaffbff77dfefff, 0xfffedffebf796ffb, 0xffedbffe6b75fefb, 0xffbfddad9fdfbcbf,
		0xffbfddad9fdfbcbf, 0xbe7d2edadfbfb7ef, 0xffffd7ef7d9fff7d, 0xf3df6efefdbddbcf, 0xf3df6efefdbddbcf,
		0x6ffdbffd2eff7bee, 0xb7ffffeedfff37bf, 0xdff4ff69e7bfff7d, 0xdff4ff69e7bfff7d, 0xfddf7ff6dfffeed7,
		0xd2ffedff6967f7ff, 0xfcffdffcffdff7f3, 0xfcffdffcffdff7f3, 0xbafde7ba7f7fdffd, 0x7f6efeffe7bf5d7f,
		0xd6f9fdfefbbffbdf, 0xd6f9fdfefbbffbdf, 0x7f66dbff3fdfff37, 0xf7be7d3cfefffdbf, 0x6bf5dfebadffdff5,
		0x6bf5dfebadffdff5, 0xfef3fbffd7efeffe, 0xba5f77fffdbed6ff, 0xffdf7b7ff76daff6, 0xffdf7b7ff76daff6,
		0xd75bfcfbdbeef7ef, 0xfdbfbeefbdd7fdff, 0xbeffffdbcf37bffd, 0xbeffffdbcf37bffd, 0xf9fdf3dfedb7fdee,
		0xfffeff7fdbdffebe, 0xeffefecdfffaefb5, 0xeffefecdfffaefb5, 0xfeb77baddf7dfeff, 0xffffef9bffafff7b,
		0xbfffed7fd3ff3dde, 0xbfffed7fd3ff3dde, 0xbf4bbdbf6ffdff5f, 0x7ff7bf7feeffff7f, 0xfbd9a7ffff7ff3fd,
		0xfbd9a7ffff7ff3fd, 0xfdf79ffff5befb7f, 0xfcffffa7ff4dee9f, 0x7fffd779e4fbfbff, 0x7fffd779e4fbfbff,
		0x6ffecffdfb6ff7d7, 0xffff7edffbfeffef, 0xbdd7f9eeffdff4bf, 0xbdd7f9eeffdff4bf, 0xff7bf7dfdf6ddbdf,
		0xffb7fe7dbffbffbc, 0xde7f3cb7e9fefbff, 0xde7f3cb7e9fefbff, 0xffbeff6bf6fbfded, 0xf7bbffbc9f7b7ed7,
		0xff77feeb7ed7f9ff, 0xff77feeb7ed7f9ff, 0xadd77fe5ffef7fbf, 0x9fef7dbfdff7d7ff, 0xefffedfdbfeb7dff,
		0xefffedfdbfeb7dff, 0xfffbeedffdfdf3fd, 0xfb7eb7efe7feffb7, 0xfbdf7ffedde7dfcb, 0xfbdf7ffedde7dfcb,
		0xff7fff79efd37bbd, 0xbfff5bfebfefffba, 0xdbffbf6d7edfffff, 0xdbffbf6d7edfffff, 0x25feeff7dffdbef7,
		0xfffbe5bf7be7bbff, 0x3edbdfeffbddf7f6, 0x3edbdfeffbddf7f6, 0xdfdd7fbbeffe97ef, 0xf9afd7fbbeff7bef,
		0xbe7defdbfdeebffd, 0xbe7defdbfdeebffd, 0xdfefd3ff379fdf75, 0xfd9eebefd37bfefb, 0xdff4feffbdfaffe7,
		0xdff4feffbdfaffe7, 0xb5fffd7edffd75db, 0xfecfffffffbff2df, 0xffff6ff7f7effcfe, 0xffff6ff7f7effcfe,
		0xfffbfdffd9779fed, 0xcffffffb7effff6d, 0xbfeff7ffdf7edfff, 0xbfeff7ffdf7edfff, 0xfbeddffdbff76b6e,
		0xefdfefffffdff5f6, 0xdb7efbeda79e5f7f, 0xdb7efbeda79e5f7f, 0xf5bef9ffd6dfe7ff, 0xf3ef6ffffd67ff7f,
		0xfef36f7effdf67df, 0xfef36f7effdf67df, 0xbbcbfddf7f7ffafd, 0x5ffcfbdbe7f7edb5, 0xff7fffb75ff7dfdf,
		0xff7fffb75ff7dfdf, 0xdf7ffb7ff6fffdef, 0xadd7ffffbb79f79f, 0xffffdbfba7bb5dff, 0xffffdbfba7bb5dff,
		0x7ffbddedfafffdb6, 0xfffbefffffe6f7fb, 0xb7f7fffffe7dbfbf, 0xb7f7fffffe7dbfbf, 0xffcb7f9bfdfffecf,
		0x5bbfde7d6df2fbfd, 0x9fff7fffff2dfffb, 0x9fff7fffff2dfffb, 0x5df6ff5befb2efbd, 0xb59bfff7fefd3ff6,
		0xfd7f9efb3ebfefbf, 0xfd7f9efb3ebfefbf, 0xfddfdfeebf7dfffe, 0xfeffb5dfdfbfff6b, 0x6ffbed7fbfdfffff,
		0x6ffbed7fbfdfffff, 0xbefdbff2fdb7f7ff, 0x6feedefd3dffff7e, 0xffdd7f9fef7fbfeb, 0xffdd7f9fef7fbfeb,
		0x7f77f7ef6dde5dfe, 0xbeffcff7d7ffffba, 0xffedb7ffffdfdfaf, 0xffedb7ffffdfdfaf, 0x67feef7df7f9fffb,
		0xdbdfb4bb7fffdbfd, 0xfcf7d9bef76ffef7, 0xfcf7d9bef76ffef7, 0xf6ff7dfbffbfde7f, 0xef7ffb5df5bffdee,
		0xffef3ff7f9efbb5f, 0xffef3ff7f9efbb5f, 0xdf7ebffff7bfebff, 0xf6ff6f6ddefde4f3, 0xfff7fefffef7fffd,
		0xfff7fefffef7fffd, 0xfcf7fdeebbdfafbf, 0xdfeb7dfffffff7fd, 0xf7ffefffd6fb7edb, 0xf7ffefffd6fb7edb,
		0xfff9b7ffedfebfcf, 0xeff5db4bbdfe7f7e, 0x9edff5ffffb7f67b, 0x9edff5ffffb7f67b, 0xfdafdbfbfff37bff,
		0x7fdffd75feffffbf, 0xf9effefffddbebf7, 0xf9effefffddbebf7, 0x36f7e9bff779b7f7, 0xbfebf6bfff6ffbff,
		0xa6bf7b77fbfdefde, 0xa6bf7b77fbfdefde, 0xffcfbcd7ff6cf7fb, 0xdbfeffdfffbffdbf, 0xf3fbfffffd7ff6fd,
		0xf3fbfffffd7ff6fd, 0xffbfdfdf7ff77dfd, 0x2dfbfdf7f7df66bb, 0x7ffffedfaffbe966, 0x7ffffedfaffbe966,
		0xfff3dd3fbe7ffff7, 0xffefaeffdfbfbfef, 0xf4fe7bfdd3fffdff, 0xf4fe7bfdd3fffdff, 0xfb5dfffffdbfdefb,
		0xcf7fbefb7edbdfbe, 0xdfebbdfffbafd2ff, 0xdfebbdfffbafd2ff, 0xfdf7ffedfff7ef77, 0xedffffffb76dfdff,
		0xef3fbaebb7ffeffc, 0xef3fbaebb7ffeffc, 0xbffefbedffdf7edf, 0xdaffff9bebffb7ff, 0x67b3ff7fdefbf7de,
		0x67b3ff7fdefbf7de, 0xfffbbff7fffcff6d, 0xfdeedffdefdfff3d, 0xb77d7ef2cdb7bb7f, 0xb77d7ef2cdb7bb7f,
		0xdfff9fff3dd2f9af, 0xefffff6fdfeffd9f, 0x7beffffbf6befd7f, 0x7beffffbf6befd7f, 0xf7bf5bf7dffdbdf7,
		0xfefdfffbc9ffdfdd, 0xfefeddfdb3ffffdb, 0xfefeddfdb3ffffdb, 0xbbef6ffa5f77fefd, 0x7bbfff7fbcbf6ffe,
		0x9eeffcf7dffefbdd, 0x9eeffcf7dffefbdd, 0x4ffffffdb5fbffbf, 0xfcffffeddff9efb7, 0xcbffde6ffcfbddbd,
		0xcbffde6ffcfbddbd, 0xfff3cbf7feffb7de, 0xff7fffbeffaddaff, 0x7dfaeff7ffffeffa, 0x7dfaeff7ffffeffa,
		0xffffbff76bffb7cf, 0xffef9fdfb7dffb7e, 0xd2dfefbfdf7dfe7d, 0xd2dfefbfdf7dfe7d, 0x6dffffffb7fffd7e,
		0xf7ffcffd9e5fbef7, 0xcfffbffd7fffef7c, 0xcfffbffd7fffef7c, 0x3fbfef7efffff7fb, 0xb7cbfededff7fedb,
		0xecfffdefbfcf75fa, 0xecfffdefbfcf75fa, 0xfe6ffedfef3ddeff, 0xfbf5bfffeff6df3d, 0xff5fbeb77dffbfdf,
		0xff5fbeb77dffbfdf, 0xedb7dfff77ffefff, 0xfcfb7bf6ff5f7fde, 0xfd2ef3ddefb77d7e, 0xfd2ef3ddefb77d7e,
		0xefd3ff7f9b6bfeff, 0xfffbbcfb7dfdffff, 0xbfb67fbfdffde7b3, 0xbfb67fbfdffde7b3, 0xf7cdf6dfddfdbb7f,
		0x6f7ef7ffbffa7be7, 0xff5f3edefbfedfdf, 0xff5f3edefbfedfdf, 0xfdfefbf9e69bdfaf, 0xf7de4fffb7f9fdfb,
		0xefb7fadffffe69ff, 0xefb7fadffffe69ff, 0xfcf3fdfebffb7fb2, 0xf3ff7fbbfff79fff, 0x6fffddafdeefffd7,
		0x6fffddafdeefffd7, 0xf6effddbfdeff7ef, 0xdff7beedf7fefff7, 0xbf59f7feef6ddf6b, 0xbf59f7feef6ddf6b,
		0xfb7dfeef3ffe5ff7, 0xffbfe977dbffbdbb, 0xfbfebb5bfefbddf7, 0xfbfebb5bfefbddf7, 0xeddbcffef7fdffff,
		0xfbffe7f7fbf6f3df, 0x3efeffafb7df7e9f, 0x3efeffafb7df7e9f, 0xbffdefffeffef6ff, 0x6f2cf6fdfebf5bf6,
		0xbffbefdf59fff7ff, 0xbffbefdf59fff7ff, 0x7fffbbdf2dbbfff7, 0x75f7fbbfdf79f6bf, 0xff7df7f9efff7fbe,
		0xff7df7f9efff7fbe, 0xb5fbdde7daffb7ff, 0xbbdf3effef6dfb5f, 0x7dff6b759effffd7, 0x7dff6b759effffd7,
		0xf7ffbff77d7fdeff, 0xfdb5ffff7efee97d, 0xf7fff5bffbfd96ef, 0xf7fff5bffbfd96ef, 0x6f7ddfffbef7f9ef,
		0xf7d7ffffbfeb75bf, 0x7fe7ffdfa7ffffbf, 0x7fe7ffdfa7ffffbf, 0xfcf7fbffdbfdbdff, 0xff7deffacfb7be6f,
		0xe6b75fffbbeffdfb, 0xe6b75fffbbeffdfb, 0xdb7b37de7b6edbff, 0xddedbfcf7ffbfdfd, 0xf67dfffffb7efbff,
		0xf67dfffffb7efbff, 0xff3fbbcfbfbf6dff, 0xefff59f6f7fd7efe, 0xfbbfd77ffdf3ffe7, 0xfbbfd77ffdf3ffe7,
		0x6f9fffbffbdbb4ff, 0xff6faffa7dbfffff, 0xfff66feedffbefbf, 0xfff66feedffbefbf, 0xbf7fffb3efadfeff,
		0xcffdfefdbffeffa6, 0xffcf3ff7e9bffefd, 0xffcf3ff7e9bffefd, 0xffedbbebf7dffff7, 0x3fbb7fffbeffffde,
		0xdffdff6ffdf76bed, 0xdffdff6ffdf76bed, 0x7ff7d9beff7de7f7, 0xf7cf7fde5fb6fffd, 0x6ffecd679fffbdff,
		0x6ffecd679fffbdff, 0xd7fd7effffefffdb, 0xff7fffffefdfcfbe, 0xfffdffb7efe7bfff, 0xfffdffb7efe7bfff,
		0xeb76bffdfcff7bbd, 0xe6b7fbfffbddbfff, 0x5bf7f7ffe7dedff7, 0x5bf7f7ffe7dedff7, 0x3ffffbf4df7b7dfe,
		0xf7d9b7fbd9fffbef, 0xfff7ffafb7effeb7, 0xfff7ffafb7effeb7, 0xdfcfefdf5ff6f6eb, 0xfbfff3fbedff7d6f,
		0xfef9aef7dbf7b75d, 0xfef9aef7dbf7b75d, 0x4dfefeffffffcb3d, 0xbfd6e9feff7fb7ff, 0xefffbefd7efafffe,
		0xefffbefd7efafffe, 0xfff3ef67f7fd37bb, 0xdbefffd77fefdf5d, 0xfffbdb3edfebeefe, 0xfffbdb3edfebeefe,
		0xfaffbdf7f9effbef, 0xff6f9f5fbfff7fff, 0xb6fdfddb7f7edeef, 0xb6fdfddb7f7edeef, 0x7ffcd6f9b7fff97f,
		0xfed7eff7ffdbfefe, 0xeffefbdffddfcbbf, 0xeffefbdffddfcbbf, 0xbff7ededf3fff7fb, 0xfbcf77bffff5ffcb,
		0xfcb7efeffbcfafbb, 0xfcb7efeffbcfafbb, 0xfedf3ff7e9fef7fb, 0xffbff7eff793efe5, 0xf7dbffbf7d67faff,
		0xf7dbffbf7d67faff, 0xefe7db7ffddefbff, 0xefdedfbfffcffffe, 0x7b7ffafbf7bf6f7e, 0x7b7ffafbf7bf6f7e,
		0x7fbafd2fdbef76b7, 0xbf7dadd77bfcf77b, 0xff9fffeefe7db7f3, 0xff9fffeefe7db7f3, 0xf37ffffeed7f9eef,
		0x7bfffffbfff65dbd, 0x9feff7ff6ffdf7fd, 0x9feff7ff6ffdf7fd, 0xffb5bffbf6d7ed6f, 0xadfb7ffeff7dfddb,
		0xeff5ffdf7edfff6f, 0xeff5ffdf7edfff6f, 0x7ffbdbb7f3fffe9f, 0xdbff37dbeffff76f, 0xffbefd77fedfffd6,
		0xffbefd77fedfffd6, 0xbffbbcfedff4fbf9, 0xeff7dacd7dfffb7e, 0xfb7fefbbff2fffdf, 0xfb7fefbbff2fffdf,
		0xfffef77fefdbfdad, 0xbcfbfdf7dfffefbf, 0xdfe6ff5bfeb2dfed, 0xdfe6ff5bfeb2dfed, 0xf7bb5ffcff7b7efa,
		0xffddefb3fdffdeff, 0xbddf7bffb7fbefdf, 0xbddf7bffb7fbefdf, 0xffef75be7ff5fefd, 0x7f7fdf7ffff3ebff,
		0xd6ffbfdf7dbebfef, 0xd6ffbfdf7dbebfef, 0xfb6fffffaffadff4, 0xf5ff79eddedfffb3, 0xffbe9ff97ff35fbf,
		0xffbe9ff97ff35fbf, 0xfeff7fffb6ffffde, 0x9febffdefdbff6db, 0x7ffbdbffdf6ffff7, 0x7ffbdbffdf6ffff7,
		0xff7deff3cfeffbfd, 0xef35fafbb5befdff, 0xfadfbdffff779feb, 0xfadfbdffff779feb, 0xfdaef77df7f7fde7,
		0xfeb7ef37ffdf7cff, 0xfbfedbfdf7befbff, 0xfbfedbfdf7befbff, 0x3fb76d6efffbfffb, 0xf7cffefeffffde6f,
		0xadfbdfffd6ffe5ff, 0xadfbdfffd6ffe5ff, 0xfbfff7bffffffef9, 0xffffbfeff7faffe5, 0xdbdbfdffdbefdfff,
		0xdbdbfdffdbefdfff, 0xefeffa7f7ef76b3e, 0x7ef77df7bbf9ffbe, 0x7fadfedffffbffe6, 0x7fadfedffffbffe6,
		0xfff7ffe7bffb7fbf, 0xb6ff7fdfffa4ffdf, 0x7efe7fbdf7dfa7ff, 0x7efe7fbdf7dfa7ff, 0xfbfdfeb6edff9fff,
		0xdb7edf7bbfff5dff, 0xbe7ff7b6ffbdfbfd, 0xbe7ff7b6ffbdfbfd, 0xfdbefbfdffb7fdb7, 0xfddbfffdb7efefff,
		0xdfa7bf7b75deef6f, 0xdfa7bf7b75deef6f, 0xbfdafffcfbefff97, 0xfedf2fffcfb7f67f, 0xfef3cff5fe5fbf9f,
		0xfef3cff5fe5fbf9f, 0xbeffbffbfdbfbf4d, 0xcdfeffef7fbbdb7f, 0xff5f66ffffbffefb, 0xff5f66ffffbffefb,
		0xcbbff77feedffbfd, 0xf6ffef7fb7ff7fbf, 0xfff7f3fdff96ffef, 0xfff7f3fdff96ffef, 0xbfdfdf7cfffdeede,
		0xf77dadff7bff9bcf, 0x7dfb5bffbf7f6ffe, 0x7dfb5bffbf7f6ffe, 0xb7efe7bf7fbeb7ff, 0xfdfdf6ddaeff5f7f};

class prime_bits {
	unsigned long n; // the number of bits
	unsigned long num_words; // the number of 64-bit words needed to store
	unsigned long* p; // point to the dynamic memory for the bits
public:
	explicit prime_bits(const unsigned long n) : // round up because we need n+1 bits
		n(n + 1), num_words(n / 64 + 1), p(new unsigned long[n / 64 + 1]) {
		for (auto i = 0UL; i < num_words; i++)
			p[i] = 0;
	}
	~prime_bits() { delete[] p; }
	// prevent you from accidentally copying bits. You really don't want to
	prime_bits(const prime_bits& orig) = delete;
	prime_bits& operator=(const prime_bits& orig) = delete;

	// let's store primes as 0 and not prime (composite) as 1 it's easier.
	// to set a bit we shift the mask to the right spot and OR it in
	void clear_prime(const unsigned long i) const {
		// this is storing only odd numbers in each mask!
		// note 1LL is crucial. If you write just 1 it's an int.
		// (1 << 40) would be 0
		// (1LL << 40) is 10000000000000000000000000...
		p[i / 128] |= 1LL << ((i % 128) >> 1);
	}
	[[nodiscard]] bool is_prime(const unsigned long i) const { return (p[i / 128] & 1LL << ((i % 128) >> 1)) == 0; }

	// passing in size so we can use this to initialize subsections
	[[nodiscard]] unsigned long eratosthenes(const unsigned long size) const {
		unsigned long count = 1; // 2 is a special case
		const auto lim = static_cast<unsigned long>(sqrt(size));
		for (auto i = 3UL; i <= lim; i += 2) {
			if (is_prime(i)) {
				count++;
				for (auto j = i * i; j <= size; j += 2 * i)
					clear_prime(j);
			}
		}
		for (auto i = lim + 1 | 1; i <= size; i += 2)
			if (is_prime(i))
				count++;
		return count;
	}
	// use bitcounting to avoid counting each one separately
	[[nodiscard]] unsigned long fast_eratosthenes() const {
		auto count = 1UL; // 2 is a special case
		auto lim = static_cast<unsigned long>(sqrt(n));
		for (auto i = 3UL; i <= lim; i += 2) {
			if (is_prime(i)) {
				count++;
				for (auto j = i * i; j <= n; j += 2 * i)
					clear_prime(j);
			}
		}
		/*
			TODO: check if this this boundary condition is right.
			handle the few primes between sqrt(n) and the next 64-bit word boundary
		*/
		const auto word_index = (lim + 127) / 128;
		for (auto i = lim + 1 | 1; i < word_index * 128; i++)
			clear_prime(i);

		lim = (lim + 127) / 128; // round up to next even word boundary

		for (auto i = lim; i < num_words; i++)
			count += _mm_popcnt_u64(~p[i]); // count how many bits are set
		return count;
	}

	// use bitcounting to avoid counting each one separately
	[[nodiscard]] unsigned long fast2_eratosthenes() const {
		auto count = 1UL; // 2 is a special case
		auto lim = static_cast<unsigned long>(sqrt(n));
		auto k = 0UL;
		for (auto i = 0UL; i < num_words; i++) {
			p[i] = wheel[k++];
			// p[i] = wheelupto11[k++];
			if (k >= 105) // 1154 for wheel11
				k = 0;
		}
		// 11*11 = 121 + 11*2 = 143 + 11*2 = 165
		// 11, 13, 17, 19, 23, 29, 31

		for (auto i = 11UL; i <= lim; i += 2) {
			if (is_prime(i)) {
				count++;
				for (auto j = i * i; j <= n; j += 2 * i)
					clear_prime(j);
			}
		}
		/*
			TODO: check if this this boundary condition is right.
			handle the few primes between sqrt(n) and the next 64-bit word boundary
		*/
		const auto word_index = (lim + 127) / 128;
		for (auto i = lim + 1 | 1; i < word_index * 128; i++)
			clear_prime(i);

		lim = (lim + 127) / 128; // round up to next even word boundary

		for (auto i = lim; i < num_words; i++)
			count += _mm_popcnt_u64(~p[i]); // count how many bits are set
		return count;
	}

	// use bitcounting to avoid counting each one separately
	[[nodiscard]] unsigned long segmented_eratosthenes(const unsigned long blocksize) const {
		auto count = eratosthenes(static_cast<unsigned long>(sqrt(n)));
		const auto start = (static_cast<unsigned long>(sqrt(n)) + 127) / 128; // round up to next word location
		[[maybe_unused]] auto end = start + blocksize;
		auto lim = static_cast<unsigned long>(sqrt(n));
		auto k = 0UL;
		for (auto i = 0UL; i < num_words; i++) {
			//			p[i] = wheel[k++];
			p[i] = wheelupto11[k++];
			if (k >= 1155) // 105 for wheel
				k = 0;
		}

		for (auto i = 13UL; i <= lim; i += 2) {
			if (is_prime(i)) {
				count++;
				for (auto j = i * i; j <= n; j += 2 * i)
					clear_prime(j);
			}
		}
		/*
		 * TODO: check if this this boundary condition is right. Handle the few primes between sqrt(n) and the next
		 * 64-bit word boundary
		 */
		const auto word_index = (lim + 127) / 128;
		for (auto i = lim + 2 | 1; i < word_index * 128; i++)
			clear_prime(i);

		lim = (lim + 127) / 128; // round up to next even word boundary

		for (auto i = lim; i < num_words; i++)
			count += _mm_popcnt_u64(~p[i]); // count how many bits are set
		return count;
	}

	// use bitcounting to avoid counting each one separately
	void build_wheel(const unsigned long s) const { // 3*5*7 = 105
		const auto size = 128 * s;
		for (auto i = 3UL; i <= size; i += 2) {
			if (is_prime(i))
				for (auto j = i * i; j <= size; j += 2 * i)
					clear_prime(j);
		}
		cout << "const uint64_t wheel[] = {\n" << hex;
		for (auto i = 0UL; i < s; i += 4) {
			for (auto j = i; j < i + 3; j++)
				cout << "0x" << p[j] << ", ";
			cout << "0x" << p[i + 4] << ",\n";
		}
		cout << "};\n" << dec;
	}
};

int main(const int argc, char* argv[]) {
	if (argc < 2) {
		cerr << "pass in the number of primes on the command line\n";
		exit(-1);
	}
	const auto n = strtol(argv[1], nullptr, 10);
	const prime_bits primes(n);
	// COPYING prime_bits objects is ILLEGAL (no copy constructor)
	// see me if you have questions
	//	prime_bits p2 = primes;
	primes.build_wheel(3 * 5 * 7); // 2,3,5,7 repeats in 3*5*7 = 105
	primes.build_wheel(3 * 5 * 7 * 11); // 2,3,5,7,11 repeats in 3*5*7*11 = = 1155
	auto t0 = high_resolution_clock::now();
	auto count = primes.eratosthenes(n);
	auto t1 = high_resolution_clock::now();
	auto elapsed = duration_cast<microseconds>(t1 - t0);
	cout << "bitwise eratosthenes: " << count << " elapsed: " << elapsed.count() << "usec\n";
	t0 = high_resolution_clock::now();
	count = primes.fast_eratosthenes();
	t1 = high_resolution_clock::now();
	elapsed = duration_cast<microseconds>(t1 - t0);
	cout << "fast eratosthenes: " << count << " elapsed: " << elapsed.count() << "usec\n";
	t0 = high_resolution_clock::now();
	count = primes.fast2_eratosthenes();
	t1 = high_resolution_clock::now();
	elapsed = duration_cast<microseconds>(t1 - t0);
	cout << "fast2 eratosthenes: " << count << " elapsed: " << elapsed.count() << "usec\n";
}
