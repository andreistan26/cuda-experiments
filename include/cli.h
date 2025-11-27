#pragma once
#include <stdlib.h>

size_t parse_size(const char *size_str) {
	const char *rem = size_str;
	size_t num = strtoull(size_str, (char **)&rem, 10);
	int shift = 1;
	switch (*rem) {
		case 'K': case 'k':
			shift = 10; break;
		case 'M': case 'm':
			shift = 20; break;
		case 'G': case 'g':
			shift = 30; break;
		default: shift = 1;
	}
	return num << shift;
}

