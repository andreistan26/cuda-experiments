#pragma once

#include <span>
#include <numeric>
#include <type_traits>

template <typename T>
concept Arithmetic = requires {
	std::is_arithmetic_v<T>;
};

template <Arithmetic T>
void print_stats(std::span<T> spn, std::string unit) {
	if (spn.size() == 0) {
		std::cerr << "Could not print stats, no datapoints." << std::endl;
		return;
	}

    std::sort(spn.begin(), spn.end());
    T median = spn[spn.size() / 2];
    T p99 = spn[size_t(0.99 * spn.size())];
    T min_val = spn.front();
    T max_val = spn.back();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Samples : " << spn.size() << std::endl;
    std::cout << "Median  : " << median << " " << unit << std::endl;
	std::cout << "Mean    : " << std::reduce(spn.begin(), spn.end()) / spn.size() << " " << unit << std::endl;
    std::cout << "Min     : " << min_val << " " << unit << std::endl;
    std::cout << "Max     : " << max_val << " " << unit << std::endl;
    std::cout << "P99     : " << p99 << " " << unit << std::endl;
}
