#include <iostream>
#include <algorithm>
#include <vector>
#include <CL/sycl.hpp>

using compute_t = float;

int main() {
	constexpr std::size_t N = 1lu << 20;
	const sycl::range vec_range{N};

	sycl::buffer<compute_t> vec_a{vec_range};
	sycl::buffer<compute_t> vec_b{vec_range};
	sycl::buffer<compute_t> vec_c{vec_range};

	// Init
	{
		sycl::host_accessor host_vec_a{vec_a, sycl::write_only};
		sycl::host_accessor host_vec_b{vec_b, sycl::write_only};

		for (std::size_t i = 0; i < N; i++) {
			host_vec_a[i] = i;
			host_vec_b[i] = N - i;
		}
	}

	sycl::queue queue{};

	auto command_group = [&](sycl::handler& handler) {
		constexpr auto read_t  = sycl::access::mode::read;
		constexpr auto write_t = sycl::access::mode::write;

		auto a = vec_a.get_access<read_t >(handler);
		auto b = vec_b.get_access<read_t >(handler);
		auto c = vec_c.get_access<write_t>(handler);

		handler.parallel_for(vec_range, [=](sycl::id<1> i) {c[i] = a[i] + b[i];});
	};

	queue.submit(command_group);

	{
		sycl::host_accessor host_vec_c{vec_c, sycl::read_only};

		double max_error = 0;
		for (std::size_t i = 0; i < N; i++) {
			max_error = std::max(max_error, std::abs(static_cast<double>(N) - host_vec_c[i]));
		}

		std::printf("max error = %e\n", max_error);
	}
}
