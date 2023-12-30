#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <CL/sycl.hpp>

using compute_t = float;

int main() {
  constexpr std::size_t N = 1lu << 28;
  const sycl::range vec_range{N};

  sycl::buffer<compute_t> vec_a{vec_range};
  sycl::buffer<compute_t> vec_b{vec_range};
  sycl::buffer<compute_t> vec_c{vec_range};

  // Init
  {
    sycl::host_accessor host_vec_a{vec_a, sycl::write_only};
    sycl::host_accessor host_vec_b{vec_b, sycl::write_only};

#pragma omp parallel for
    for (std::size_t i = 0; i < N; i++) {
      host_vec_a[i] = i;
      host_vec_b[i] = N - i;
    }
  }

  std::printf("vec_size = %e [GB]\n",
              N * sizeof(compute_t) * 1e-9);

  sycl::queue queue{sycl::gpu_selector_v};
  std::printf("device = %s\n",
              queue.get_device().get_info<sycl::info::device::name>().c_str());

  auto command_group = [&](sycl::handler& handler) {
    constexpr auto read_t  = sycl::access::mode::read;
    constexpr auto write_t = sycl::access::mode::write;

    auto a = vec_a.get_access<read_t >(handler);
    auto b = vec_b.get_access<read_t >(handler);
    auto c = vec_c.get_access<write_t>(handler);

    handler.parallel_for(vec_range, [=](sycl::id<1> i) {c[i] = a[i] + b[i];});
  };

  const auto start_clock = std::chrono::system_clock::now();
  queue.submit(command_group);
  queue.wait();
  const auto end_clock = std::chrono::system_clock::now();
  const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9;
  std::printf("bw = %e [GB/s]\n",
              sizeof(compute_t) * 3 * N / elapsed_time * 1e-9
             );

  {
    sycl::host_accessor host_vec_c{vec_c, sycl::read_only};

    double max_error = 0;
#pragma omp parallel for reduction(max: max_error)
    for (std::size_t i = 0; i < N; i++) {
      max_error = std::max(max_error, std::abs(static_cast<double>(N) - host_vec_c[i]));
    }

    std::printf("max error = %e\n", max_error);
  }
}
