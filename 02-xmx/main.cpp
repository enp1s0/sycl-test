// Ref: https://github.com/intel/llvm-test-suite/blob/2e687fcd57149112d7b29e9e85bf74ce85f1119c/SYCL/Matrix/joint_matrix_half_impl.hpp
#include <cstdio>
#include <vector>
#include <CL/sycl.hpp>

using in_t = sycl::ext::oneapi::bfloat16;
using out_t = float;
constexpr std::uint32_t FRAG_M = 8;
constexpr std::uint32_t FRAG_N = 8;
constexpr std::uint32_t FRAG_K = 16;

int main() {
  std::vector<in_t>  mat_a(FRAG_M * FRAG_K);
  std::vector<in_t>  mat_b(FRAG_K * FRAG_N);
  std::vector<out_t> mat_c(FRAG_M * FRAG_N);

  sycl::buffer<in_t > dev_mat_a{mat_a.data(), sycl::range<1>(FRAG_M * FRAG_K)};
  sycl::buffer<in_t > dev_mat_b{mat_b.data(), sycl::range<1>(FRAG_K * FRAG_N)};
  sycl::buffer<out_t> dev_mat_c{mat_c.data(), sycl::range<1>(FRAG_M * FRAG_N)};

  // Init
  sycl::queue queue{sycl::gpu_selector_v};
  std::printf("device = %s\n",
              queue.get_device().get_info<sycl::info::device::name>().c_str());

  // Matmul
  queue.submit([&](sycl::handler& cgh) {
    auto acc_a = dev_mat_a.get_access<sycl::access::mode::read      >(cgh);
    auto acc_b = dev_mat_b.get_access<sycl::access::mode::read      >(cgh);
    auto acc_c = dev_mat_c.get_access<sycl::access::mode::read_write>(cgh);

    cgh.parallel_for(sycl::nd_range(sycl::range{FRAG_K}, sycl::range{FRAG_K}), [acc_a, acc_b, acc_c](sycl::nd_item<1> it) {
      const auto sg = it.get_sub_group();

      sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, in_t , sycl::ext::oneapi::experimental::matrix::use::a          , FRAG_M, FRAG_K, sycl::ext::oneapi::experimental::matrix::layout::row_major> frag_a;
      sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, in_t , sycl::ext::oneapi::experimental::matrix::use::b          , FRAG_K, FRAG_N, sycl::ext::oneapi::experimental::matrix::layout::col_major> frag_b;
      sycl::ext::oneapi::experimental::matrix::joint_matrix<sycl::sub_group, out_t, sycl::ext::oneapi::experimental::matrix::use::accumulator, FRAG_M, FRAG_N> frag_c;

      sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg, frag_a, acc_a.get_pointer(), FRAG_K);
      sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg, frag_b, acc_b.get_pointer(), FRAG_K);
      sycl::ext::oneapi::experimental::matrix::joint_matrix_load(sg, frag_c, acc_c.get_pointer(), FRAG_M, sycl::ext::oneapi::experimental::matrix::layout::row_major);

      sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(sg, frag_c, frag_a, frag_b, frag_c);

      sycl::ext::oneapi::experimental::matrix::joint_matrix_store(sg, frag_c, acc_c.get_pointer(), FRAG_M, sycl::ext::oneapi::experimental::matrix::layout::row_major);
    });
  }).wait();
}
