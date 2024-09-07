// Ref: https://github.com/intel/llvm-test-suite/blob/2e687fcd57149112d7b29e9e85bf74ce85f1119c/SYCL/Matrix/joint_matrix_half_impl.hpp
#include <cstdio>
#include <string>
#include <vector>
#include <CL/sycl.hpp>

using in_t = sycl::ext::oneapi::bfloat16;
using out_t = float;
constexpr std::uint32_t FRAG_M = 8;
constexpr std::uint32_t FRAG_N = 8;
constexpr std::uint32_t FRAG_K = 16;

template <class bs_t, class fp_t>
union conv_t {
  bs_t bs;
  fp_t fp;
};

template <class T>
struct same_size_bs_t {using type = void;};
template <>
struct same_size_bs_t<float> {using type = std::uint32_t;};
template <>
struct same_size_bs_t<sycl::ext::oneapi::bfloat16> {using type = std::uint16_t;};

template <class T>
std::string to_binary(const T v) {
  auto bs = conv_t<typename same_size_bs_t<T>::type, T>{.fp = v}.bs;

  std::string str = "";
  for (std::uint32_t i = 0; i < sizeof(T) * 8; i++) {
    if (bs & 0x1) {
      str = "1" + str;
    } else {
      str = "0" + str;
    }
    bs >>= 1;
  }

  return "0x" + str;
}

template <class T>
T to_fp(const typename same_size_bs_t<T>::type v) {
  return conv_t<typename same_size_bs_t<T>::type, T>{.bs = v}.fp;
}

void eval_ip(
  const in_t vec_a[FRAG_K],
  const in_t vec_b[FRAG_K],
  const std::uint32_t num_non_zero
  ) {
  std::vector<in_t>  mat_a(FRAG_M * FRAG_K);
  std::vector<in_t>  mat_b(FRAG_K * FRAG_N);
  std::vector<out_t> mat_c(FRAG_M * FRAG_N);

  // Init
  sycl::queue queue{sycl::gpu_selector_v};
  std::printf("device = %s\n",
              queue.get_device().get_info<sycl::info::device::name>().c_str());
  for (auto& v : mat_a) {v = 0;}
  for (auto& v : mat_b) {v = 0;}
  for (auto& v : mat_c) {v = 0;}
  out_t ref = 0;
  for (std::uint32_t i = 0; i < num_non_zero; i++) {
    mat_a[i] = vec_a[i];
    mat_b[i] = vec_b[i];

    ref += static_cast<out_t>(mat_a[i]) * static_cast<out_t>(mat_b[i]);

    std::printf("vec_a[%02u] = %s (%+e), vec_a[%02u] = %s (%+e)\n",
                i,
                to_binary(mat_a[i]).c_str(),
                static_cast<float>(mat_a[i]),
                i,
                to_binary(mat_b[i]).c_str(),
                static_cast<float>(mat_b[i])
                );
  }

  sycl::buffer<in_t , 2> dev_mat_a{mat_a.data(), sycl::range<2>(FRAG_M, FRAG_K)};
  sycl::buffer<in_t , 2> dev_mat_b{mat_b.data(), sycl::range<2>(FRAG_K, FRAG_N)};
  sycl::buffer<out_t, 2> dev_mat_c{mat_c.data(), sycl::range<2>(FRAG_M, FRAG_N)};

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

  sycl::host_accessor host_mat_c{dev_mat_c, sycl::read_only};
  std::printf("C = %s (%+e)\n", to_binary(host_mat_c[0][0]).c_str(), host_mat_c[0][0]);
  std::printf("R = %s (%+e)\n", to_binary(ref).c_str(), ref);
}

int main() {
  for (std::uint32_t i = 0; i < 20; i++)
  {
    std::printf("## shift = %u\n", i);
    in_t vec_a[FRAG_K] = {static_cast<in_t>(1), static_cast<in_t>(1) / (1u << 10)};
    in_t vec_b[FRAG_K] = {static_cast<in_t>(1), static_cast<in_t>(5) / (1u << i)};

    eval_ip(vec_a, vec_b, 2);
  }
}
