#include <iostream>
#include <CL/sycl.hpp>

int main() {
  for (const auto& platform : sycl::platform::get_platforms()) {
    std::printf("* Platform: %s\n",
                platform.get_info<sycl::info::platform::name>().c_str());
    for (const auto& device : platform.get_devices()) {
      std::printf("  * Device : %s\n",
                  device.get_info<sycl::info::device::name>().c_str());
    }
  }
}
