#include <immintrin.h>

#include <array>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

void f1() {
  std::array<float, 32> arr{};
  __m512 v = _mm512_load_ps(arr.data());
  auto y = _mm512_fmadd_ps(v, v, v);
  float s;
  _mm512_store_ps(&s, y);
  std::cout << s << std::endl;
}

constexpr auto siz256 = sizeof(__m256) / sizeof(float);
std::array<float, siz256> vec2arr(const float* l) {
  std::array<float, siz256> lx{};
  std::copy(l, std::next(l, siz256), lx.begin());
  return lx;
}

float f2(const std::vector<float>& l, decltype(l) r) {
  __m256 base = _mm256_setzero_ps();
  for (size_t i = 0; i < l.size() && i < r.size(); i += siz256) {
    __m256 vl = _mm256_load_ps(vec2arr(l.data() + i).data());
    __m256 vr = _mm256_load_ps(vec2arr(r.data() + i).data());
    base = _mm256_fmadd_ps(vl, vr, base);
  }
  std::array<float, siz256> s;
  _mm256_store_ps(s.data(), base);
  return std::accumulate(s.cbegin(), std::next(s.cbegin(), siz256), float{});
}

// auto f2() {
//   constexpr auto siz = sizeof(__m256) / sizeof(float);
//   std::array<float, siz> arr{};  // 32*8=256;
//   for (size_t i = 0; i < siz; i++) {
//     arr[i] = i;
//   }
//   __m256 v = _mm256_load_ps(arr.data());
//   auto y = _mm256_fmadd_ps(v, v, v);  // v*v + v
//   std::array<float, siz> s;
//   _mm256_store_ps(s.data(), y);
//   return std::reduce(s.cbegin(), std::next(s.cbegin(), siz));
// }

// std::vector<__m256> は warning: ignoring attributes on template argument '__m256' [-Wignored-attributes] がでる
float f3(const std::vector<float>& li, decltype(li) ri) {
  using packedFloats = std::array<float, siz256>;
  std::vector<packedFloats> l, r;
  for (size_t i = 0; i < li.size() && i < ri.size(); i += 8) {
    l.push_back(vec2arr(li.data() + i));
    r.push_back(vec2arr(ri.data() + i));
  }

  __m256 base = _mm256_setzero_ps();
  for (size_t i = 0; i < l.size(); i++) {
    base = _mm256_fmadd_ps(_mm256_load_ps(l[i].data()), _mm256_load_ps(r[i].data()), base);
  }
  std::array<float, siz256> s;
  _mm256_store_ps(s.data(), base);
  return std::accumulate(s.cbegin(), std::next(s.cbegin(), siz256), float{});
}

int main() {
  using namespace std;
  vector<float> v;
  for (size_t i = 0; i < 100; i++) {
    v.push_back(static_cast<float>(i) / 11);
  }
  vector<float> u;
  for (size_t i = 0; i < 64; i++) {
    u.push_back(static_cast<float>(i) / 11);
  }
  const auto len = std::min(v.size(), u.size());
  const auto resp = std::inner_product(v.cbegin(), std::next(v.cbegin() + len), u.cbegin(), float{0});

  std::cout << resp << std::endl;
  std::cout << f2(v, u) << std::endl;
  std::cout << f3(v, u) << std::endl;
}