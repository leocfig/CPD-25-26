#include <immintrin.h>
#include <omp.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>

// Hardcoded for working with 8 doubles -> 2 256-bit avx2 registers
// Also useful since one block fills a unique cache line
#define BLOCK_SIZE 8
#define ALIGN 32

struct AlignedFree {
  void operator()(void* ptr) const {
    if (!ptr) return;
    std::free(ptr);
  }
};

template <typename T>
using AlignedPtr = std::unique_ptr<T[], AlignedFree>;

// Helper function to create a self-managed unique_ptr using aligned memory
template <typename T>
AlignedPtr<T> make_aligned(size_t size, size_t alignment = ALIGN) {
  if (alignment == 0 || (alignment & (alignment - 1))) {
    throw std::invalid_argument("Alignment must be power of 2");
  }

  size_t bytes = size * sizeof(T);
  void* ptr = nullptr;

  size_t remainder = bytes % alignment;
  size_t padded_bytes = (remainder == 0) ? bytes : (bytes + alignment - remainder);
  ptr = std::aligned_alloc(alignment, padded_bytes);

  if (!ptr) throw std::bad_alloc();

  return AlignedPtr<T>(static_cast<T*>(ptr));
}

void parse_input(std::ifstream& in_stream, AlignedPtr<uint>& assigns, AlignedPtr<double>& docs,
                 uint& C, uint& D, uint& S) {
  std::ios::sync_with_stdio(false);
  in_stream.tie(nullptr);

  if (!(in_stream >> C >> D >> S)) {
    throw std::runtime_error("Invalid Header");
  }

  // To avoid extra branch mispredictions, we pad the document array with extra "ghost" documents
  // Since we process blocks of documents, we don't have to bounds checking. These ghost documents
  // are initialized to 0, so they contribute nothing to the result
  uint padded_doc_size = (D + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

  assigns = make_aligned<uint>(padded_doc_size);
  docs = make_aligned<double>(padded_doc_size * S);

  std::fill_n(docs.get(), padded_doc_size * S, 0.0);

  // Initialize clusters using round-robin manner
  for (uint d = 0; d < D; d++) assigns[d] = d % C;
  for (uint d = D; d < padded_doc_size; d++) assigns[d] = C;  // Ghost cluster

  uint doc_id;

  for (uint d = 0; d < D; d++) {
    in_stream >> doc_id;  // We don't need this

    uint block_idx = d / BLOCK_SIZE;
    uint lane = d % BLOCK_SIZE;
    size_t block_start_offset = block_idx * (S * BLOCK_SIZE);

    for (uint s = 0; s < S; s++) {
      in_stream >> docs[block_start_offset + (s * BLOCK_SIZE) + lane];
    }
  }
}

inline void print_result(const uint* assigns, uint D) {
  for (uint d = 0; d < D; d++) {
    std::cout << assigns[d] << '\n';
  }
  std::cout << std::flush;
}

void update_step(const double* __restrict__ docs, double* __restrict__ centroids,
                 double* __restrict__ accum_sums, uint* __restrict__ accum_counts,
                 const uint* __restrict__ assigns, uint C, uint D, uint S) {
  std::fill_n(accum_sums, (C + 1) * S, 0);
  std::fill_n(accum_counts, (C + 1), 0);

  for (uint block_offset = 0; block_offset < D; block_offset += BLOCK_SIZE) {
    for (uint i = 0; i < BLOCK_SIZE; i++) {
      uint k = assigns[block_offset + i];
      accum_counts[k]++;
    }

    size_t block_start_idx = (block_offset / BLOCK_SIZE) * (S * BLOCK_SIZE);

    for (uint subj_idx = 0; subj_idx < S; subj_idx++) {
      const double* subj_weights = &docs[block_start_idx + (subj_idx * BLOCK_SIZE)];

      for (uint i = 0; i < BLOCK_SIZE; ++i) {
        double val = subj_weights[i];
        uint k = assigns[block_offset + i];

        accum_sums[k * S + subj_idx] += val;
      }
    }
  }

  for (uint cluster_idx = 0; cluster_idx < C; cluster_idx++) {
    double count = static_cast<double>(accum_counts[cluster_idx]);
    double scale = (count >= 1) ? (1.0 / count) : 0.0;  // Avoid div by zero
    uint cluster_start = cluster_idx * S;

    for (uint subj_idx = 0; subj_idx < S; subj_idx++) {
      double val = accum_sums[cluster_start + subj_idx] * scale;

      centroids[cluster_start + subj_idx] = val;
    }
  }
}

bool reassign_step(const double* __restrict__ docs, const double* __restrict__ centroids,
                   uint* __restrict__ assigns, uint C, uint D, uint D_padded, uint S) {
  uint changed_count = 0;
  const __m256d MAX_DOUBLE = _mm256_set1_pd(std::numeric_limits<double>::max());

  for (uint block_idx = 0; block_idx < D_padded; block_idx += BLOCK_SIZE) {
    size_t block_start = (block_idx / BLOCK_SIZE) * (S * BLOCK_SIZE);

    // Keep track of minimum distances and best cluster indices
    // We use 2 256-bit registers (A, B) -> 4 + 4 = 8 Doubles
    __m256d min_dist_A = MAX_DOUBLE;
    __m256d min_dist_B = MAX_DOUBLE;
    // Indices are 32bit integers, however we use doubles to simplify the code (avoids
    // castings)
    __m256d best_cluster_A = _mm256_setzero_pd();
    __m256d best_cluster_B = _mm256_setzero_pd();

    uint c = 0;
    // Using manual loop unrolling to optimize the pipeline by comparing 4 centroids
    for (; c + 4 <= C; c += 4) {
      __m256d d0_A = _mm256_setzero_pd(), d0_B = _mm256_setzero_pd();
      __m256d d1_A = _mm256_setzero_pd(), d1_B = _mm256_setzero_pd();
      __m256d d2_A = _mm256_setzero_pd(), d2_B = _mm256_setzero_pd();
      __m256d d3_A = _mm256_setzero_pd(), d3_B = _mm256_setzero_pd();

      for (uint s = 0; s < S; ++s) {
        // Load dimension s of the first 8 documents (split across 2 registers)
        __m256d doc_A = _mm256_load_pd(&docs[block_start + s * BLOCK_SIZE]);
        __m256d doc_B = _mm256_load_pd(&docs[block_start + s * BLOCK_SIZE + 4]);

        // For each dimension we accumulate the squared distances (x_s - c_k)**2
        // fmadd -> Fused Multiply Add. Multiplies (doc_X - centY) by itself and adds to
        // the distance
        __m256d cent0 = _mm256_set1_pd(centroids[(c + 0) * S + s]);
        d0_A = _mm256_fmadd_pd(_mm256_sub_pd(doc_A, cent0), _mm256_sub_pd(doc_A, cent0), d0_A);
        d0_B = _mm256_fmadd_pd(_mm256_sub_pd(doc_B, cent0), _mm256_sub_pd(doc_B, cent0), d0_B);

        __m256d cent1 = _mm256_set1_pd(centroids[(c + 1) * S + s]);
        d1_A = _mm256_fmadd_pd(_mm256_sub_pd(doc_A, cent1), _mm256_sub_pd(doc_A, cent1), d1_A);
        d1_B = _mm256_fmadd_pd(_mm256_sub_pd(doc_B, cent1), _mm256_sub_pd(doc_B, cent1), d1_B);

        __m256d cent2 = _mm256_set1_pd(centroids[(c + 2) * S + s]);
        d2_A = _mm256_fmadd_pd(_mm256_sub_pd(doc_A, cent2), _mm256_sub_pd(doc_A, cent2), d2_A);
        d2_B = _mm256_fmadd_pd(_mm256_sub_pd(doc_B, cent2), _mm256_sub_pd(doc_B, cent2), d2_B);

        __m256d cent3 = _mm256_set1_pd(centroids[(c + 3) * S + s]);
        d3_A = _mm256_fmadd_pd(_mm256_sub_pd(doc_A, cent3), _mm256_sub_pd(doc_A, cent3), d3_A);
        d3_B = _mm256_fmadd_pd(_mm256_sub_pd(doc_B, cent3), _mm256_sub_pd(doc_B, cent3), d3_B);
      }

      // Efficient argmin with SIMD: https://en.algorithmica.org/hpc/algorithms/argmin/
      // Except we use blend instructions instead of testz

      // We have 8 distances and must find the argmin.
      // NOTE: This is much easier to do with AVX512 __mmask8. However Deucalion x86 nodes
      // don't support it.
      // 1. Create a LT compare mask between two distance vectors.
      // 2. Use blendv to merge two distances producing the minimum values of both
      // 3. Create argmin vector -> current closest cluster id's

      // NOTE: These steps must be done two elements at a time. We have to perform a
      // "tournament" to find the minimum distances. ((1<2)<(3<4)) and ((5<6)<(7<8))

      __m256d idx0 = _mm256_set1_pd(static_cast<double>(c + 0));
      __m256d idx1 = _mm256_set1_pd(static_cast<double>(c + 1));
      __m256d idx2 = _mm256_set1_pd(static_cast<double>(c + 2));
      __m256d idx3 = _mm256_set1_pd(static_cast<double>(c + 3));

      __m256d min01_A = _mm256_min_pd(d0_A, d1_A);
      __m256d cmp10_A = _mm256_cmp_pd(d1_A, d0_A, _CMP_LT_OQ);
      __m256d idx01_A = _mm256_blendv_pd(idx0, idx1, cmp10_A);

      __m256d min23_A = _mm256_min_pd(d2_A, d3_A);
      __m256d cmp32_A = _mm256_cmp_pd(d3_A, d2_A, _CMP_LT_OQ);
      __m256d idx23_A = _mm256_blendv_pd(idx2, idx3, cmp32_A);

      __m256d local_min_A = _mm256_min_pd(min01_A, min23_A);
      __m256d cmp0123_A = _mm256_cmp_pd(min23_A, min01_A, _CMP_LT_OQ);
      __m256d local_idx_A = _mm256_blendv_pd(idx01_A, idx23_A, cmp0123_A);

      __m256d global_cmp_A = _mm256_cmp_pd(local_min_A, min_dist_A, _CMP_LT_OQ);
      min_dist_A = _mm256_min_pd(min_dist_A, local_min_A);
      best_cluster_A = _mm256_blendv_pd(best_cluster_A, local_idx_A, global_cmp_A);

      __m256d min01_B = _mm256_min_pd(d0_B, d1_B);
      __m256d cmp10_B = _mm256_cmp_pd(d1_B, d0_B, _CMP_LT_OQ);
      __m256d idx01_B = _mm256_blendv_pd(idx0, idx1, cmp10_B);

      __m256d min23_B = _mm256_min_pd(d2_B, d3_B);
      __m256d cmp32_B = _mm256_cmp_pd(d3_B, d2_B, _CMP_LT_OQ);
      __m256d idx23_B = _mm256_blendv_pd(idx2, idx3, cmp32_B);

      __m256d local_min_B = _mm256_min_pd(min01_B, min23_B);
      __m256d cmp0123_B = _mm256_cmp_pd(min23_B, min01_B, _CMP_LT_OQ);
      __m256d local_idx_B = _mm256_blendv_pd(idx01_B, idx23_B, cmp0123_B);

      __m256d global_cmp_B = _mm256_cmp_pd(local_min_B, min_dist_B, _CMP_LT_OQ);
      min_dist_B = _mm256_min_pd(min_dist_B, local_min_B);
      best_cluster_B = _mm256_blendv_pd(best_cluster_B, local_idx_B, global_cmp_B);
    }

    // We unroll 4 centroids at time, so we must deal with the remainder (ex: 6 centroids
    // -> 2 remainder)
    for (; c < C; ++c) {
      __m256d d_A = _mm256_setzero_pd();
      __m256d d_B = _mm256_setzero_pd();
      for (uint s = 0; s < S; ++s) {
        __m256d doc_A = _mm256_load_pd(&docs[block_start + s * 8 + 0]);
        __m256d doc_B = _mm256_load_pd(&docs[block_start + s * 8 + 4]);
        __m256d cent = _mm256_set1_pd(centroids[c * S + s]);
        d_A = _mm256_fmadd_pd(_mm256_sub_pd(doc_A, cent), _mm256_sub_pd(doc_A, cent), d_A);
        d_B = _mm256_fmadd_pd(_mm256_sub_pd(doc_B, cent), _mm256_sub_pd(doc_B, cent), d_B);
      }

      __m256d idx_c = _mm256_set1_pd(static_cast<double>(c));

      __m256d cmp_A = _mm256_cmp_pd(d_A, min_dist_A, _CMP_LT_OQ);
      min_dist_A = _mm256_min_pd(min_dist_A, d_A);
      best_cluster_A = _mm256_blendv_pd(best_cluster_A, idx_c, cmp_A);

      __m256d cmp_B = _mm256_cmp_pd(d_B, min_dist_B, _CMP_LT_OQ);
      min_dist_B = _mm256_min_pd(min_dist_B, d_B);
      best_cluster_B = _mm256_blendv_pd(best_cluster_B, idx_c, cmp_B);
    }

    // Retrieve the 4 + 4 indices as (32-bit integer) and merge into 256-bit register
    __m128i idx_A_128 = _mm256_cvtpd_epi32(best_cluster_A);
    __m128i idx_B_128 = _mm256_cvtpd_epi32(best_cluster_B);
    __m256i new_idx_256 = _mm256_inserti128_si256(_mm256_castsi128_si256(idx_A_128), idx_B_128, 1);

    // Create a mask and compare which assignments will change (a = b ? 0x0 : 0xFFFFFFFF)
    // Then convert that 256bit mask to a bit mask (8bits -> 1 per document)
    __m256i old_idx_256 = _mm256_loadu_si256((__m256i*)&assigns[block_idx]);
    __m256i eq_mask = _mm256_cmpeq_epi32(new_idx_256, old_idx_256);
    int eq_bits = _mm256_movemask_ps(_mm256_castsi256_ps(eq_mask));

    // Mask out ghost documents
    int valid_docs = std::min(BLOCK_SIZE, std::max(0, (int)D - (int)block_idx));
    int valid_mask = (1 << valid_docs) - 1;
    int changed_bits = (~eq_bits) & valid_mask;

    // Accumulate bits. If changed_count remains 0 at the end -> Converged!
    changed_count |= changed_bits;

    // Store the new assignments
    if (valid_docs == BLOCK_SIZE) {
      _mm256_storeu_si256((__m256i*)&assigns[block_idx], new_idx_256);
    } else if (__glibc_unlikely(valid_docs > 0)) {
      // valid_docs is the number of remaining non ghost documents. We create a mask for the valid
      // docs
      __m256i v_docs = _mm256_set1_epi32(valid_docs);
      __m256i v_seq = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
      __m256i store_mask = _mm256_cmpgt_epi32(v_docs, v_seq);

      // Only writes to memory if the mask lane's MSB is 1.
      // Ghost documents will be calculated, we cannot update their assignements, otherwise the
      // algorithm breaks
      _mm256_maskstore_epi32(reinterpret_cast<int*>(&assigns[block_idx]), store_mask, new_idx_256);
    }
  }

  return (changed_count != 0) ? true : false;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " IN_FILE" << std::endl;
    return 1;
  }

  std::ifstream input;
  uint C, D, S;

  AlignedPtr<uint> assignments;
  AlignedPtr<double> docs;

  try {
    input.open(argv[1]);

    if (!input) throw std::runtime_error("Failed to parse file '" + std::string(argv[1]));

    parse_input(input, assignments, docs, C, D, S);

    input.close();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  uint D_padded = (D + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

  AlignedPtr<double> centroids = make_aligned<double>(C * S);
  AlignedPtr<double> accum_sums = make_aligned<double>((C + 1) * S);
  AlignedPtr<uint> accum_counts = make_aligned<uint>(C + 1);

  double exec_time = -omp_get_wtime();

  update_step(docs.get(), centroids.get(), accum_sums.get(), accum_counts.get(), assignments.get(),
              C, D, S);

  while (reassign_step(docs.get(), centroids.get(), assignments.get(), C, D, D_padded, S)) {
    update_step(docs.get(), centroids.get(), accum_sums.get(), accum_counts.get(),
                assignments.get(), C, D, S);
  }

  exec_time += omp_get_wtime();

  std::cerr << exec_time << "s" << '\n';
  print_result(assignments.get(), D);

  // All allocated memory will be cleaned up by RAII

  return 0;
}
