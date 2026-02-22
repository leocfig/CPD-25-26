#include <immintrin.h>
#include <omp.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>

#define BLOCK_SIZE 32
#define ALIGN 32

struct AlignedFree {
  void operator()(void* ptr) const {
    if (!ptr) return;
    std::free(ptr);
  }
};

template <typename T>
using AlignedPtr = std::unique_ptr<T[], AlignedFree>;

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

template <typename T>
inline T assume_aligned(T ptr, size_t align = ALIGN) {
  return static_cast<T>(__builtin_assume_aligned(ptr, align));
}

void parse_input(std::ifstream& in_stream, AlignedPtr<uint>& assigns, AlignedPtr<double>& docs,
                 uint& C, uint& D, uint& S) {
  std::ios::sync_with_stdio(false);
  in_stream.tie(nullptr);

  if (!(in_stream >> C >> D >> S)) {
    throw std::runtime_error("Invalid Header");
  }

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

void print_result(const uint* assignments, uint D) {
  for (uint d = 0; d < D; d++) {
    std::cout << assignments[d] << '\n';
  }
  std::cout << std::flush;
}

void update_step(const double* __restrict__ docs_raw, double* __restrict__ centroids_raw,
                 double* __restrict__ accum_sums, uint* __restrict__ accum_counts,
                 const uint* __restrict__ assigns_raw, uint C, uint D, uint S) {
  const double* __restrict__ docs = assume_aligned<const double*>(docs_raw);
  double* __restrict__ centroids = assume_aligned<double*>(centroids_raw);
  const uint* __restrict__ assigns = assume_aligned<const uint*>(assigns_raw);

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

uint reassign_step(const double* __restrict__ docs_raw, const double* __restrict__ centroids_raw,
                   uint* __restrict__ assigns_raw, uint C, uint D, uint D_padded, uint S) {
  const double* __restrict__ docs = assume_aligned<const double*>(docs_raw);
  const double* __restrict__ centroids = assume_aligned<const double*>(centroids_raw);
  uint* __restrict__ assigns = assume_aligned<uint*>(assigns_raw);

  uint changed_count = 0;

  for (uint block_idx = 0; block_idx < D_padded; block_idx += BLOCK_SIZE) {
    size_t block_start = (block_idx / BLOCK_SIZE) * (S * BLOCK_SIZE);

    double min_dist[BLOCK_SIZE];
    uint best_cluster[BLOCK_SIZE];

    for (uint lane = 0; lane < BLOCK_SIZE; lane++) {
      min_dist[lane] = std::numeric_limits<double>::max();
      best_cluster[lane] = 0;
    }

    uint c = 0;
    // 1. Unrolled Centroid Loop (4 at a time)
    for (; c + 4 <= C; c += 4) {
      double dist0[BLOCK_SIZE] = {0};
      double dist1[BLOCK_SIZE] = {0};
      double dist2[BLOCK_SIZE] = {0};
      double dist3[BLOCK_SIZE] = {0};

      // Calculate distances across all dimensions
      for (uint s = 0; s < S; ++s) {
        double cent0 = centroids[(c + 0) * S + s];
        double cent1 = centroids[(c + 1) * S + s];
        double cent2 = centroids[(c + 2) * S + s];
        double cent3 = centroids[(c + 3) * S + s];
        const double* doc_block = &docs[block_start + s * BLOCK_SIZE];

// Tell GCC to ignore dependencies and vectorize
#pragma GCC ivdep
        for (uint lane = 0; lane < BLOCK_SIZE; lane++) {
          double doc_val = doc_block[lane];

          double diff0 = doc_val - cent0;
          double diff1 = doc_val - cent1;
          double diff2 = doc_val - cent2;
          double diff3 = doc_val - cent3;

          dist0[lane] += diff0 * diff0;
          dist1[lane] += diff1 * diff1;
          dist2[lane] += diff2 * diff2;
          dist3[lane] += diff3 * diff3;
        }
      }
#pragma GCC ivdep
      for (uint lane = 0; lane < BLOCK_SIZE; lane++) {
        dist0[lane] = sqrt(dist0[lane]);
        dist1[lane] = sqrt(dist1[lane]);
        dist2[lane] = sqrt(dist2[lane]);
        dist3[lane] = sqrt(dist3[lane]);
      }

      // Update argmin for all 4 unrolled centroids
      for (uint lane = 0; lane < BLOCK_SIZE; lane++) {
        if (dist0[lane] < min_dist[lane]) {
          min_dist[lane] = dist0[lane];
          best_cluster[lane] = c + 0;
        }
        if (dist1[lane] < min_dist[lane]) {
          min_dist[lane] = dist1[lane];
          best_cluster[lane] = c + 1;
        }
        if (dist2[lane] < min_dist[lane]) {
          min_dist[lane] = dist2[lane];
          best_cluster[lane] = c + 2;
        }
        if (dist3[lane] < min_dist[lane]) {
          min_dist[lane] = dist3[lane];
          best_cluster[lane] = c + 3;
        }
      }
    }

    // 2. Handle remaining tail centroids (up to 3 left over)
    for (; c < C; ++c) {
      double dist[BLOCK_SIZE] = {0};
      for (uint s = 0; s < S; ++s) {
        double cent = centroids[c * S + s];
        const double* doc_block = &docs[block_start + s * BLOCK_SIZE];

#pragma GCC ivdep
        for (uint lane = 0; lane < BLOCK_SIZE; lane++) {
          double diff = doc_block[lane] - cent;
          dist[lane] += sqrt(diff * diff);
        }
      }
#pragma GCC ivdep
      for (uint lane = 0; lane < BLOCK_SIZE; lane++) {
        if (dist[lane] < min_dist[lane]) {
          min_dist[lane] = dist[lane];
          best_cluster[lane] = c;
        }
      }
    }

    uint remaining_docs = (block_idx + BLOCK_SIZE > D) ? (D - block_idx) : BLOCK_SIZE;
    for (uint lane = 0; lane < remaining_docs; lane++) {
      if (assigns[block_idx + lane] != best_cluster[lane]) {
        changed_count++;
        assigns[block_idx + lane] = best_cluster[lane];
      }
    }
  }

  return changed_count;
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

  print_result(assignments.get(), D);

  std::cerr << exec_time << "s" << '\n';

  // All allocated memory will be cleaned up by RAII

  return 0;
}