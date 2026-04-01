#include <omp.h>
#include <mpi.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <iomanip>
#include <cmath>

// Hardcoded for working with 8 doubles -> 2 256-bit avx2 registers
// Also useful since one block fills a unique cache line
#define BLOCK_SIZE 8
#define ALIGN 32

// Docs Decomposition Macros
#define DOCS_LOW(id,p,n)  ((id)*(n)/(p))
#define DOCS_HIGH(id,p,n) (DOCS_LOW((id)+1,p,n) - 1)
#define DOCS_SIZE(id,p,n) (DOCS_HIGH(id,p,n) - DOCS_LOW(id,p,n) + 1)

#define SEED 1234
#define RAND_RANGE 10.0
#define UNIF01 ((double) rand() / RAND_MAX)

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
  if (!size) return AlignedPtr<T>(nullptr);

  size_t bytes = size * sizeof(T);
  void* ptr = nullptr;

  size_t remainder = bytes % alignment;
  size_t padded_bytes = (remainder == 0) ? bytes : (bytes + alignment - remainder);
  ptr = std::aligned_alloc(alignment, padded_bytes);

  if (!ptr) throw std::bad_alloc();

  return AlignedPtr<T>(static_cast<T*>(ptr));
}

struct Grid {
  MPI_Comm grid_comm, row_comm, col_comm;
  int my_row, my_col;
  int nrows, ncols;
};

Grid make_grid(int num_procs, int id) {
  Grid g;
  int sq = (int)std::round(std::sqrt((double)num_procs));
  if (sq * sq != num_procs)
    throw std::runtime_error("Number of processes must be a perfect square");

  g.nrows = g.ncols = sq;

  int dims[2]    = {sq, sq};
  int periods[2] = {0, 0};
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &g.grid_comm);

  int coords[2];
  MPI_Cart_coords(g.grid_comm, id, 2, coords);
  g.my_row = coords[0];
  g.my_col = coords[1];

  MPI_Comm_split(g.grid_comm, g.my_row, g.my_col, &g.row_comm);
  MPI_Comm_split(g.grid_comm, g.my_col, g.my_row, &g.col_comm);
  return g;
}

void parse_input(std::ifstream& in_stream, AlignedPtr<uint>& assigns, AlignedPtr<double>& docs,
                 uint& C, uint& D, uint& S, const Grid& g,
                 uint& task_nr_docs, uint& task_first_doc,
                 uint& task_nr_cents, uint& task_first_cent) {
  std::ios::sync_with_stdio(false);
  in_stream.tie(nullptr);

  if (!(in_stream >> C >> D >> S)) {
    throw std::runtime_error("Invalid Header");
  }

  // Distributed block decomposition
  task_first_doc  = DOCS_LOW(g.my_row, g.nrows, D);
  task_nr_docs    = DOCS_SIZE(g.my_row, g.nrows, D);
  task_first_cent = DOCS_LOW(g.my_col, g.ncols, C);
  task_nr_cents   = DOCS_SIZE(g.my_col, g.ncols, C);

  srand(SEED);

  // Skipping
  for (uint d = 0; d < task_first_doc; d++)
    for (uint s = 0; s < S; s++)
        UNIF01;

  // To avoid extra branch mispredictions, we pad the document array with extra "ghost" documents
  // Since we process blocks of documents, we don't have to bounds checking. These ghost documents
  // are initialized to 0, so they contribute nothing to the result
  uint padded_doc_size = (task_nr_docs + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

  assigns = make_aligned<uint>(padded_doc_size);
  docs    = make_aligned<double>(padded_doc_size * S);

  std::fill_n(docs.get(), padded_doc_size * S, 0.0);

  // NOTE: This is a modified version of our parsing that uses RNG for the subject values.
  // This allows us to test extremely big instances in Deucalion without having to generate large files
  for (uint d = 0; d < task_nr_docs; d++) {
    uint block_idx = d / BLOCK_SIZE;
    uint lane = d % BLOCK_SIZE;
    size_t block_start_offset = block_idx * (S * BLOCK_SIZE);

    for (uint s = 0; s < S; s++) {
      docs[block_start_offset + (s * BLOCK_SIZE) + lane] = UNIF01 * RAND_RANGE;
    }
  }
}

AlignedPtr<uint> gather_results(const uint* local_assigns, uint task_nr_docs, uint D, const Grid& g) {
  MPI_Comm col0_comm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, g.my_col == 0 ? 0 : MPI_UNDEFINED, g.my_row, &col0_comm);

  if (g.my_col != 0) return nullptr;

  if (!g.my_row) {
    AlignedPtr<int> recvcounts = make_aligned<int>(g.nrows);
    AlignedPtr<int> displs     = make_aligned<int>(g.nrows);

    displs[0] = 0;
    for (int i = 0; i < g.nrows; i++) {
      recvcounts[i] = (int)DOCS_SIZE(i, g.nrows, D);
      if (i > 0) displs[i] = displs[i-1] + recvcounts[i-1];
    }

    AlignedPtr<uint> all_assigns = make_aligned<uint>(D);

    MPI_Gatherv(local_assigns, (int)task_nr_docs, MPI_UNSIGNED,
                all_assigns.get(), recvcounts.get(), displs.get(), MPI_UNSIGNED,
                0, col0_comm);

    MPI_Comm_free(&col0_comm);
    return all_assigns;
  } else {
    MPI_Gatherv(local_assigns, (int)task_nr_docs, MPI_UNSIGNED,
                nullptr, nullptr, nullptr, MPI_UNSIGNED,
                0, col0_comm);

    MPI_Comm_free(&col0_comm);
    return nullptr;
  }
}

inline void print_result(const uint* assigns, uint D) {
  for (uint d = 0; d < D; d++)
    std::cout << assigns[d] << '\n';
  std::cout << std::flush;
}

void update_step(const double* __restrict__ docs, double* __restrict__ centroids,
                 const uint* __restrict__ assigns, uint S,
                 uint task_nr_docs, uint task_nr_cents, uint task_first_cent,
                 double* __restrict__ all_sums, uint* __restrict__ all_counts, int number_threads,
                 double* __restrict__ mpi_recv_buf,
                 uint changed_count, bool& changed, double& total_comm_time,
                 const Grid& g) {

  int tid = omp_get_thread_num();
  double* local_sums   = all_sums   + (size_t)tid * (task_nr_cents + 1) * S;
  uint*   local_counts = all_counts + (size_t)tid * (task_nr_cents + 1);

  std::fill_n(local_sums,   (task_nr_cents + 1) * S, 0.0);
  std::fill_n(local_counts,  task_nr_cents + 1,       0u);

  #pragma omp for
  for (uint block_offset = 0; block_offset < task_nr_docs; block_offset += BLOCK_SIZE) {
    for (uint i = 0; i < BLOCK_SIZE; i++) {
      uint k = assigns[block_offset + i];
      uint lk = (k >= task_first_cent && k < task_first_cent + task_nr_cents)
                ? (k - task_first_cent) : task_nr_cents;
      local_counts[lk]++;
    }

    size_t block_start_idx = (block_offset / BLOCK_SIZE) * (S * BLOCK_SIZE);

    for (uint subj_idx = 0; subj_idx < S; subj_idx++) {
      const double* subj_weights = &docs[block_start_idx + (subj_idx * BLOCK_SIZE)];

      for (uint i = 0; i < BLOCK_SIZE; ++i) {
        double val = subj_weights[i];
        uint k = assigns[block_offset + i];
        uint lk = (k >= task_first_cent && k < task_first_cent + task_nr_cents)
                  ? (k - task_first_cent) : task_nr_cents;
        local_sums[lk * S + subj_idx] += val;
      }
    }
  }

  // Reduce OMP thread-local arrays into mpi_recv_buf - each thread owns a slice of clusters.
  // Counts go to mpi_recv_buf[task_nr_cents*S .. task_nr_cents*S+task_nr_cents] and sums go to mpi_recv_buf[0 .. task_nr_cents*S].
  #pragma omp for
  for (uint k = 0; k < task_nr_cents; k++) {
    double sum_count = 0.0;
    for (int t = 0; t < number_threads; t++)
      sum_count += all_counts[(size_t)t * (task_nr_cents + 1) + k];
    mpi_recv_buf[task_nr_cents * S + k] = sum_count;

    for (uint s = 0; s < S; s++) {
      double sum = 0.0;
      for (int t = 0; t < number_threads; t++)
        sum += all_sums[(size_t)t * (task_nr_cents + 1) * S + k * S + s];
      mpi_recv_buf[k * S + s] = sum;
    }
  }

  #pragma omp single
  {
    // Append changed_count at the end of the buffer so it gets reduced together with sums and counts
    mpi_recv_buf[task_nr_cents * S + task_nr_cents] = (double)changed_count;

    double t_comm = -MPI_Wtime();

    MPI_Allreduce(MPI_IN_PLACE, mpi_recv_buf,
                  (int)(task_nr_cents * S + task_nr_cents + 1),
                  MPI_DOUBLE, MPI_SUM, g.col_comm); 

    double global_changed = mpi_recv_buf[task_nr_cents * S + task_nr_cents];
    MPI_Bcast(&global_changed, 1, MPI_DOUBLE, 0, g.row_comm);

    t_comm += MPI_Wtime();

    total_comm_time += t_comm;
    changed = (global_changed != 0.0);
  }

  // Verify convergence
  if (!changed) return;

  // Compute centroids using recv_buf global
  #pragma omp for
  for (uint k = 0; k < task_nr_cents; k++) {
    double count = mpi_recv_buf[task_nr_cents * S + k];
    double scale = (count >= 1.0) ? (1.0 / count) : 0.0;
    for (uint s = 0; s < S; s++)
      centroids[k * S + s] = mpi_recv_buf[k * S + s] * scale;
  }
}

struct DistIdx {
  double dist;
  int    idx;
};

void reassign_step(const double* __restrict__ docs, const double* __restrict__ centroids,
                   uint* __restrict__ assigns, uint C_padded_local,
                   uint task_nr_docs, uint D_padded, uint S,
                   uint task_first_cent, DistIdx* __restrict__ local_pairs,
                   uint& changed_count, const Grid& g) {

  #pragma omp for
  for (uint block_idx = 0; block_idx < D_padded; block_idx += BLOCK_SIZE) {
    size_t block_start = (block_idx / BLOCK_SIZE) * (S * BLOCK_SIZE);

    double min_dist[BLOCK_SIZE];
    int    best_idx[BLOCK_SIZE];
    for (uint i = 0; i < BLOCK_SIZE; i++) {
      min_dist[i] = std::numeric_limits<double>::max();
      best_idx[i] = (int)C_padded_local; // sentinel
    }

    for (uint c = 0; c < C_padded_local; c += 4) {
      double d0[BLOCK_SIZE] = {0}, d1[BLOCK_SIZE] = {0};
      double d2[BLOCK_SIZE] = {0}, d3[BLOCK_SIZE] = {0};

      for (uint s = 0; s < S; ++s) {
        double cent0 = centroids[(c + 0) * S + s];
        double cent1 = centroids[(c + 1) * S + s];
        double cent2 = centroids[(c + 2) * S + s];
        double cent3 = centroids[(c + 3) * S + s];
        const double* doc_block = &docs[block_start + s * BLOCK_SIZE];

        #pragma GCC ivdep
        for (uint lane = 0; lane < BLOCK_SIZE; lane++) {
          double v = doc_block[lane];
          double diff0 = v - cent0, diff1 = v - cent1;
          double diff2 = v - cent2, diff3 = v - cent3;
          d0[lane] += diff0 * diff0;
          d1[lane] += diff1 * diff1;
          d2[lane] += diff2 * diff2;
          d3[lane] += diff3 * diff3;
        }
      }

      // argmin tournament (mesma lógica do AVX original, mas escalar)
      #pragma GCC ivdep
      for (uint lane = 0; lane < BLOCK_SIZE; lane++) {
        // d0 vs d1
        double m01, m23;
        int    i01, i23;
        if (d1[lane] < d0[lane]) { m01 = d1[lane]; i01 = (int)(task_first_cent + c + 1); }
        else                     { m01 = d0[lane]; i01 = (int)(task_first_cent + c + 0); }

        if (d3[lane] < d2[lane]) { m23 = d3[lane]; i23 = (int)(task_first_cent + c + 3); }
        else                     { m23 = d2[lane]; i23 = (int)(task_first_cent + c + 2); }

        double local_min = (m23 < m01) ? m23 : m01;
        int    local_idx = (m23 < m01) ? i23 : i01;

        if (local_min < min_dist[lane]) {
          min_dist[lane] = local_min;
          best_idx[lane] = local_idx;
        }
      }
    }

    for (uint i = 0; i < BLOCK_SIZE; i++)
      local_pairs[block_idx + i] = { min_dist[i], best_idx[i] };
  }

  #pragma omp single
  {
    MPI_Allreduce(MPI_IN_PLACE, local_pairs, (int)D_padded, MPI_DOUBLE_INT, MPI_MINLOC, g.row_comm);
  }

  #pragma omp for reduction(|:changed_count)
  for (uint block_idx = 0; block_idx < D_padded; block_idx += BLOCK_SIZE) {
    int valid_docs = std::min((int)BLOCK_SIZE,
                              std::max(0, (int)task_nr_docs - (int)block_idx));

    for (int i = 0; i < valid_docs; i++) {
      uint new_id = (uint)local_pairs[block_idx + i].idx;
      if (assigns[block_idx + i] != new_id) {
        changed_count |= 1u;
        assigns[block_idx + i] = new_id;
      }
    }
  }
}

int main(int argc, char** argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  int id, num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  if (provided < MPI_THREAD_FUNNELED) {
    if (!id) std::cerr << "MPI does not provide required threading level" << std::endl;
    MPI_Finalize();
    return 1;
  }

  if (argc < 2) {
    if (!id) std::cerr << "Usage: " << argv[0] << " IN_FILE" << std::endl;
    MPI_Finalize();
    return 1;
  }

  Grid g;
  try {
    g = make_grid(num_procs, id);
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    MPI_Finalize();
    return 1;
  }

  uint C, D, S;
  uint task_nr_docs, task_first_doc;
  uint task_nr_cents, task_first_cent;

  AlignedPtr<uint> assignments;
  AlignedPtr<double> docs;

  try {
    std::ifstream input(argv[1]);
    if (!input) throw std::runtime_error("Failed to open file '" + std::string(argv[1]));

    double t_parse = -MPI_Wtime();
    parse_input(input, assignments, docs, C, D, S, g,
                task_nr_docs, task_first_doc, task_nr_cents, task_first_cent);
    t_parse += MPI_Wtime();
    if (!id) std::cerr << "Parse time: " << t_parse << "s\n";

  } catch (const std::exception& e) {
    std::cerr << "[rank " << id << "] " << e.what() << std::endl;
    MPI_Finalize();
    return 1;
  }

  uint D_padded       = (task_nr_docs  + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

  // Pad C_local to the next multiple of 4 so reassign_step has no remainder loop.
  // Ghost centroids are filled with a large (but finite) value so they are
  // never selected as nearest cluster, and (max/2)^2 won't overflow in fmadd.
  uint C_padded_local = (task_nr_cents + 3) / 4 * 4;

  // omp_get_max_threads since we are still not in omp parallel
  int number_threads = omp_get_max_threads();

  // Allocate C_padded_local slots; initialise ghost centroids to +inf/2
  AlignedPtr<double> centroids = make_aligned<double>(C_padded_local * S);
  std::fill_n(centroids.get(), C_padded_local * S, 0.0);
  if (C_padded_local > task_nr_cents)
    std::fill_n(centroids.get() + task_nr_cents * S,
                (C_padded_local - task_nr_cents) * S,
                std::numeric_limits<double>::max() / 2.0);

  AlignedPtr<double> all_sums   = make_aligned<double>((size_t)number_threads * (task_nr_cents + 1) * S);
  AlignedPtr<uint>   all_counts = make_aligned<uint>  ((size_t)number_threads * (task_nr_cents + 1));

  uint changed_count = 1;
  bool changed = true;

  // task_nr_cents*S doubles for sums + task_nr_cents doubles for counts + 1 double for the changed_count variable
  AlignedPtr<double> mpi_recv_buf = make_aligned<double>(task_nr_cents * S + task_nr_cents + 1);

  size_t pairs_bytes  = D_padded * sizeof(DistIdx);
  size_t pairs_padded = (pairs_bytes + ALIGN - 1) / ALIGN * ALIGN;
  void* pairs_raw = std::aligned_alloc(ALIGN, pairs_padded);
  if (!pairs_raw) throw std::bad_alloc();
  DistIdx* local_pairs = static_cast<DistIdx*>(pairs_raw);

  const double INF_VAL = std::numeric_limits<double>::max();
  for (uint d = 0; d < D_padded; d++)
    local_pairs[d] = { INF_VAL, (int)C };

  double total_comm_time = 0.0;

  MPI_Barrier(MPI_COMM_WORLD);
  double exec_time = -MPI_Wtime();

  #pragma omp parallel
  { 
    // Assignments are initialized using round-robin based on the global document index, not the local index
    #pragma omp for nowait
    for (uint d = 0; d < task_nr_docs; d++) assignments[d] = (task_first_doc + d) % C;

    #pragma omp single
    for (uint d = task_nr_docs; d < D_padded; d++) assignments[d] = C; // ghost cluster

    update_step(docs.get(), centroids.get(), assignments.get(),
                S, task_nr_docs, task_nr_cents, task_first_cent,
                all_sums.get(), all_counts.get(), number_threads,
                mpi_recv_buf.get(), changed_count, changed, total_comm_time, g);

    while (changed) {
      #pragma omp single
      changed_count = 0;

      reassign_step(docs.get(), centroids.get(), assignments.get(),
                    C_padded_local, task_nr_docs, D_padded, S,
                    task_first_cent, local_pairs, changed_count, g);

      update_step(docs.get(), centroids.get(), assignments.get(),
                  S, task_nr_docs, task_nr_cents, task_first_cent,
                  all_sums.get(), all_counts.get(), number_threads,
                  mpi_recv_buf.get(), changed_count, changed, total_comm_time, g);
    }
  }

  double t_gather = -MPI_Wtime();
  AlignedPtr<uint> all_assigns = gather_results(assignments.get(), task_nr_docs, D, g);
  t_gather += MPI_Wtime();

  exec_time += MPI_Wtime();

  if (!id) {
    std::cerr << std::fixed << std::setprecision(6)
              << "Total time: " << exec_time << "s\n"
              << "Allreduce time: " << total_comm_time << "s\n"
              << "Gather time: " << t_gather << "s\n"
              << "Computation time: " << exec_time - total_comm_time - t_gather << "s\n";
  }

  if (!id) {
    std::cerr << std::fixed << std::setprecision(1) << exec_time << "s\n";
    std::cerr.flush();
    print_result(all_assigns.get(), D);
  }

  // All allocated memory will be cleaned up by RAII
  std::free(pairs_raw);
  MPI_Comm_free(&g.row_comm);
  MPI_Comm_free(&g.col_comm);
  MPI_Comm_free(&g.grid_comm);
  MPI_Finalize();
  return 0;
}