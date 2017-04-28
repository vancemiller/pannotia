#ifndef TIMING_H
#define TIMING_H

#define TIMESTAMP(NAME) \
  cudaEvent_t NAME; \
  struct timespec NAME ## _cpu;\
  if (clock_gettime(CLOCK_MONOTONIC, &NAME ## _cpu)) { \
    fprintf(stderr, "Failed to get time: %s\n", strerror(errno)); \
  } \
  cudaEventCreateWithFlags(&NAME, cudaEventBlockingSync); \
  cudaEventRecord(NAME);

#define ELAPSED(start, end) \
  ({float ms; \
  ms = cudaElapsed(start, end); \
  if (ms < 0.000001) { \
   struct timespec cstart = start ## _cpu; \
   struct timespec cend = end ## _cpu; \
    ms = cpuElapsed(cstart, cend); \
  } \
  ms;})

inline float cpuElapsed(struct timespec start, struct timespec end) {
  return (1e9 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) * 1e-6;
}

inline float cudaElapsed(cudaEvent_t start, cudaEvent_t end) {
  float ms;
  cudaEventSynchronize(start); // synchronize on the event so we can record elapsed time
  cudaEventSynchronize(end); // synchronize on the event so we can record elapsed time
  cudaEventElapsedTime(&ms, start, end);
  return ms;
}

#endif
