# Results

## Environment
- All tests were ran in `lab2p3`
- `taskset -c 0` was used to pin program to core 0

## Implementations
### `docs_auto.cpp`

```
[nix-shell:~/.private]$ taskset -c 0 perf stat -ddd ./docs_auto tests/ex1000-50000-200.in 1> /dev/null
30.8975s

 Performance counter stats for './docs_auto tests/ex1000-50000-200.in':

          32631,48 msec task-clock                       #    1,000 CPUs utilized
               276      context-switches                 #    8,458 /sec
                 0      cpu-migrations                   #    0,000 /sec
             20531      page-faults                      #  629,178 /sec
       87156527895      cycles                           #    2,671 GHz                         (33,33%)
      295279678674      instructions                     #    3,39  insn per cycle              (39,99%)
       22140995179      branches                         #  678,516 M/sec                       (46,66%)
          34418760      branch-misses                    #    0,16% of all branches             (53,33%)
                        TopdownL1                 #      0,5 %  tma_backend_bound
                                                  #     74,9 %  tma_bad_speculation
                                                  #      4,3 %  tma_frontend_bound
                                                  #     20,4 %  tma_retiring             (60,00%)
       92503184652      L1-dcache-loads                  #    2,835 G/sec                       (66,67%)
        8045883106      L1-dcache-load-misses            #    8,70% of all L1-dcache accesses   (66,67%)
         273550511      LLC-loads                        #    8,383 M/sec                       (66,67%)
           3134592      LLC-load-misses                  #    1,15% of all LL-cache accesses    (66,67%)
   <not supported>      L1-icache-loads
          13388711      L1-icache-load-misses                                                   (26,67%)
       92672735063      dTLB-loads                       #    2,840 G/sec                       (26,67%)
             24733      dTLB-load-misses                 #    0,00% of all dTLB cache accesses  (26,66%)
   <not supported>      iTLB-loads
             18577      iTLB-load-misses                                                        (26,66%)
   <not supported>      L1-dcache-prefetches
   <not supported>      L1-dcache-prefetch-misses

      32,638061766 seconds time elapsed

      32,298603000 seconds user
       0,042569000 seconds sys
```

### `docs_avx2.cpp`

```
[nix-shell:~/.private]$ taskset -c 0 perf stat -ddd ./docs_avx2 tests/ex1000-50000-200.in 1> /dev/null
23.1895s

 Performance counter stats for './docs_avx2 tests/ex1000-50000-200.in':

          24927,10 msec task-clock                       #    1,000 CPUs utilized
               205      context-switches                 #    8,224 /sec
                 0      cpu-migrations                   #    0,000 /sec
             20526      page-faults                      #  823,441 /sec
       66599665594      cycles                           #    2,672 GHz                         (33,33%)
      252210446514      instructions                     #    3,79  insn per cycle              (40,00%)
       10740851538      branches                         #  430,890 M/sec                       (46,67%)
          44743114      branch-misses                    #    0,42% of all branches             (53,34%)
                        TopdownL1                 #      0,5 %  tma_backend_bound
                                                  #     70,0 %  tma_bad_speculation
                                                  #      7,2 %  tma_frontend_bound
                                                  #     22,2 %  tma_retiring             (60,00%)
       48020345991      L1-dcache-loads                  #    1,926 G/sec                       (66,67%)
        3688329212      L1-dcache-load-misses            #    7,68% of all L1-dcache accesses   (66,67%)
         122341119      LLC-loads                        #    4,908 M/sec                       (66,67%)
           1171732      LLC-load-misses                  #    0,96% of all LL-cache accesses    (66,67%)
   <not supported>      L1-icache-loads
          10497381      L1-icache-load-misses                                                   (26,67%)
       48109071899      dTLB-loads                       #    1,930 G/sec                       (26,67%)
             23726      dTLB-load-misses                 #    0,00% of all dTLB cache accesses  (26,67%)
   <not supported>      iTLB-loads
             11820      iTLB-load-misses                                                        (26,67%)
   <not supported>      L1-dcache-prefetches
   <not supported>      L1-dcache-prefetch-misses

      24,929119000 seconds time elapsed

      24,666264000 seconds user
       0,041523000 seconds sys
```