# Benchmarks
The naming for the benchmarks is a little confusing. Right now, they are marked with the benchmarks_ tag in their filename. The first one written, "benchmarks.jl" is an exception.

## Benchmarks.jl
This was written to query some timings specific to the optimization problem, i.e., Linear Algebra solve time, VecchiaCache creation time, and solving model time. 

## Benchmarks_pprof.jl
This was written to generate the flame graph for VecchiaMLE using PProf, hence the name. 

## Benchmarks_data.jl
This was written to see how big of a problem we could solve. It reports the time and memory use for VecchiaMLE for increasing dimensions, stopping when we hit a large time (25s). 
