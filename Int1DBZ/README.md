
development tests are test/dev.jl

For allocation meas use:
remove Revise from startup.jl
julia -t1 --track-allocation=user --project=.
Examine *.jl.*.mem
Clear up with:
find . -name "*.mem" -delete
