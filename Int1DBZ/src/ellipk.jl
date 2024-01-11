# my complete elliptic integral evaluator via AGM, over complex plane
# Barnett 1/10/24
function AGM(a0,b0)             # compute arithmetic-geometric mean
    a = Complex(a0); b = Complex(b0);   # needed even if ab<0 real
    i::Int = 0;
    while abs(a-b)/abs(a) > 1e-15    # robust; quadratically convergent
        a,b = (a+b)/2, sqrt(a*b)      # in case negative reals
        i += 1
    end
    (a+b)/2 #,i       # iteration count just for info, was never more that 7
end
"""
    ellipkAGM(k) returns complete elliptic integral K(k), for k the modulus,
    real or complex. Note m=k^2 is used as the argument in some definitions.
    For k>1 real, the limit is taken with Im k = 0^+ (above the cut).
"""
ellipkAGM(k) = pi/(2*AGM(sqrt(1-Complex(k)^2),1))  # Gauss 1799 amazing (see Tkachev)
