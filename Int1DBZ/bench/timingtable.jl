using Int1DBZ
using Printf
using OffsetArrays
using StaticArrays
using LinearAlgebra
BLAS.set_num_threads(1)
using Random.Random
using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds=0.1

η=1e-5; ω=0.5; tol=1e-6;
mtail = 1e-2;              # how small exp decay of coeffs gets to by m=M

tex=true   # false for human-readable

if tex       # note escaping of \ and $.  [r"..." fails with @printf macro]
    @printf "\\begin{tabular}{r|r|r|rrr|rrr|rr|}\n"
    @printf "&&& \\multicolumn{3}{c}{standard GK} & \\multicolumn{3}{c}{pole-sub. GK} & \\multicolumn{2}{c}{ratios}\\\\ \n"
    @printf "\$M\$ & \$n\$ & \$t_\\tbox{eval}\$ (\$\\mu\$s) & \$n_\\tbox{evals}\$ & \$t\$ (ms) & \$t_\\tbox{over}\$ (\$\\mu\$s) & \$n_\\tbox{evals}\$ & \$t\$ (ms) & \$t_\\tbox{over}\$ (\$\\mu\$s) & evals & time\\\\ \n"
    @printf "\\hline\n"
else
    @printf "--------------------------------------------------------------------\n"
    @printf "M\tn\tt_eval\tplain GK  \t\tpole-sub   \t\tratios (improvement factors)\n"
    @printf "\t\t\t#evals\tt(ms)\t#evals\tover(us)\tt(ms)\t#evals\tover(us)\ttime\n"
    @printf "--------------------------------------------------------------------\n"
end

function f(x::Number, Hm::AbstractArray, ω, η)        # integrand func
    return tr(inv(complex(ω,η)*I - fourier_kernel(Hm,x)))
end

for M = [10 100]
    for n = [1 2 4 8]
        Random.seed!(0)
        mlist = -M:M  # matrix, OV of SA's version, some painful iterators here
        decayrate = log(1/mtail)/M
        am = OffsetVector(exp.(-decayrate*abs.(mlist)), mlist)  # rand w/ decay
        Hm = OffsetVector([SMatrix{n,n}(am[m] * randn(ComplexF64,(n,n))) for m in mlist], mlist)
        Hmconj = OffsetVector([Hm[m]' for m in mlist], mlist)
        Hm = (Hm + reverse(Hmconj))/2                     # H(x) hermitian if x Re
        tobj = @benchmark f(x,$Hm,$ω,$η) setup=( x=2π*rand() )  # integrand
        tf = median(tobj.times)/1e3      # convert ns to us
        Al = realadap_lxvm(Hm,ω,η,tol=tol)    # ground-truth
        Am, Em, segsm, nem = realmyadap(Hm,ω,η,tol=tol)   # b'mark no tuple out
        tobj = @benchmark realmyadap($Hm,ω,η,tol=tol)
        tm = median(tobj.times)/1e6      # convert ns to ms
        nsegs = nem/15                   # since (7,15) GK
        tps = (tm*1e3 - nem*tf)/nsegs    # us overhead per seg used
        Ap, Ep, segsp, nep = realquadinv(Hm,ω,η,tol=tol, rootmeth="F", verb = tex ? 0 : 1)
        tobj = @benchmark realquadinv($Hm,ω,η,tol=tol, rootmeth="F")
        tp = median(tobj.times)/1e6
        nsegsp = nep/15
        tpsp = (tp*1e3 - nep*tf)/nsegsp  # pole-sub us overhead per seg used
        if tex
            @printf "%d & %d & %.2g & %d & %.2g & %.2g & %d & %.2g & %.2g & %.2g & %.2g\\\\ \n" M n tf nem tm tps nep tp tpsp nem/nep tm/tp
        else
            @printf "%d\t%d\t%.2g\t%d\t%.2g\t%d\t%.2g\t%.2g\t%.2g\n" M n tf nem tm tps nep tp tpsp nem/nep tm/tp
        end        
    end
    if tex; @printf "\\hline\n"; end
end
if tex
    @printf "\\end{tabular}\n"
else
    @printf "--------------------------------------------------------------------\n"
end

# concl: "F" faster unless n>=8 matrix size
