for i in MIPSolver SamplePricer SamplePricer_C TSP VRP
do
    echo
    echo ===== $i =====
    echo
    cd $i
    make OPT=dbg depend
    make OPT=opt depend
    make OPT=dbg
    make OPT=opt
    cd -
done
