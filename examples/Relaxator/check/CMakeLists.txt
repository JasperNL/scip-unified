include(CTest)
#
# define the instance sets
#
# semicolon '\;' is used to split an instance and its optimal objective value
# For infeasible instances, '+infinity' is used (or '-infinity' in case of maximization)
#
set(instances
    "instances/MIP/enigma.mps\;0"
    "instances/MIP/flugpl.mps\;1201500"
    "instances/MIP/gt2.mps\;21166"
    "instances/MIP/lseu.mps\;1120"
    "instances/MINLP/circle.cip\;4.57424778"
    "instances/MINLP/m3.osil\;37.8"
#    "instances/MINLP/parincQuadratic.osil\;49920.5564"
# this instance currently fails
    "instances/MINLP/tltr.mps\;48.0666666667"
)

#
# loop over the instances
#
foreach(instance ${instances})
    list(GET instance 0 path)
    list(GET instance 1 optval)
    get_filename_component(basename ${path} NAME_WE)
        #
        # treat the instance as a tuple (list) of two values
        #
        add_test(NAME examples-relaxator-${basename} COMMAND $<TARGET_FILE:relaxator> -f ${CMAKE_CURRENT_SOURCE_DIR}/../../../check/${path} -o ${optval} ${optval})
endforeach(instance)