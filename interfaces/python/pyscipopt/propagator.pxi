cdef SCIP_RETCODE PyPropCopy (SCIP* scip, SCIP_PROP* prop):
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropFree (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.propfree()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropInit (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.propinit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropExit (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.propexit()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropInitpre (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.propinitpre()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropExitpre (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.propexitpre()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropInitsol (SCIP* scip, SCIP_PROP* prop):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.propinitsol()
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropExitsol (SCIP* scip, SCIP_PROP* prop, SCIP_Bool restart):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    PyProp.propexitsol(restart)
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropPresol (SCIP* scip, SCIP_PROP* prop, int nrounds, SCIP_PRESOLTIMING presoltiming,
                                int nnewfixedvars, int nnewaggrvars, int nnewchgvartypes, int nnewchgbds, int nnewholes,
                                int nnewdelconss, int nnewaddconss, int nnewupgdconss, int nnewchgcoefs, int nnewchgsides,
                                int* nfixedvars, int* naggrvars, int* nchgvartypes, int* nchgbds, int* naddholes,
                                int* ndelconss, int* naddconss, int* nupgdconss, int* nchgcoefs, int* nchgsides, SCIP_RESULT* result):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    # dictionary for input/output parameters
    result_dict = {}
    result_dict["nfixedvars"]   = nfixedvars[0]
    result_dict["naggrvars"]    = naggrvars[0]
    result_dict["nchgvartypes"] = nchgvartypes[0]
    result_dict["nchgbds"]      = nchgbds[0]
    result_dict["naddholes"]    = naddholes[0]
    result_dict["ndelconss"]    = ndelconss[0]
    result_dict["naddconss"]    = naddconss[0]
    result_dict["nupgdconss"]   = nupgdconss[0]
    result_dict["nchgcoefs"]    = nchgcoefs[0]
    result_dict["nchgsides"]    = nchgsides[0]
    result_dict["result"]       = result[0]
    PyProp.proppresol(nrounds, presoltiming,
                      nnewfixedvars, nnewaggrvars, nnewchgvartypes, nnewchgbds, nnewholes,
                      nnewdelconss, nnewaddconss, nnewupgdconss, nnewchgcoefs, nnewchgsides, result_dict)
    result[0]       = result_dict["result"]
    nfixedvars[0]   = result_dict["nfixedvars"]
    naggrvars[0]    = result_dict["naggrvars"]
    nchgvartypes[0] = result_dict["nchgvartypes"]
    nchgbds[0]      = result_dict["nchgbds"]
    naddholes[0]    = result_dict["naddholes"]
    ndelconss[0]    = result_dict["ndelconss"]
    naddconss[0]    = result_dict["naddconss"]
    nupgdconss[0]   = result_dict["nupgdconss"]
    nchgcoefs[0]    = result_dict["nchgcoefs"]
    nchgsides[0]    = result_dict["nchgsides"]
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropExec (SCIP* scip, SCIP_PROP* prop, SCIP_PROPTIMING proptiming, SCIP_RESULT* result):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    PyProp = <Prop>propdata
    returnvalues = PyProp.propexec(proptiming)
    result_dict = returnvalues
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef SCIP_RETCODE PyPropResProp (SCIP* scip, SCIP_PROP* prop, SCIP_VAR* infervar, int inferinfo,
                                 SCIP_BOUNDTYPE boundtype, SCIP_BDCHGIDX* bdchgidx, SCIP_Real relaxedbd, SCIP_RESULT* result):
    cdef SCIP_PROPDATA* propdata
    propdata = SCIPpropGetData(prop)
    confvar = pythonizeVar(infervar, "conflictvar")

#TODO: parse bdchgidx?

    PyProp = <Prop>propdata
    returnvalues = PyProp.propresprop(confvar, inferinfo, boundtype, relaxedbd)
    result_dict = returnvalues
    result[0] = result_dict.get("result", <SCIP_RESULT>result[0])
    return SCIP_OKAY

cdef class Prop:
    cdef public object data     # storage for the python user
    cdef public Model model

    def propfree(self):
        pass

    def propinit(self):
        pass

    def propexit(self):
        pass

    def propinitsol(self):
        pass

    def propexitsol(self, restart):
        pass

    def propinitpre(self):
        pass

    def propexitpre(self):
        pass

    def proppresol(self, nrounds, presoltiming, result_dict):
        pass

    def propexec(self, proptiming):
        print("python error in propexec: this method needs to be implemented")
        return {}

    def propresprop(self, confvar, inferinfo, bdtype, relaxedbd):
        print("python error in propresprop: this method needs to be implemented")
        return {}
