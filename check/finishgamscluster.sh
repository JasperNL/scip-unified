#!/usr/bin/env bash
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*                  This file is part of the program and library             *
#*         SCIP --- Solving Constraint Integer Programs                      *

#*  Copyright 2002-2022 Zuse Institute Berlin                                *
#*                                                                           *
#*  Licensed under the Apache License, Version 2.0 (the "License");          *
#*  you may not use this file except in compliance with the License.         *
#*  You may obtain a copy of the License at                                  *
#*                                                                           *
#*      http://www.apache.org/licenses/LICENSE-2.0                           *
#*                                                                           *
#*  Unless required by applicable law or agreed to in writing, software      *
#*  distributed under the License is distributed on an "AS IS" BASIS,        *
#*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
#*  See the License for the specific language governing permissions and      *
#*  limitations under the License.                                           *
#*                                                                           *
#*  You should have received a copy of the Apache-2.0 license                *
#*  along with SCIP; see the file LICENSE. If not visit scipopt.org.         *
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

# Cleans up after gams testrun, calls 'evalcheck_gamscluster.sh'.
# To be invoked by 'check_gamscluster.sh'.

# Input environment variables
# GMSDIR needs to be defined, the corresponding directory will be deleted.

# New environment variables defined by this script: None

if test -z "${GMSDIR}"
then
    echo "Error: finishgamscluster.sh called with empty GMSDIR variable."
    exit 0
fi

if test -d "${GMSDIR}"
then
    rm "${GMSDIR}/*"
    rmdir "${GMSDIR}"
fi

if test -z "${EVALFILE}"
then
    echo "Error: finishgamscluster.sh called with empty EVALFILE variable."
    exit 0
fi

./evalcheck_gamscluster.sh "${EVALFILE}"
