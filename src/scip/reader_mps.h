/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*  Copyright (c) 2002-2023 Zuse Institute Berlin (ZIB)                      */
/*                                                                           */
/*  Licensed under the Apache License, Version 2.0 (the "License");          */
/*  you may not use this file except in compliance with the License.         */
/*  You may obtain a copy of the License at                                  */
/*                                                                           */
/*      http://www.apache.org/licenses/LICENSE-2.0                           */
/*                                                                           */
/*  Unless required by applicable law or agreed to in writing, software      */
/*  distributed under the License is distributed on an "AS IS" BASIS,        */
/*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/*  See the License for the specific language governing permissions and      */
/*  limitations under the License.                                           */
/*                                                                           */
/*  You should have received a copy of the Apache-2.0 license                */
/*  along with SCIP; see the file LICENSE. If not visit scipopt.org.         */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   reader_mps.h
 * @ingroup FILEREADERS
 * @brief  (extended) MPS file reader
 * @author Thorsten Koch
 * @author Tobias Achterberg
 *
 * This reader allows to parse and write MPS files with linear and quadratic constraints and objective,
 * special ordered sets of type 1 and 2, indicators on linear constraints, and semicontinuous variables.
 * For writing, linear (general and specialized), indicator, quadratic, second order cone, and
 * special ordered set constraints are supported.
 *
 * See http://en.wikipedia.org/wiki/MPS_%28format%29 for a description.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_READER_MPS_H__
#define __SCIP_READER_MPS_H__

#include "scip/def.h"
#include "scip/type_cons.h"
#include "scip/type_prob.h"
#include "scip/type_reader.h"
#include "scip/type_result.h"
#include "scip/type_retcode.h"
#include "scip/type_scip.h"
#include "scip/type_var.h"

#ifdef __cplusplus
extern "C" {
#endif

/** includes the mps file reader into SCIP
 *
 *  @ingroup FileReaderIncludes
 */
SCIP_EXPORT
SCIP_RETCODE SCIPincludeReaderMps(
   SCIP*                 scip                /**< SCIP data structure */
   );

/**@addtogroup FILEREADERS
 *
 * @{
 */

/** reads problem from file */
SCIP_EXPORT
SCIP_RETCODE SCIPreadMps(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_READER*          reader,             /**< the file reader itself */
   const char*           filename,           /**< full path and name of file to read, or NULL if stdin should be used */
   SCIP_RESULT*          result,             /**< pointer to store the result of the file reading call */
   const char***         varnames,           /**< storage for the variable names, or NULL */
   const char***         consnames,          /**< storage for the constraint names, or NULL */
   int*                  varnamessize,       /**< the size of the variable names storage, or NULL */
   int*                  consnamessize,      /**< the size of the constraint names storage, or NULL */
   int*                  nvarnames,          /**< the number of stored variable names, or NULL */
   int*                  nconsnames          /**< the number of stored constraint names, or NULL */
   );

/** writes problem to file */
SCIP_EXPORT
SCIP_RETCODE SCIPwriteMps(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_READER*          reader,             /**< the file reader itself */
   FILE*                 file,               /**< output file, or NULL if standard output should be used */
   const char*           name,               /**< problem name */
   SCIP_Bool             transformed,        /**< TRUE iff problem is the transformed problem */
   SCIP_OBJSENSE         objsense,           /**< objective sense */
   SCIP_Real             objscale,           /**< scalar applied to objective function; external objective value is
                                              * extobj = objsense * objscale * (intobj + objoffset) */
   SCIP_Real             objoffset,          /**< objective offset from bound shifting and fixing */
   SCIP_VAR**            vars,               /**< array with active variables ordered binary, integer, implicit, continuous */
   int                   nvars,              /**< number of active variables in the problem */
   int                   nbinvars,           /**< number of binary variables */
   int                   nintvars,           /**< number of general integer variables */
   int                   nimplvars,          /**< number of implicit integer variables */
   int                   ncontvars,          /**< number of continuous variables */
   SCIP_VAR**            fixedvars,          /**< array with fixed and aggregated variables */
   int                   nfixedvars,         /**< number of fixed and aggregated variables in the problem */
   SCIP_CONS**           conss,              /**< array with constraints of the problem */
   int                   nconss,             /**< number of constraints in the problem */
   SCIP_RESULT*          result              /**< pointer to store the result of the file writing call */
   );

/** @} */

#ifdef __cplusplus
}
#endif

#endif
