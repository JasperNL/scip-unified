/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2022 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not visit scipopt.org.         */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   reader_fix.h
 * @ingroup FILEREADERS
 * @brief  file reader for variable fixings
 * @author Tobias Achterberg
 *
 * This reader allows to read a file containing fixation values for variables of the current problem. Each line of the
 * file should have format
 *
 *    \<variable name\> \<value to fix\>
 *
 * Note that only a subset of the variables may need to appear in the file. Lines with unknown variable names are
 * ignored. The writing functionality is currently not supported.
 *
 * @note The format is equal to the (not xml) solution format of SCIP.
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_READER_FIX_H__
#define __SCIP_READER_FIX_H__

#include "scip/def.h"
#include "scip/type_retcode.h"
#include "scip/type_scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** includes the fix file reader into SCIP
 *
 *  @ingroup FileReaderIncludes
 */
SCIP_EXPORT
SCIP_RETCODE SCIPincludeReaderFix(
   SCIP*                 scip                /**< SCIP data structure */
   );

#ifdef __cplusplus
}
#endif

#endif
