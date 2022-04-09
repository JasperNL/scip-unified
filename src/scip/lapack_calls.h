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

/**@file   lapack_calls.h
 * @brief  interface methods for lapack functions
 * @author Marc Pfetsch
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_LAPACK_CALLS_H__
#define __SCIP_LAPACK_CALLS_H__

#include "scip/def.h"
#include "blockmemshell/memory.h"
#include "scip/type_retcode.h"

#ifdef __cplusplus
extern "C" {
#endif

/** returns whether Lapack s available, i.e., whether it has been linked in */
SCIP_EXPORT
SCIP_Bool SCIPlapackIsAvailable(void);

/** computes eigenvalues and eigenvectors of a dense symmetric matrix
 *
 *  Calls Lapack's DSYEV function.
 */
SCIP_EXPORT
SCIP_RETCODE SCIPlapackComputeEigenvalues(
   BMS_BUFMEM*           bufmem,             /**< buffer memory (or NULL if IPOPT is used) */
   SCIP_Bool             geteigenvectors,    /**< should also eigenvectors should be computed? */
   int                   N,                  /**< dimension */
   SCIP_Real*            a,                  /**< matrix data on input (size N*N); eigenvectors on output if geteigenvectors == TRUE */
   SCIP_Real*            w                   /**< array to store eigenvalues (size N) (or NULL) */
   );

/** solves a linear problem of the form Ax = b for a regular matrix A
 *
 *  Calls Lapack's DGETRF routine to calculate a LU factorization and uses this factorization to solve
 *  the linear problem Ax = b.
 */
SCIP_EXPORT
SCIP_RETCODE SCIPlapackSolveLinearEquations(
   BMS_BUFMEM*           bufmem,             /**< buffer memory (or NULL if IPOPT is used) */
   int                   N,                  /**< dimension */
   SCIP_Real*            A,                  /**< matrix data on input (size N*N); filled column-wise */
   SCIP_Real*            b,                  /**< right hand side vector (size N) */
   SCIP_Real*            x,                  /**< buffer to store solution (size N) */
   SCIP_Bool*            success             /**< pointer to store if the solving routine was successful */
   );

#ifdef __cplusplus
}
#endif

#endif
