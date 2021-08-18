/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2020 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not visit scipopt.org.         */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   pub_nlpi.h
 * @ingroup PUBLICCOREAPI
 * @brief  public methods for NLP solver interfaces
 * @author Thorsten Gellermann
 * @author Stefan Vigerske
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_PUB_NLPI_H__
#define __SCIP_PUB_NLPI_H__

#include "scip/def.h"
#include "scip/type_nlpi.h"
#include "scip/type_misc.h"

#ifdef __cplusplus
extern "C" {
#endif

/**@addtogroup PublicNLPIInterfaceMethods
 *
 * @{
 */

/** compares two NLPIs w.r.t. their priority */
SCIP_DECL_SORTPTRCOMP(SCIPnlpiComp);

/** gets data of an NLPI */
SCIP_EXPORT
SCIP_NLPIDATA* SCIPnlpiGetData(
   SCIP_NLPI*            nlpi                /**< NLP interface structure */
   );

/** gets NLP solver name */
SCIP_EXPORT
const char* SCIPnlpiGetName(
   SCIP_NLPI*            nlpi                /**< NLP interface structure */
   );

/** gets NLP solver descriptions */
SCIP_EXPORT
const char* SCIPnlpiGetDesc(
   SCIP_NLPI*            nlpi                /**< NLP interface structure */
   );

/** gets NLP solver priority */
SCIP_EXPORT
int SCIPnlpiGetPriority(
   SCIP_NLPI*            nlpi                /**< NLP interface structure */
   );

/**@name Statistics */
/**@{ */

/** gives number of problems created for NLP solver so far */
SCIP_EXPORT
int SCIPnlpiGetNProblems(
   SCIP_NLPI*            nlpi                /**< NLP interface structure */
   );

/** gives total time spend in problem creation/modification/freeing */
SCIP_EXPORT
SCIP_Real SCIPnlpiGetProblemTime(
   SCIP_NLPI*            nlpi                /**< NLP interface structure */
   );

/** total number of NLP solves so far */
SCIP_EXPORT
int SCIPnlpiGetNSolves(
   SCIP_NLPI*            nlpi                /**< NLP interface structure */
   );

/** gives total time spend in NLP solves (as reported by solver) */
SCIP_EXPORT
SCIP_Real SCIPnlpiGetSolveTime(
   SCIP_NLPI*            nlpi                /**< NLP interface structure */
   );

/** gives total time spend in function evaluation during NLP solves
 *
 * If timing/nlpieval is off (the default), depending on the NLP solver, this may just return 0.
 */
SCIP_EXPORT
SCIP_Real SCIPnlpiGetEvalTime(
   SCIP_NLPI*            nlpi                /**< NLP interface structure */
   );

/** gives total number of iterations spend by NLP solver so far */
SCIP_EXPORT
SCIP_Longint SCIPnlpiGetNIterations(
   SCIP_NLPI*            nlpi                /**< NLP interface structure */
   );

/** gives number of times a solve ended with a specific termination status */
SCIP_EXPORT
int SCIPnlpiGetNTermStat(
   SCIP_NLPI*            nlpi,               /**< NLP interface structure */
   SCIP_NLPTERMSTAT      termstatus          /**< the termination status to query for */
   );

/** gives number of times a solve ended with a specific solution status */
SCIP_EXPORT
int SCIPnlpiGetNSolStat(
   SCIP_NLPI*            nlpi,               /**< NLP interface structure */
   SCIP_NLPSOLSTAT       solstatus           /**< the solution status to query for */
   );

/** adds statistics from one NLPI to another */
SCIP_EXPORT
void SCIPnlpiMergeStatistics(
   SCIP_NLPI*            targetnlpi,         /**< NLP interface where to add statistics */
   SCIP_NLPI*            sourcenlpi          /**< NLP interface from which add statistics */
   );

/**@} */ /* Statistics */

/**@} */ /* PublicNLPIMethods */

#ifdef __cplusplus
}
#endif

#endif /* __SCIP_PUB_NLPI_H__ */
