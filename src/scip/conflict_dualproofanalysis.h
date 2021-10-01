/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2021 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not visit scipopt.org.         */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   conflict.h
 * @ingroup INTERNALAPI
 * @brief  internal methods for dual proof conflict analysis
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_CONFLICT_DUALPROOFANALYSIS_H__
#define __SCIP_CONFLICT_DUALPROOFANALYSIS_H__

#include "blockmemshell/memory.h"
#include "scip/def.h"
#include "scip/type_branch.h"
#include "scip/type_conflict.h"
#include "scip/type_conflictstore.h"
#include "scip/type_event.h"
#include "scip/type_implics.h"
#include "scip/type_lp.h"
#include "scip/type_prob.h"
#include "scip/type_reopt.h"
#include "scip/type_retcode.h"
#include "scip/type_set.h"
#include "scip/type_stat.h"
#include "scip/type_tree.h"
#include "scip/type_var.h"
#include "scip/type_cuts.h"


/* because calculations might cancel out some values, we stop the infeasibility analysis if a value is bigger than
 * 2^53 = 9007199254740992
 */
#define NUMSTOP 9007199254740992.0
#define BOUNDSWITCH                0.51 /**< threshold for bound switching - see cuts.c */
#define POSTPROCESS               FALSE /**< apply postprocessing to the cut - see cuts.c */
#define USEVBDS                   FALSE /**< use variable bounds - see cuts.c */
#define ALLOWLOCAL                FALSE /**< allow to generate local cuts - see cuts. */
#define MINFRAC                   0.05  /**< minimal fractionality of floor(rhs) - see cuts.c */
#define MAXFRAC                   0.999 /**< maximal fractionality of floor(rhs) - see cuts.c */

/*
 * Proof Sets
 */


void SCIPproofsetFree(
   SCIP_PROOFSET**       proofset,           /**< proof set */
   BMS_BLKMEM*           blkmem              /**< block memory */
   );

/** returns the number of variables in the proofset */

int SCIPproofsetGetNVars(
   SCIP_PROOFSET*        proofset            /**< proof set */
   );
/** returns the number of variables in the proofset */


/** creates and clears the proofset */

SCIP_RETCODE SCIPconflictInitProofset(
   SCIP_CONFLICT*        conflict,           /**< conflict analysis data */
   BMS_BLKMEM*           blkmem              /**< block memory of transformed problem */
   );


/* create proof constraints out of proof sets */

SCIP_RETCODE SCIPconflictFlushProofset(
   SCIP_CONFLICT*        conflict,           /**< conflict analysis data */
   SCIP_CONFLICTSTORE*   conflictstore,      /**< conflict store */
   BMS_BLKMEM*           blkmem,             /**< block memory */
   SCIP_SET*             set,                /**< global SCIP settings */
   SCIP_STAT*            stat,               /**< dynamic problem statistics */
   SCIP_PROB*            transprob,          /**< transformed problem after presolve */
   SCIP_PROB*            origprob,           /**< original problem */
   SCIP_TREE*            tree,               /**< branch and bound tree */
   SCIP_REOPT*           reopt,              /**< reoptimization data structure */
   SCIP_LP*              lp,                 /**< current LP data */
   SCIP_BRANCHCAND*      branchcand,         /**< branching candidate storage */
   SCIP_EVENTQUEUE*      eventqueue,         /**< event queue */
   SCIP_CLIQUETABLE*     cliquetable         /**< clique table data structure */
   );

#ifdef SCIP_DEBUG
static
void debugPrintViolationInfo(
   SCIP_SET*             set,                /**< global SCIP settings */
   SCIP_Real             minact,             /**< min activity */
   SCIP_Real             rhs,                /**< right hand side */
   const char*           infostr             /**< additional info for this debug message, or NULL */
   );
#else
#define debugPrintViolationInfo(...) /**/
#endif


/** perform conflict analysis based on a dual unbounded ray
 *
 *  given an aggregation of rows lhs <= a^Tx such that lhs > maxactivity. if the constraint has size one we add a
 *  bound change instead of the constraint.
 */
SCIP_RETCODE SCIPconflictAnalyzeDualProof(
   SCIP_CONFLICT*        conflict,           /**< conflict analysis data */
   SCIP_SET*             set,                /**< global SCIP settings */
   SCIP_STAT*            stat,               /**< dynamic SCIP statistics */
   BMS_BLKMEM*           blkmem,             /**< block memory */
   SCIP_PROB*            origprob,           /**< original problem */
   SCIP_PROB*            transprob,          /**< transformed problem */
   SCIP_TREE*            tree,               /**< tree data */
   SCIP_REOPT*           reopt,              /**< reoptimization data */
   SCIP_LP*              lp,                 /**< LP data */
   SCIP_AGGRROW*         proofrow,           /**< aggregated row representing the proof */
   int                   validdepth,         /**< valid depth of the dual proof */
   SCIP_Real*            curvarlbs,          /**< current lower bounds of active problem variables */
   SCIP_Real*            curvarubs,          /**< current upper bounds of active problem variables */
   SCIP_Bool             initialproof,       /**< do we analyze the initial reason of infeasibility? */
   SCIP_Bool*            globalinfeasible,   /**< pointer to store whether global infeasibility could be proven */
   SCIP_Bool*            success             /**< pointer to store success result */
   );

#endif