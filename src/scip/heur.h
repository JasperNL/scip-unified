/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2006 Tobias Achterberg                              */
/*                                                                           */
/*                  2002-2006 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: heur.h,v 1.45 2006/01/03 12:22:46 bzfpfend Exp $"

/**@file   heur.h
 * @brief  internal methods for primal heuristics
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_HEUR_H__
#define __SCIP_HEUR_H__


#include "scip/def.h"
#include "blockmemshell/memory.h"
#include "scip/type_retcode.h"
#include "scip/type_result.h"
#include "scip/type_set.h"
#include "scip/type_primal.h"
#include "scip/type_heur.h"
#include "scip/pub_heur.h"



/** creates a primal heuristic */
extern
SCIP_RETCODE SCIPheurCreate(
   SCIP_HEUR**           heur,               /**< pointer to primal heuristic data structure */
   SCIP_SET*             set,                /**< global SCIP settings */
   BMS_BLKMEM*           blkmem,             /**< block memory for parameter settings */
   const char*           name,               /**< name of primal heuristic */
   const char*           desc,               /**< description of primal heuristic */
   char                  dispchar,           /**< display character of primal heuristic */
   int                   priority,           /**< priority of the primal heuristic */
   int                   freq,               /**< frequency for calling primal heuristic */
   int                   freqofs,            /**< frequency offset for calling primal heuristic */
   int                   maxdepth,           /**< maximal depth level to call heuristic at (-1: no limit) */
   SCIP_Bool             pseudonodes,        /**< call heuristic at nodes where only a pseudo solution exist? */
   SCIP_Bool             duringplunging,     /**< call heuristic during plunging? */
   SCIP_Bool             duringlploop,       /**< call heuristic during the LP price-and-cut loop? */
   SCIP_Bool             afternode,          /**< call heuristic after or before the current node was solved? */
   SCIP_DECL_HEURFREE    ((*heurfree)),      /**< destructor of primal heuristic */
   SCIP_DECL_HEURINIT    ((*heurinit)),      /**< initialize primal heuristic */
   SCIP_DECL_HEUREXIT    ((*heurexit)),      /**< deinitialize primal heuristic */
   SCIP_DECL_HEURINITSOL ((*heurinitsol)),   /**< solving process initialization method of primal heuristic */
   SCIP_DECL_HEUREXITSOL ((*heurexitsol)),   /**< solving process deinitialization method of primal heuristic */
   SCIP_DECL_HEUREXEC    ((*heurexec)),      /**< execution method of primal heuristic */
   SCIP_HEURDATA*        heurdata            /**< primal heuristic data */
   );

/** calls destructor and frees memory of primal heuristic */
extern
SCIP_RETCODE SCIPheurFree(
   SCIP_HEUR**           heur,               /**< pointer to primal heuristic data structure */
   SCIP_SET*             set                 /**< global SCIP settings */
   );

/** initializes primal heuristic */
extern
SCIP_RETCODE SCIPheurInit(
   SCIP_HEUR*            heur,               /**< primal heuristic */
   SCIP_SET*             set                 /**< global SCIP settings */
   );

/** calls exit method of primal heuristic */
extern
SCIP_RETCODE SCIPheurExit(
   SCIP_HEUR*            heur,               /**< primal heuristic */
   SCIP_SET*             set                 /**< global SCIP settings */
   );

/** informs primal heuristic that the branch and bound process is being started */
extern
SCIP_RETCODE SCIPheurInitsol(
   SCIP_HEUR*            heur,               /**< primal heuristic */
   SCIP_SET*             set                 /**< global SCIP settings */
   );

/** informs primal heuristic that the branch and bound process data is being freed */
extern
SCIP_RETCODE SCIPheurExitsol(
   SCIP_HEUR*            heur,               /**< primal heuristic */
   SCIP_SET*             set                 /**< global SCIP settings */
   );

/** calls execution method of primal heuristic */
extern
SCIP_RETCODE SCIPheurExec(
   SCIP_HEUR*            heur,               /**< primal heuristic */
   SCIP_SET*             set,                /**< global SCIP settings */
   SCIP_PRIMAL*          primal,             /**< primal data */
   int                   depth,              /**< depth of current node */
   int                   lpforkdepth,        /**< depth of the last node with solved LP */
   SCIP_Bool             currentnodehaslp,   /**< is LP being processed in the current node? */
   SCIP_Bool             plunging,           /**< is the next node to be processed a child or sibling? */
   SCIP_Bool             nodesolved,         /**< is the current node already solved? */
   SCIP_Bool             inlploop,           /**< are we currently in the LP solving loop? */
   int*                  ndelayedheurs,      /**< pointer to count the number of delayed heuristics */
   SCIP_RESULT*          result              /**< pointer to store the result of the callback method */
   );

/** sets priority of primal heuristic */
extern
void SCIPheurSetPriority(
   SCIP_HEUR*            heur,               /**< primal heuristic */
   SCIP_SET*             set,                /**< global SCIP settings */
   int                   priority            /**< new priority of the primal heuristic */
   );


#endif
