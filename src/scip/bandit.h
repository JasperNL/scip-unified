/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2017 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   bandit.h
 * @ingroup INTERNALAPI
 * @brief  internal methods for bandit algorithms
 * @author Gregor Hendel
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_BANDIT_H__
#define __SCIP_BANDIT_H__


#include "scip/def.h"
#include "blockmemshell/memory.h"
#include "scip/type_retcode.h"
#include "scip/type_result.h"
#include "scip/type_set.h"
#include "scip/type_primal.h"
#include "scip/type_bandit.h"
#include "scip/stat.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates a bandit algorithm */
extern
SCIP_RETCODE SCIPbanditCreate(
   SCIP_BANDIT**         bandit,             /**< pointer to bandit algorithm data structure */
   SCIP_BANDITVTABLE*    banditvtable,       /**< virtual table for this bandit algorithm */
   SCIP_SET*             set,                /**< global SCIP settings */
   BMS_BLKMEM*           blkmem,             /**< block memory for parameter settings */
   int                   nactions,           /**< the number of actions for this bandit */
   SCIP_BANDITDATA*      banditdata          /**< algorithm specific bandit data */
   );

/** calls destructor and frees memory of bandit algorithm */
extern
SCIP_RETCODE SCIPbanditFree(
   SCIP_BANDIT**         bandit,             /**< pointer to bandit algorithm data structure */
   SCIP_SET*             set,                /**< global SCIP settings */
   BMS_BLKMEM*           blkmem              /**< block memory */
   );

/** reset the bandit algorithm */
extern
SCIP_RETCODE SCIPbanditReset(
   SCIP_BANDIT*          bandit,             /**< pointer to bandit algorithm data structure */
   SCIP_SET*             set,                /**< global SCIP settings */
   SCIP_Real*            priorities,         /**< priorities for every action, or NULL if not needed */
   unsigned int          seed                /**< initial random seed for bandit selection */
   );

/** select the next action */
extern
SCIP_RETCODE SCIPbanditSelect(
   SCIP_BANDIT*          bandit,             /**< pointer to bandit algorithm data structure */
   SCIP_SET*             set,                /**< global SCIP settings */
   int*                  action              /**< pointer to store the selected action */
   );

/** update the score of the selected action */
extern
SCIP_RETCODE SCIPbanditUpdate(
   SCIP_BANDIT*          bandit,             /**< pointer to bandit algorithm data structure */
   SCIP_SET*             set,                /**< global SCIP settings */
   int                   action,             /**< index of action for which the score should be updated */
   SCIP_Real             score               /**< observed gain of the i'th action */
   );

/** get data of this bandit algorithm */
extern
SCIP_BANDITDATA* SCIPbanditGetData(
   SCIP_BANDIT*          bandit              /**< pointer to bandit algorithm data structure */
   );

/** set the data of this bandit algorithm */
extern
void SCIPbanditSetData(
   SCIP_BANDIT*          bandit,             /**< bandit algorithm data structure */
   SCIP_BANDITDATA*      banditdata          /**< bandit algorihm specific data */
   );

/** create a bandit VTable for bandit algorithm callback functions */
extern
SCIP_RETCODE SCIPbanditvtableCreate(
   SCIP_BANDITVTABLE**   banditvtable,       /**< pointer to virtual table for bandit algorithm */
   const char*           name,               /**< a name for the algorithm represented by this vtable */
   SCIP_SET*             set,                /**< global SCIP settings */
   SCIP_MESSAGEHDLR*     messagehdlr,        /**< message handler */
   BMS_BLKMEM*           blkmem,             /**< block memory for parameter settings */
   SCIP_DECL_BANDITFREE  ((*banditfree)),    /**< callback to free bandit specific data structures */
   SCIP_DECL_BANDITSELECT((*banditselect)),  /**< selection callback for bandit selector */
   SCIP_DECL_BANDITUPDATE((*banditupdate)),  /**< update callback for bandit algorithms */
   SCIP_DECL_BANDITRESET ((*banditreset))    /**< update callback for bandit algorithms */
   );


/** free a bandit vTable for bandit algorithm callback functions */
extern
void SCIPbanditvtableFree(
   SCIP_BANDITVTABLE**   banditvtable        /**< pointer to virtual table for bandit algorithm */
   );

/** return the name of this bandit virtual function table */
extern
const char* SCIPbanditvtableGetName(
   SCIP_BANDITVTABLE*    banditvtable        /**< virtual table for bandit algorithm */
   );


/** return the random number generator of a bandit algorithm */
extern
SCIP_RANDNUMGEN* SCIPbanditGetRandnumgen(
   SCIP_BANDIT*          bandit              /**< bandit algorithm data structure */
   );

/** return number of actions of this bandit algorithm */
extern
int SCIPbanditGetNActions(
   SCIP_BANDIT*          bandit              /**< bandit algorithm data structure */
   );


#ifdef __cplusplus
}
#endif

#endif
