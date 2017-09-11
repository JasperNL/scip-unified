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

/**@file   bandit_ucb.h
 * @ingroup BanditAlgorithms
 * @brief  public methods for UCB bandit algorithm
 * @author Gregor Hendel
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_BANDIT_UCB_H__
#define __SCIP_BANDIT_UCB_H__


#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/**@addtogroup PublicBanditMethods
 *
 * UCB (Upper confidence bounds) is a deterministic algorithm
 * for the multi-armed bandit problem.
 *
 * In every iteration, UCB selects the action that maximizes
 * a tradeoff between its performance in the past
 * and a variance term.
 *
 * The influence of the variance (confidence width) can be
 * controlled by the parameter \f$ \alpha f$.
 *
 * @{
 */

/** include virtual function table for UCB bandit algorithms */
EXTERN
SCIP_RETCODE SCIPincludeBanditvtableUcb(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** create UCB bandit algorithm */
EXTERN
SCIP_RETCODE SCIPcreateBanditUcb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BANDIT**         ucb,                /**< pointer to store bandit algorithm */
   int                   nactions,           /**< the number of actions for this bandit algorithm */
   SCIP_Real             alpha               /**< parameter to increase confidence width */
   );

/** returns the upper confidence bound of a selected action */
EXTERN
SCIP_Real SCIPgetConfidenceBoundUcb(
   SCIP_BANDIT*          ucb,                /**< UCB bandit algorithm */
   int                   action              /**< index of the queried action */
   );

/** return start permutation of the UCB bandit algorithm */
EXTERN
int* SCIPgetStartPermutationUcb(
   SCIP_BANDIT*          ucb                 /**< UCB bandit algorithm */
   );

/** @}*/

/*
 * Callback methods of bandit algorithm
 */

/** callback to free bandit specific data structures */
extern
SCIP_DECL_BANDITFREE(banditFreeUcb);

/** selection callback for bandit selector */
extern
SCIP_DECL_BANDITSELECT(banditSelectUcb);

/** update callback for bandit algorithm */
extern
SCIP_DECL_BANDITUPDATE(banditUpdateUcb);

/** reset callback for bandit algorithm */
extern
SCIP_DECL_BANDITRESET(banditResetUcb);

/** internal method to create UCB bandit algorithm */
extern
SCIP_RETCODE SCIPbanditCreateUcb(
   BMS_BLKMEM*           blkmem,             /**< block memory */
   BMS_BUFMEM*           bufmem,             /**< buffer memory */
   SCIP_BANDITVTABLE*    vtable,             /**< virtual function table for UCB bandit algorithm */
   SCIP_BANDIT**         ucb,                /**< pointer to store bandit algorithm */
   int                   nactions,           /**< the number of actions for this bandit algorithm */
   SCIP_Real             alpha,              /**< parameter to increase confidence width */
   unsigned int          initseed            /**< initial random seed */
   );

#ifdef __cplusplus
}

#endif

#endif
