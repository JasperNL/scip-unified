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
/*  along with SCIP; see the file COPYING. If not visit scip.zib.de.         */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   struct_nlhdlr.h
 * @brief  structure definitions related to nonlinear handlers of nonlinear constraints
 * @author Ksenia Bestuzheva
 * @author Benjamin Mueller
 * @author Felipe Serrano
 * @author Stefan Vigerske
 */

#ifndef SCIP_STRUCT_NLHLDR_H_
#define SCIP_STRUCT_NLHLDR_H_

#include "scip/type_scip.h"
#include "scip/type_nlhdlr.h"

// MOVE rename to SCIP_Nlhdlr
/** generic data and callback methods of an nonlinear handler */
struct SCIP_ConsExpr_Nlhdlr
{
   char*                         name;             /**< nonlinearity handler name */
   char*                         desc;             /**< nonlinearity handler description (can be NULL) */
   SCIP_CONSEXPR_NLHDLRDATA*     data;             /**< data of handler */
   int                           detectpriority;   /**< detection priority of nonlinearity handler */
   int                           enfopriority;     /**< enforcement priority of nonlinearity handler */
   SCIP_Bool                     enabled;          /**< whether the nonlinear handler should be used */

   SCIP_Longint                  nenfocalls; /**< number of times, the enforcement or estimation callback was called */
   SCIP_Longint                  nintevalcalls; /**< number of times, the interval evaluation callback was called */
   SCIP_Longint                  npropcalls; /**< number of times, the propagation callback was called */
   SCIP_Longint                  nseparated; /**< number of times, the expression handler enforced by separation */
   SCIP_Longint                  ncutoffs;   /**< number of cutoffs found so far by this nonlinear handler */
   SCIP_Longint                  ndomreds;   /**< number of domain reductions found so far by this expression handler */
   SCIP_Longint                  ndetections;/**< number of detect calls in which structure was detected (success returned by detect call) (over all runs) */
   SCIP_Longint                  ndetectionslast;/**< number of detect calls in which structure was detected (success returned by detect call) (in last round) */
   SCIP_Longint                  nbranchscores; /**< number of times, branching scores were added by this nonlinear handler */

   SCIP_CLOCK*                   detecttime; /**< time used for detection */
   SCIP_CLOCK*                   enfotime;   /**< time used for enforcement or estimation */
   SCIP_CLOCK*                   proptime;   /**< time used for reverse propagation */
   SCIP_CLOCK*                   intevaltime;/**< time used for interval evaluation */

   SCIP_DECL_CONSEXPR_NLHDLRFREEHDLRDATA((*freehdlrdata));  /**< callback to free data of handler (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLRFREEEXPRDATA((*freeexprdata));  /**< callback to free expression specific data (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLRCOPYHDLR((*copyhdlr));          /**< callback to copy nonlinear handler (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLRINIT((*init));                  /**< initialization callback (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLREXIT((*exit));                  /**< deinitialization callback (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLRDETECT((*detect));              /**< structure detection callback */
   SCIP_DECL_CONSEXPR_NLHDLREVALAUX((*evalaux));            /**< auxiliary evaluation callback */
   SCIP_DECL_CONSEXPR_NLHDLRINITSEPA((*initsepa));          /**< separation initialization callback (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLRENFO((*enfo));                  /**< enforcement callback (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLRESTIMATE((*estimate));          /**< estimator callback (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLREXITSEPA((*exitsepa));          /**< separation deinitialization callback (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLRINTEVAL((*inteval));            /**< interval evaluation callback (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLRREVERSEPROP((*reverseprop));    /**< reverse propagation callback (can be NULL) */
};


#endif /* SCIP_STRUCT_NLHLDR_H_ */
