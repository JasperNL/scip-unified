/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2016 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   struct_cons_expr.h
 * @brief  (public) data structures of expression constraints
 * @author Stefan Vigerske
 * @author Benjamin Mueller
 *
 * These are in particular data structures to manage the expressions in cons_expr.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_STRUCT_CONS_EXPR_H__
#define __SCIP_STRUCT_CONS_EXPR_H__

#include "scip/type_cons_expr.h"
#include "scip/type_misc.h"

#ifdef __cplusplus
extern "C" {
#endif

/** generic data and callback methods of an expression handler */
struct SCIP_ConsExpr_ExprHdlr
{
   char*                   name;          /**< expression handler name */
   char*                   desc;          /**< expression handler description (can be NULL) */
   SCIP_CONSEXPR_EXPRHDLRDATA*   data;    /**< data of handler */
   unsigned int            precedence;    /**< precedence of expression operation relative to other expression (used for printing) */

   SCIP_Longint            nsepacalls;    /**< number of times, the separation or estimation callbacks were called */
   SCIP_Longint            npropcalls;    /**< number of times, the propagation callback was called */
   SCIP_Longint            ncutsfound;    /**< number of cuts added by this expression handler */
   SCIP_Longint            ncutoffs;      /**< number of cutoffs found so far by this expression handler */
   SCIP_Longint            ndomreds;      /**< number of domain reductions found so far by this expression handler */
   SCIP_Longint            nsimplifycalls; /**< number of times, the simplification callback was called */
   SCIP_Longint            nbranchscores; /**< number of times, branching scores were added by (or for) this expression handler */

   SCIP_CLOCK*             sepatime;      /**< time used for separation or estimation */
   SCIP_CLOCK*             proptime;      /**< time used for propagation */
   SCIP_CLOCK*             intevaltime;   /**< time used for interval evaluation */
   SCIP_CLOCK*             simplifytime;  /**< time used for expression simplification */

   SCIP_DECL_CONSEXPR_EXPRCOPYHDLR((*copyhdlr));  /**< handler copy callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRFREEHDLR((*freehdlr));  /**< handler free callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRCOPYDATA((*copydata));  /**< data copy callback, or NULL for expressions that have no data */
   SCIP_DECL_CONSEXPR_EXPRFREEDATA((*freedata));  /**< data free callback, or NULL for expressions that have no data or which data does not need to be freed */
   SCIP_DECL_CONSEXPR_EXPRSIMPLIFY((*simplify));  /**< simplify callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRCMP((*compare));        /**< compare callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRPRINT((*print));        /**< print callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRPARSE((*parse));        /**< parse callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPREVAL((*eval));          /**< point evaluation callback (can never be NULL) */
   SCIP_DECL_CONSEXPR_EXPRBWDIFF((*bwdiff));      /**< derivative evaluation callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRINTEVAL((*inteval));    /**< interval evaluation callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRINITSEPA((*initsepa));  /**< separation initialization callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPREXITSEPA((*exitsepa));  /**< separation deinitialization callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRSEPA((*sepa));          /**< separation callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRESTIMATE((*estimate));  /**< estimation callback (can be NULL) */
   SCIP_DECL_CONSEXPR_REVERSEPROP((*reverseprop)); /**< reverse propagation callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRHASH((*hash));          /**< hash callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRBRANCHSCORE((*brscore)); /**< branching score callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRCURVATURE((*curvature)); /**< curvature detection callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRMONOTONICITY((*monotonicity)); /**< monotonicity detection callback (can be NULL) */
   SCIP_DECL_CONSEXPR_EXPRINTEGRALITY((*integrality)); /**< integrality detection callback (can be NULL) */
};

/** a node in the expression graph that is handled by the expression constraint handler */
struct SCIP_ConsExpr_Expr
{
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr;      /**< expression type (as pointer to its handler) */
   SCIP_CONSEXPR_EXPRDATA* exprdata;      /**< expression data */

   int                     nchildren;     /**< number of children */
   int                     childrensize;  /**< length of children array */
   SCIP_CONSEXPR_EXPR**    children;      /**< children expressions */

   int                     nuses;         /**< reference counter */
   int                     nlockspos;     /**< positive locks counter */
   int                     nlocksneg ;    /**< negative locks counter */

   /* enforcement of expr == auxvar (or expr <= auxvar, or expr >= auxvar) */
   SCIP_CONSEXPR_EXPRENFO** enfos;        /**< enforcements */
   int                     nenfos;        /**< number of enforcements */

   /* separation */
   SCIP_VAR*               auxvar;        /**< auxiliary variable used for outer approximation cuts */
   unsigned int            sepatag;       /**< tag of point for which an outer approximation cut has been computed last, or 0 */

   /* branching */
   SCIP_Real               brscore;       /**< branching score for the expression (passed on to children) */
   unsigned int            brscoretag;    /**< tag to decide whether a branching score of an expression needs to be initialized */
   unsigned int            brscoreevaltag;/**< tag to decide whether the branching scoring callback of an expression needs to be called */

   /* point-evaluation */
   unsigned int            evaltag;       /**< tag of point for which the expression has been evaluated last, or 0 */
   SCIP_Real               evalvalue;     /**< value of expression from last evaluation (corresponding to evaltag) */
   SCIP_Real               derivative;    /**< partial derivative of a "root path" w.r.t. this expression
                                            *  (see documentation of Differentiation methods in cons_expr.c) */
   unsigned int            difftag;       /**< when computing partial derivatives of an expression w.r.t. a variable,
                                            *  the tag allows us to decide whether the expression depends on the
                                            *  variable; the tag will be checked in SCIPgetConsExprExprPartialDiff() */

   /* interval-evaluation */
   unsigned int            intevaltag;    /**< tag of domains for which tag for which the expression has been evaluated last, or 0 */
   SCIP_INTERVAL           interval;      /**< interval from the last interval evaluation */

   /* propagation */
   SCIP_Bool               inqueue;       /**< flag to store whether an expression is in the queue of reverse propagation */
   SCIP_Bool               hastightened;  /**< flag to store whether expression has been tightened during reverse propagation */

   /* separation initialization */
   unsigned int            initsepatag;   /**< flag to store whether an expression has been called during the separation initialization */

   /* expression walker data */
   SCIP_CONSEXPR_EXPR*     walkparent;    /**< parent expression in expression walk */
   int                     walkcurrentchild; /**< child that is currently visited (or will be visited next) by expression walk */
   SCIP_CONSEXPREXPRWALK_IO walkio;       /**< space for walker callback to store some (temporary) data, e.g., to simulate input or output values of a recursive call */

   /* curvature information */
   SCIP_EXPRCURV           curvature;     /**< curvature of the expression w.r.t. bounds that have been used in the last curvature detection */

   /* monotonicity information of each child */
   SCIP_MONOTONE*          monotonicity;  /**< array containing monotonicity of expression w.r.t. each children */
   int                     monotonicitysize; /**< length of monotonicity array */

   /* integrality information */
   SCIP_Bool               isintegral;     /**< flag to store whether an expression is integral */
};

/** generic data and callback methods of an nonlinear handler */
struct SCIP_ConsExpr_Nlhdlr
{
   char*                         name;       /**< nonlinearity handler name */
   char*                         desc;       /**< nonlinearity handler description (can be NULL) */
   SCIP_CONSEXPR_NLHDLRDATA*     data;       /**< data of handler */
   unsigned int                  priority;   /**< priority of nonlinearity handler */
   SCIP_Bool                     enabled;    /**< whether the nonlinear handler should be used */

   SCIP_Longint                  nsepacalls; /**< number of times, the separation or estimation callback was called */
   SCIP_Longint                  npropcalls; /**< number of times, the propagation callback was called */
   SCIP_Longint                  ncutsfound; /**< number of cuts added by this expression handler */
   SCIP_Longint                  ncutoffs;   /**< number of cutoffs found so far by this nonlinear handler */
   SCIP_Longint                  ndomreds;   /**< number of domain reductions found so far by this expression handler */
   SCIP_Longint                  ndetections;/**< number of detect calls in which structure was detected (success returned by detect call) (over all runs) */
   SCIP_Longint                  nbranchscores; /**< number of times, branching scores were added by this nonlinear handler */

   SCIP_CLOCK*                   detecttime; /**< time used for detection */
   SCIP_CLOCK*                   sepatime;   /**< time used for separation or estimation */
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
   SCIP_DECL_CONSEXPR_NLHDLRSEPA((*sepa));                  /**< separation callback (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLRESTIMATE((*estimate));          /**< estimator callback (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLREXITSEPA((*exitsepa));          /**< separation deinitialization callback (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLRINTEVAL((*inteval));            /**< interval evaluation callback (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLRREVERSEPROP((*reverseprop));    /**< reverse propagation callback (can be NULL) */
   SCIP_DECL_CONSEXPR_NLHDLRBRANCHSCORE((*branchscore));    /**< branching scoring callback (can be NULL) */
};

/** enforcement data of an expression */
struct SCIP_ConsExpr_ExprEnfo
{
   SCIP_CONSEXPR_NLHDLR*         nlhdlr;          /**< nonlinear handler */
   SCIP_CONSEXPR_NLHDLREXPRDATA* nlhdlrexprdata;  /**< data of nonlinear handler */
   SCIP_Bool                     issepainit;      /**< was the initsepa callback of nlhdlr called */
   SCIP_Real                     auxvalue;        /**< auxiliary value of expression w.r.t. currently enforced solution */
};

/** expression tree iterator */
struct SCIP_ConsExpr_Iterator
{
   SCIP_CONSEXPRITERATOR_TYPE  itertype;     /**< type of expression iterator */
   BMS_BLKMEM*                 blkmem;       /**< block memory */
   SCIP_CONSEXPR_EXPR*         curr;         /**< current expression of the iterator */
   SCIP_CONSEXPR_EXPR**        dfsexprs;     /**< DFS stack */
   int*                        dfsnvisited;  /**< number of visited children for each expression in the stack */
   int                         dfsnexprs;    /**< total number of expression in stack */
   int                         dfssize;      /**< size DFS stack */
   SCIP_QUEUE*                 queue;        /**< BFS queue */
};

#ifdef __cplusplus
}
#endif

#endif /* __SCIP_STRUCT_CONS_EXPR_H__ */
