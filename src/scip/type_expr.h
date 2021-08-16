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

/**@file   type_expr.h
 * @ingroup TYPEDEFINITIONS
 * @brief  type definitions related to algebraic expressions
 * @author Ksenia Bestuzheva
 * @author Benjamin Mueller
 * @author Felipe Serrano
 * @author Stefan Vigerske
 *
 *  This file defines the interface for expression handlers.
 *
 *  - \ref EXPRHDLRS "List of available expression handlers"
 */

#ifndef SCIP_TYPE_EXPR_H_
#define SCIP_TYPE_EXPR_H_

#include "scip/def.h"
#include "scip/intervalarith.h"
#include "scip/type_scip.h"
#include "scip/type_sol.h"
#include "scip/type_var.h"
#include "scip/type_tree.h"

typedef struct SCIP_ExprData  SCIP_EXPRDATA;     /**< expression data */
typedef struct SCIP_Expr      SCIP_EXPR;         /**< expression */

typedef struct SCIP_Expr_OwnerData SCIP_EXPR_OWNERDATA; /**< data stored by expression owner in expression */

/** curvature types */
typedef enum
{
   SCIP_EXPRCURV_UNKNOWN    = 0,             /**< unknown curvature (or indefinite) */
   SCIP_EXPRCURV_CONVEX     = 1,             /**< convex */
   SCIP_EXPRCURV_CONCAVE    = 2,             /**< concave */
   SCIP_EXPRCURV_LINEAR     = SCIP_EXPRCURV_CONVEX | SCIP_EXPRCURV_CONCAVE/**< linear = convex and concave */
} SCIP_EXPRCURV;

/** monotonicity */
typedef enum
{
   SCIP_MONOTONE_UNKNOWN      = 0,          /**< unknown */
   SCIP_MONOTONE_INC          = 1,          /**< increasing */
   SCIP_MONOTONE_DEC          = 2,          /**< decreasing */
   SCIP_MONOTONE_CONST        = SCIP_MONOTONE_INC | SCIP_MONOTONE_DEC /**< constant */
} SCIP_MONOTONE;

/** the maximal number of estimates an expression handler can return in the INITESTIMATES callback */
#define SCIP_EXPR_MAXINITESTIMATES 10

/** callback for freeing ownerdata of expression
 *
 * This callback is called while an expression is freed.
 * The callback shall free the ownerdata, if any.
 * That is, the callback is also called on expressions that only store this callback, but no ownerdata.
 *
 * Note, that the children of the expression have already been release when this callback is called.
 * The callback must not try to access the expressions children.
 *
 *  - scip      : SCIP main data structure
 *  - expr      : the expression which is freed
 *  - ownerdata : the ownerdata stored in the expression
 */
#define SCIP_DECL_EXPR_OWNERFREE(x) SCIP_RETCODE x(\
   SCIP*                 scip, \
   SCIP_EXPR*            expr, \
   SCIP_EXPR_OWNERDATA** ownerdata)

/** callback for printing ownerdata of expression
 *
 * This callback is called when printing details on an expression, e.g., SCIPdismantleExpr().
 *
 *  - scip      : SCIP main data structure
 *  - expr      : the expression which is printed
 *  - file      : file to print to, or NULL for stdout
 *  - ownerdata : the ownerdata stored in the expression
 */
#define SCIP_DECL_EXPR_OWNERPRINT(x) SCIP_RETCODE x(\
   SCIP*                 scip, \
   FILE*                 file, \
   SCIP_EXPR*            expr, \
   SCIP_EXPR_OWNERDATA*  ownerdata)

/** callback for owner-specific activity evaluation
 *
 * This callback is called when evaluating the activity of an expression, e.g., SCIPevalActivity().
 * The callback should ensure that activity is updated, if required, by calling SCIPsetActivity().
 * The callback can use the activitytag in the expression to recognize whether it needs to become active.
 *
 *  - scip           : SCIP main data structure
 *  - expr           : the expression for which activity should be updated
 *  - ownerdata      : the ownerdata stored in the expression
 */
#define SCIP_DECL_EXPR_OWNEREVALACTIVITY(x) SCIP_RETCODE x(\
   SCIP*                 scip, \
   SCIP_EXPR*            expr, \
   SCIP_EXPR_OWNERDATA*  ownerdata)

/** callback for creating ownerdata of expression
 *
 * This callback is called when an expression has been created.
 * It can create data which is then stored in the expression.
 *
 *  - scip              : SCIP main data structure
 *  - expr              : the expression that has been created
 *  - ownercreatedata   : data that has been passed on by future owner of expression that can be used to create ownerdata
 *  - ownerdata         : buffer to store ownerdata that shall be stored in expression (can be NULL, as it is initialized)
 *  - ownerfree         : buffer to store function to be called to free ownerdata when expression is freed (can be NULL, as it is initialized)
 *  - ownerprint        : buffer to store function to be called to print ownerdata (can be NULL, as it is initialized)
 *  - ownerevalactivity : buffer to store function to be called to evaluate activity (can be NULL, as it is initialized)
 */
#define SCIP_DECL_EXPR_OWNERCREATE(x) SCIP_RETCODE x(\
   SCIP*                 scip,                              \
   SCIP_EXPR*            expr,                              \
   SCIP_EXPR_OWNERDATA** ownerdata,                         \
   SCIP_DECL_EXPR_OWNERFREE((**ownerfree)),                 \
   SCIP_DECL_EXPR_OWNERPRINT((**ownerprint)),               \
   SCIP_DECL_EXPR_OWNEREVALACTIVITY((**ownerevalactivity)), \
   void*                 ownercreatedata)

/** callback that returns bounds for a given variable as used in interval evaluation
 *
 * Implements a relaxation scheme for variable bounds and translates between different infinity values.
 * Returns an interval that contains the current variable bounds, but might be (slightly) larger.
 *
 *  - scip           : SCIP main data structure
 *  - var            : variable for which to obtain bounds
 *  - intevalvardata : data that belongs to this callback
 */
#define SCIP_DECL_EXPR_INTEVALVAR(x) SCIP_INTERVAL x (\
   SCIP*     scip,          \
   SCIP_VAR* var,           \
   void*     intevalvardata \
   )

/** expression mapping callback for expression copy callback
 *
 * The method maps an expression (in a source SCIP instance) to an expression
 * (in a target SCIP instance) and captures the target expression.
 *
 *  - targetscip      : target SCIP main data structure
 *  - targetexpr      : pointer to store the mapped expression, or NULL if expression shall be copied; initialized to NULL
 *  - sourcescip      : source SCIP main data structure
 *  - sourceexpr      : expression to be mapped
 *  - ownercreate     : callback to call when creating a new expression
 *  - ownercreatedata : data for ownercreate callback
 *  - mapexprdata     : data of callback
 */
#define SCIP_DECL_EXPR_MAPEXPR(x) SCIP_RETCODE x (\
   SCIP*       targetscip,                     \
   SCIP_EXPR** targetexpr,                     \
   SCIP*       sourcescip,                     \
   SCIP_EXPR*  sourceexpr,                     \
   SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)), \
   void*       ownercreatedata,                \
   void*       mapexprdata)

/**@name Expression Handler */
/**@{ */

typedef struct SCIP_Exprhdlr     SCIP_EXPRHDLR;     /**< expression handler */
typedef struct SCIP_ExprhdlrData SCIP_EXPRHDLRDATA; /**< expression handler data */

/** expression handler copy callback
 *
 * the method includes the expression handler into a SCIP instance
 *
 * This method is usually called when doing a copy of SCIP.
 *
 *  - scip           : target SCIP main data structure
 *  - sourceexprhdlr : expression handler in source SCIP
 */
#define SCIP_DECL_EXPRCOPYHDLR(x) SCIP_RETCODE x (\
   SCIP*          scip, \
   SCIP_EXPRHDLR* sourceexprhdlr)

/** expression handler free callback
 *
 * the callback frees the data of an expression handler
 *
 * - scip          : SCIP main data structure
 * - exprhdlr      : expression handler
 * - exprhdlrdata  : expression handler data to be freed
 */
#define SCIP_DECL_EXPRFREEHDLR(x) SCIP_RETCODE x (\
   SCIP*               scip,     \
   SCIP_EXPRHDLR*      exprhdlr, \
   SCIP_EXPRHDLRDATA** exprhdlrdata)

/** expression data copy callback
 *
 * the method copies the data of an expression
 *
 * This method is called when creating copies of an expression within
 * the same or between different SCIP instances. It is given the
 * source expression which data shall be copied. It expects
 * that *targetexprdata will be set. This data will then be used
 * to create a new expression.
 *
 * This callback must be implemented for expressions that have data.
 *
 *  - targetscip     : target SCIP main data structure
 *  - targetexprhdlr : expression handler in target SCIP
 *  - targetexprdata : pointer to store the copied expression data
 *  - sourcescip     : source SCIP main data structure
 *  - sourceexpr     : expression in source SCIP which data is to be copied
 */
#define SCIP_DECL_EXPRCOPYDATA(x) SCIP_RETCODE x (\
   SCIP*           targetscip,     \
   SCIP_EXPRHDLR*  targetexprhdlr, \
   SCIP_EXPRDATA** targetexprdata, \
   SCIP*           sourcescip,     \
   SCIP_EXPR*      sourceexpr)

/** expression data free callback
 *
 * The method frees the data of an expression.
 * It assumes that expr->exprdata will be set to NULL.
 *
 * This callback must be implemented for expressions that have data.
 *
 *  - scip : SCIP main data structure
 *  - expr : the expression which data to be freed
 */
#define SCIP_DECL_EXPRFREEDATA(x) SCIP_RETCODE x (\
   SCIP*      scip, \
   SCIP_EXPR* expr)

/** expression print callback
 *
 * the method prints an expression
 * it is called while iterating over the expression graph at different stages
 *
 *  - scip             : SCIP main data structure
 *  - expr             : expression which data is to be printed
 *  - stage            : stage of expression iteration
 *  - currentchild     : index of current child if in stage visitingchild or visitedchild
 *  - parentprecedence : precedence of parent
 *  - file             : the file to print to
 */
#define SCIP_DECL_EXPRPRINT(x) SCIP_RETCODE x (\
   SCIP*               scip, \
   SCIP_EXPR*          expr, \
   SCIP_EXPRITER_STAGE stage, \
   int                 currentchild, \
   unsigned int        parentprecedence, \
   FILE*               file)

/** expression parse callback
 *
 * the method parses an expression
 * it is called when parsing an expression and an operator with the expr handler name is found
 *
 *  - scip            : SCIP main data structure
 *  - string          : string containing expression to be parse
 *  - ownercreate     : function to call to create ownerdata
 *  - ownercreatedata : data to pass to ownercreate
 *  - endstring       : buffer to store the position of string after parsing
 *  - expr            : buffer to store the parsed expression
 *  - success         : buffer to store whether the parsing was successful or not
 */
#define SCIP_DECL_EXPRPARSE(x) SCIP_RETCODE x (\
   SCIP*          scip,                        \
   SCIP_EXPRHDLR* exprhdlr,                    \
   const char*    string,                      \
   const char**   endstring,                   \
   SCIP_EXPR**    expr,                        \
   SCIP_Bool*     success,                     \
   SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)), \
   void*          ownercreatedata)

/** expression curvature detection callback
 *
 * The method returns whether an expression can have a desired curvature under conditions on the
 * curvature of the children.
 * That is, the method shall return TRUE in success and requirements on the curvature for each child
 * which will suffice for this expression to be convex (or concave, or linear, as specified by caller)
 * w.r.t. the current activities of all children.
 * It can return "unknown" for a child's curvature if its curvature does not matter (though that's
 * rarely the case).
 *
 * The method assumes that the activity evaluation of the expression has been called before
 * and the expression has been simplified.
 *
 *  - scip          : SCIP main data structure
 *  - expr          : expression to check the curvature for
 *  - exprcurvature : desired curvature of this expression
 *  - success       : buffer to store whether the desired curvature was obtained
 *  - childcurv     : array to store required curvature for each child
 */
#define SCIP_DECL_EXPRCURVATURE(x) SCIP_RETCODE x (\
   SCIP*          scip,          \
   SCIP_EXPR*     expr,          \
   SCIP_EXPRCURV  exprcurvature, \
   SCIP_Bool*     success,       \
   SCIP_EXPRCURV* childcurv)

/** expression monotonicity detection callback
 *
 * The method computes the monotonicity of an expression with respect to a given child.
 *
 *  - scip     : SCIP main data structure
 *  - expr     : expression to check the monotonicity for
 *  - childidx : index of the considered child expression
 *  - result   : buffer to store the monotonicity
 */
#define SCIP_DECL_EXPRMONOTONICITY(x) SCIP_RETCODE x (\
   SCIP*          scip,     \
   SCIP_EXPR*     expr,     \
   int            childidx, \
   SCIP_MONOTONE* result)

/** expression integrality detection callback
 *
 * The method computes whether an expression evaluates always to an integral value.
 *
 *  - scip       : SCIP main data structure
 *  - expr       : expression to check the integrality for
 *  - isintegral : buffer to store whether expr is integral
 */
#define SCIP_DECL_EXPRINTEGRALITY(x) SCIP_RETCODE x (\
   SCIP*      scip, \
   SCIP_EXPR* expr, \
   SCIP_Bool* isintegral)

/** expression hash callback
 *
 * The method hashes an expression by taking the hashes of its children into account.
 *
 *  - scip           : SCIP main data structure
 *  - expr           : expression to be hashed
 *  - hashkey        : buffer to store the hash value
 *  - childrenhashes : array with hash values of children
 */
#define SCIP_DECL_EXPRHASH(x) SCIP_RETCODE x (\
   SCIP*         scip,    \
   SCIP_EXPR*    expr,    \
   unsigned int* hashkey, \
   unsigned int* childrenhashes)

/** expression compare callback
 *
 * the method receives two expressions, expr1 and expr2. Must return
 * -1 if expr1 < expr2
 * 0  if expr1 = expr2
 * 1  if expr1 > expr2
 *
 *  - scip  : SCIP main data structure
 *  - expr1 : first expression in comparison
 *  - expr2 : second expression in comparison
 */
#define SCIP_DECL_EXPRCOMPARE(x) int x (\
   SCIP*      scip,  \
   SCIP_EXPR* expr1, \
   SCIP_EXPR* expr2)

/** expression (point-) evaluation callback
 *
 * The method evaluates an expression by taking the values of its children into account.
 *
 *  - scip : SCIP main data structure
 *  - expr : expression to be evaluated
 *  - val  : buffer where to store value
 *  - sol  : solution that is evaluated (can be NULL)
 */
#define SCIP_DECL_EXPREVAL(x) SCIP_RETCODE x (\
   SCIP*      scip, \
   SCIP_EXPR* expr, \
   SCIP_Real* val,  \
   SCIP_SOL*  sol)

/** backward derivative evaluation callback
 *
 * The method should compute the partial derivative of expr w.r.t. its child at childidx.
 * That is, it should return
 * \f[
 *   \frac{\partial \text{expr}}{\partial \text{child}_{\text{childidx}}}
 * \f]
 *
 *  - scip     : SCIP main data structure
 *  - expr     : expression to be differentiated
 *  - childidx : index of the child
 *  - val      : buffer to store the partial derivative w.r.t. the i-th children
 */
#define SCIP_DECL_EXPRBWDIFF(x) SCIP_RETCODE x (\
   SCIP*      scip,     \
   SCIP_EXPR* expr,     \
   int        childidx, \
   SCIP_Real* val)

/** forward derivative evaluation callback
 *
 * The method should evaluate the directional derivative of expr.
 * The expr should be interpreted as an operator \f$ \text{expr}(c_1, \ldots, c_n) \f$, where \f$ c_1, \ldots, c_n \f$
 * are the children of the expr.
 * The directional derivative is evaluated at the point
 *   \f$ \text{SCIPexprGetEvalValue}(c_1), \ldots, \text{SCIPexprGetEvalValue}(c_n) \f$
 * in the direction given by direction.
 *
 * This method should return
 * \f[
 *    \sum_{i = 1}^n \frac{\partial \text{expr}}{\partial c_i} D_u c_i,
 * \f]
 * where \f$ u \f$ is the direction and \f$ D_u c_i \f$ is the directional derivative of the i-th child,
 * which can be accessed via SCIPexprGetDot.
 *
 * See Differentiation methods in scip_expr.h for more details.
 *
 *  - scip      : SCIP main data structure
 *  - expr      : expression to be differentiated
 *  - dot       : buffer to store derivative value
 *  - direction : direction of the derivative (useful only for var expressions)
 */
#define SCIP_DECL_EXPRFWDIFF(x) SCIP_RETCODE x (\
   SCIP*      scip, \
   SCIP_EXPR* expr, \
   SCIP_Real* dot,  \
   SCIP_SOL*  direction)

/** derivative evaluation callback for Hessian directions (backward over forward)
 *
 * The method computes the total derivative, w.r.t its children, of the partial derivative of expr w.r.t childidx
 * Equivalently, it computes the partial derivative w.r.t childidx of the total derivative
 *
 * The expr should be interpreted as an operator \f$ \text{expr}(c_1, \ldots, c_n) \f$, where \f$ c_1, \ldots, c_n \f$
 * are the children of the expr.
 * The directional derivative is evaluated at the point
 *   \f$ \text{SCIPexprGetEvalValue}(c_1), \ldots, \text{SCIPexprGetEvalValue}(c_n) \f$
 * in the direction given by direction.
 *
 * This method should return
 * \f[
 *    \sum_{i = 1}^n \frac{\partial^2 \text{expr}}{\partial c_i} \partial c_{\text{childidx}} D_u c_i,
 * \f]
 *
 * where \f$ u \f$ is the direction and \f$ D_u c_i \f$ is the directional derivative of the i-th child,
 * which can be accessed via SCIPexprGetDot.
 *
 * Thus, if \f$ n = 1 \f$ (i.e. if expr represents a univariate operator), the method should return
 * \f[
 *    \text{expr}^{\prime \prime}}(\text{SCIPexprGetEvalValue}(c))  D_u c.
 * \f]
 *
 * See Differentiation methods in scip_expr.h for more details.
 *
 *  - scip      : SCIP main data structure
 *  - expr      : expression to be evaluated
 *  - childidx  : index of the child
 *  - bardot    : buffer to store derivative value
 *  - direction : direction of the derivative (useful only for var expressions)
 */
#define SCIP_DECL_EXPRBWFWDIFF(x) SCIP_RETCODE x (\
   SCIP*      scip,     \
   SCIP_EXPR* expr,     \
   int        childidx, \
   SCIP_Real* bardot,   \
   SCIP_SOL*  direction)

/** expression (interval-) evaluation callback
 *
 * The method evaluates an expression by taking the intervals of its children into account.
 *
 *  - scip           : SCIP main data structure
 *  - expr           : expression to be evaluated
 *  - interval       : buffer where to store interval
 *  - intevalvar     : callback to be called when interval evaluating a variable
 *  - intevalvardata : data to be passed to intevalvar callback
 */
#define SCIP_DECL_EXPRINTEVAL(x) SCIP_RETCODE x (\
   SCIP*          scip,                      \
   SCIP_EXPR*     expr,                      \
   SCIP_INTERVAL* interval,                  \
   SCIP_DECL_EXPR_INTEVALVAR((*intevalvar)), \
   void*          intevalvardata)

/** expression under/overestimation callback
 *
 * The method tries to compute a linear under- or overestimator that is as tight as possible
 * at a given point. The estimator must be valid w.r.t. the bounds given by localbounds.
 * If the value of the estimator in the reference point is smaller (larger) than targetvalue
 * when underestimating (overestimating), then no estimator needs to be computed.
 * Note, that targetvalue can be infinite if any estimator will be accepted.
 * If successful, it shall store the coefficient of the i-th child in entry coefs[i] and
 * the constant part in \par constant.
 * If the estimator is also valid w.r.t. the bounds given by globalbounds, then *islocal shall
 * be set to FALSE.
 * The callback shall set branchcand[i] to FALSE if branching in the i-th child would not
 * improve the estimator. That is, branchcand[i] will be initialized to TRUE for all children.
 *
 *  - scip         : SCIP main data structure
 *  - expr         : expression
 *  - localbounds  : current bounds for children
 *  - globalbounds : global bounds for children
 *  - refpoint     : values for children in the reference point where to estimate
 *  - overestimate : whether the expression needs to be over- or underestimated
 *  - targetvalue  : a value that the estimator shall exceed, can be +/-infinity
 *  - coefs        : array to store coefficients of estimator
 *  - constant     : buffer to store constant part of estimator
 *  - islocal      : buffer to store whether estimator is valid locally only
 *  - success      : buffer to indicate whether an estimator could be computed
 *  - branchcand   : array to indicate which children (not) to consider for branching
 */
#define SCIP_DECL_EXPRESTIMATE(x) SCIP_RETCODE x (\
   SCIP*          scip,         \
   SCIP_EXPR*     expr,         \
   SCIP_INTERVAL* localbounds,  \
   SCIP_INTERVAL* globalbounds, \
   SCIP_Real*     refpoint,     \
   SCIP_Bool      overestimate, \
   SCIP_Real      targetvalue,  \
   SCIP_Real*     coefs,        \
   SCIP_Real*     constant,     \
   SCIP_Bool*     islocal,      \
   SCIP_Bool*     success,      \
   SCIP_Bool*     branchcand)

/** expression initial under/overestimation callback
 *
 * The method tries to compute a few linear under- or overestimator that approximate the
 * behavior of the expression. The estimator must be valid w.r.t. the bounds given by bounds.
 * These estimators may be used to initialize a linear relaxation.
 * The callback shall return the number of computed estimators in nreturned,
 * store the coefficient of the i-th child for the j-th estimator in entry coefs[j][i],
 * and store the constant part for the j-th estimator in *constant[j].
 *
 *  - scip         : SCIP main data structure
 *  - expr         : expression
 *  - bounds       : bounds for children
 *  - overestimate : whether the expression shall be overestimated or underestimated
 *  - coefs        : buffer to store coefficients of computed estimators
 *  - constant     : buffer to store constant of computed estimators
 *  - nreturned    : buffer to store number of estimators that have been computed
 */
#define SCIP_DECL_EXPRINITESTIMATES(x) SCIP_RETCODE x ( \
   SCIP*          scip,                                 \
   SCIP_EXPR*     expr,                                 \
   SCIP_INTERVAL* bounds,                               \
   SCIP_Bool      overestimate,                         \
   SCIP_Real*     coefs[SCIP_EXPR_MAXINITESTIMATES],    \
   SCIP_Real      constant[SCIP_EXPR_MAXINITESTIMATES], \
   int*           nreturned)

/** expression simplify callback
 *
 * the method receives the expression to be simplified and a pointer to store the simplified expression
 *
 * input:
 *  - scip            : SCIP main data structure
 *  - expr            : expression to simplify
 *  - ownercreate     : function to call to create ownerdata
 *  - ownercreatedata : data to pass to ownercreate
 *  - simplifiedexpr  : buffer to store the simplified expression
 */
#define SCIP_DECL_EXPRSIMPLIFY(x) SCIP_RETCODE x (\
   SCIP*          scip,                        \
   SCIP_EXPR*     expr,                        \
   SCIP_EXPR**    simplifiedexpr,              \
   SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)), \
   void*          ownercreatedata)

/** expression callback for reverse propagation
 *
 * The method propagates given bounds over the children of an expression.
 *
 *  - scip           : SCIP main data structure
 *  - expr           : expression
 *  - bounds         : the bounds on the expression that should be propagated
 *  - childrenbounds : array to store computed bounds for children, initialized with current activity
 *  - infeasible     : buffer to store whether a children bounds were propagated to an empty interval
 */
#define SCIP_DECL_EXPRREVERSEPROP(x) SCIP_RETCODE x (\
   SCIP*          scip,           \
   SCIP_EXPR*     expr,           \
   SCIP_INTERVAL  bounds,         \
   SCIP_INTERVAL* childrenbounds, \
   SCIP_Bool*     infeasible)

/** @} */  /* expression handler */



/** @name expression iterator
 * @{
 */

/** maximal number of iterators that can be active on an expression graph concurrently
 *
 * How often an expression graph iteration can be started within an active iteration, plus one.
 */
#define SCIP_EXPRITER_MAXNACTIVE 5

/** stages of expression DFS iteration */
#define SCIP_EXPRITER_ENTEREXPR     1u /**< an expression is visited the first time (before any of its children are visited) */
#define SCIP_EXPRITER_VISITINGCHILD 2u /**< a child of an expression is to be visited */
#define SCIP_EXPRITER_VISITEDCHILD  4u /**< a child of an expression has been visited */
#define SCIP_EXPRITER_LEAVEEXPR     8u /**< an expression is to be left (all of its children have been processed) */
#define SCIP_EXPRITER_ALLSTAGES     (SCIP_EXPRITER_ENTEREXPR | SCIP_EXPRITER_VISITINGCHILD | SCIP_EXPRITER_VISITEDCHILD | SCIP_EXPRITER_LEAVEEXPR)

/** type to represent stage of DFS iterator */
typedef unsigned int SCIP_EXPRITER_STAGE;

/** user data storage type for expression iteration */
typedef union
{
   SCIP_Real     realval;            /**< a floating-point value */
   int           intval;             /**< an integer value */
   int           intvals[2];         /**< two integer values */
   unsigned int  uintval;            /**< an unsigned integer value */
   void*         ptrval;             /**< a pointer */
} SCIP_EXPRITER_USERDATA;

/** mode for expression iterator */
typedef enum
{
   SCIP_EXPRITER_RTOPOLOGIC,         /**< reverse topological order */
   SCIP_EXPRITER_BFS,                /**< breadth-first search */
   SCIP_EXPRITER_DFS                 /**< depth-first search */
} SCIP_EXPRITER_TYPE;

typedef struct SCIP_ExprIterData SCIP_EXPRITERDATA;  /**< expression iterator data for a specific expression */
typedef struct SCIP_ExprIter     SCIP_EXPRITER;      /**< expression iterator */

/** @} */

/** @name expression printing
 * @{
 */

#define SCIP_EXPRPRINT_EXPRSTRING   0x1u /**< print the math. function that the expression represents (e.g., "c0+c1") */
#define SCIP_EXPRPRINT_EXPRHDLR     0x2u /**< print expression handler name */
#define SCIP_EXPRPRINT_NUSES        0x4u /**< print number of uses (reference counting) */
#define SCIP_EXPRPRINT_EVALVALUE    0x8u /**< print evaluation value */
#define SCIP_EXPRPRINT_EVALTAG     0x18u /**< print evaluation value and tag */
#define SCIP_EXPRPRINT_ACTIVITY    0x20u /**< print activity value */
#define SCIP_EXPRPRINT_ACTIVITYTAG 0x60u /**< print activity value and corresponding tag */
#define SCIP_EXPRPRINT_OWNER       0x80u /**< print ownerdata */

/** print everything */
#define SCIP_EXPRPRINT_ALL SCIP_EXPRPRINT_EXPRSTRING | SCIP_EXPRPRINT_EXPRHDLR | SCIP_EXPRPRINT_NUSES | SCIP_EXPRPRINT_EVALTAG | SCIP_EXPRPRINT_ACTIVITYTAG | SCIP_EXPRPRINT_OWNER

typedef unsigned int              SCIP_EXPRPRINT_WHAT; /**< type for exprprint bitflags */
typedef struct SCIP_ExprPrintData SCIP_EXPRPRINTDATA;  /**< printing a expression file data */

/** @} */


#endif /* SCIP_TYPE_EXPR_H_ */
