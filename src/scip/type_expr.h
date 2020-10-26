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
 * @brief  type definitions related to algebraic expressions
 * @author Ksenia Bestuzheva
 * @author Benjamin Mueller
 * @author Felipe Serrano
 * @author Stefan Vigerske
 */

#ifndef SCIP_TYPE_EXPR_H_
#define SCIP_TYPE_EXPR_H_

typedef struct SCIP_ConsExpr_ExprData  SCIP_CONSEXPR_EXPRDATA;     /**< expression data */
typedef struct SCIP_ConsExpr_Expr      SCIP_CONSEXPR_EXPR;         /**< expression */

typedef struct SCIP_ConsExpr_QuadExpr      SCIP_CONSEXPR_QUADEXPR;      /**< representation of expression as quadratic */

/** monotonicity of an expression */
typedef enum
{
   SCIP_MONOTONE_UNKNOWN      = 0,          /**< unknown */
   SCIP_MONOTONE_INC          = 1,          /**< increasing */
   SCIP_MONOTONE_DEC          = 2,          /**< decreasing */
   SCIP_MONOTONE_CONST        = SCIP_MONOTONE_INC | SCIP_MONOTONE_DEC /**< constant */

} SCIP_MONOTONE;


/** callback that returns bounds for a given variable as used in interval evaluation
 *
 * Implements a relaxation scheme for variable bounds and translates between different infinity values.
 *
 *  input:
 *  - scip           : SCIP main data structure
 *  - var            : variable for which to obtain bounds
 *  - intevalvardata : data that belongs to this callback
 *  output:
 *  - returns an interval that contains the current variable bounds, but might be (slightly) larger
 */
#define SCIP_DECL_CONSEXPR_INTEVALVAR(x) SCIP_INTERVAL x (\
   SCIP* scip, \
   SCIP_VAR* var, \
   void* intevalvardata \
   )

/** variable mapping callback for expression data callback
 *
 * The method maps a variable (in a source SCIP instance) to a variable
 * (in a target SCIP instance) and captures the target variable.
 *
 *  input:
 *  - targetscip         : target SCIP main data structure
 *  - targetvar          : pointer to store the mapped variable
 *  - sourcescip         : source SCIP main data structure
 *  - sourcevar          : variable to be mapped
 *  - mapvardata         : data of callback
 */
#define SCIP_DECL_CONSEXPR_MAPVAR(x) SCIP_RETCODE x (\
   SCIP* targetscip, \
   SCIP_VAR** targetvar, \
   SCIP* sourcescip, \
   SCIP_VAR* sourcevar, \
   void* mapvardata \
   )

/**@name Expression Handler */
/**@{ */

/** expression handler copy callback
 *
 * the method includes the expression handler into a expression constraint handler
 *
 * This method is usually called when doing a copy of an expression constraint handler.
 *
 *  input:
 *  - scip              : target SCIP main data structure
 *  - consexprhdlr      : target expression constraint handler
 *  - sourceconsexprhdlr : expression constraint handler in source SCIP
 *  - sourceexprhdlr    : expression handler in source SCIP
 *  - valid             : to store indication whether the expression handler was copied
 */
#define SCIP_DECL_CONSEXPR_EXPRCOPYHDLR(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSHDLR* consexprhdlr, \
   SCIP_CONSHDLR* sourceconsexprhdlr, \
   SCIP_CONSEXPR_EXPRHDLR* sourceexprhdlr, \
   SCIP_Bool* valid)

/** expression handler free callback
 *
 * the callback frees the data of an expression handler
 *
 *  input:
 *  - scip          : SCIP main data structure
 *  - consexprhdlr  : expression constraint handler
 *  - exprhdlr      : expression handler
 *  - exprhdlrdata  : expression handler data to be freed
 */
#define SCIP_DECL_CONSEXPR_EXPRFREEHDLR(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSHDLR* consexprhdlr, \
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr, \
   SCIP_CONSEXPR_EXPRHDLRDATA** exprhdlrdata)

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
 *  input:
 *  - targetscip         : target SCIP main data structure
 *  - targetexprhdlr     : expression handler in target SCIP
 *  - targetexprdata     : pointer to store the copied expression data
 *  - sourcescip         : source SCIP main data structure
 *  - sourceexpr         : expression in source SCIP which data is to be copied,
 *  - mapvar             : variable mapping callback for use by variable expression handler
 *  - mapvardata         : data of variable mapping callback
 */
#define SCIP_DECL_CONSEXPR_EXPRCOPYDATA(x) SCIP_RETCODE x (\
   SCIP* targetscip, \
   SCIP_CONSEXPR_EXPRHDLR* targetexprhdlr, \
   SCIP_CONSEXPR_EXPRDATA** targetexprdata, \
   SCIP* sourcescip, \
   SCIP_CONSEXPR_EXPR* sourceexpr, \
   SCIP_DECL_CONSEXPR_MAPVAR(mapvar), \
   void* mapvardata)

/** expression data free callback
 *
 * The method frees the data of an expression.
 * It assumes that expr->exprdata will be set to NULL.
 *
 * This callback must be implemented for expressions that have data.
 *
 *  input:
 *  - scip          : SCIP main data structure
 *  - expr          : the expression which data to be freed
 */
#define SCIP_DECL_CONSEXPR_EXPRFREEDATA(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSEXPR_EXPR* expr)

/** expression print callback
 *
 * the method prints an expression
 * it is called while iterating over the expression graph at different stages
 *
 * input:
 *  - scip : SCIP main data structure
 *  - expr : expression which data is to be printed
 *  - stage: stage of expression graph iteration
 *  - currentchild: index of current child if in stage visitingchild or visitedchild
 *  - parentprecedence: precedence of parent
 *  - file : the file to print to
 */
#define SCIP_DECL_CONSEXPR_EXPRPRINT(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSEXPR_EXPR* expr, \
   SCIP_CONSEXPRITERATOR_STAGE stage, \
   int currentchild, \
   unsigned int parentprecedence, \
   FILE* file)

/** expression parse callback
 *
 * the method parses an expression
 * it is called when parsing a constraint and an operator with the expr handler name is found
 *
 * input:
 *  - scip         : SCIP main data structure
 *  - consexprhdlr : expression constraint handler
 *  - string       : string containing expression to be parse
 *
 *  output:
 *  - endstring    : pointer to store the position of string after parsing
 *  - expr         : pointer to store the parsed expression
 *  - success      : pointer to store whether the parsing was successful or not
 */
#define SCIP_DECL_CONSEXPR_EXPRPARSE(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSHDLR* consexprhdlr, \
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr, \
   const char* string, \
   const char** endstring, \
   SCIP_CONSEXPR_EXPR** expr, \
   SCIP_Bool* success)

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
 * input:
 *  - scip : SCIP main data structure
 *  - conshdlr : expression constraint handler
 *  - expr : expression to check the curvature for
 *  - exprcurvature : desired curvature of this expression
 *  - success: buffer to store whether the desired curvature be obtained
 *  - childcurv: array to store required curvature for each child
 */
#define SCIP_DECL_CONSEXPR_EXPRCURVATURE(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSHDLR* conshdlr, \
   SCIP_CONSEXPR_EXPR* expr, \
   SCIP_EXPRCURV exprcurvature, \
   SCIP_Bool* success, \
   SCIP_EXPRCURV* childcurv )

/** expression monotonicity detection callback
 *
 * The method computes the monotonicity of an expression with respect to a given child.
 *
 * input:
 *  - scip : SCIP main data structure
 *  - conshdlr: cons_expr constraint handler
 *  - expr : expression to check the monotonicity for
 *  - childidx : index of the considered child expression
 *  - result : buffer to store the monotonicity
 */
#define SCIP_DECL_CONSEXPR_EXPRMONOTONICITY(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSHDLR* conshdlr, \
   SCIP_CONSEXPR_EXPR* expr, \
   int childidx, \
   SCIP_MONOTONE* result)

/** expression integrality detection callback
 *
 * The method computes whether an expression evaluates always to an integral value.
 *
 * input:
 *  - scip : SCIP main data structure
 *  - expr : expression to check the integrality for
 *  - isintegral : buffer to store whether expr is integral
 */
#define SCIP_DECL_CONSEXPR_EXPRINTEGRALITY(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSEXPR_EXPR* expr, \
   SCIP_Bool* isintegral)

/** expression hash callback
 *
 * The method hashes an expression by taking the hashes of its children into account.
 *
 * input:
 *  - scip : SCIP main data structure
 *  - expr : expression to be hashed
 *  - hashkey : pointer to store the hash value
 *  - childrenhashes : array with hash values of children
 */
#define SCIP_DECL_CONSEXPR_EXPRHASH(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSEXPR_EXPR* expr, \
   unsigned int* hashkey, \
   unsigned int* childrenhashes)

/** expression compare callback
 *
 * the method receives two expressions, expr1 and expr2. Must return
 * -1 if expr1 < expr2
 * 0  if expr1 = expr2
 * 1  if expr1 > expr2
 *
 * input:
 *  - expr1 : first expression to compare
 *  - expr2 : second expression to compare
 */
#define SCIP_DECL_CONSEXPR_EXPRCOMPARE(x) int x (\
   SCIP_CONSEXPR_EXPR* expr1, \
   SCIP_CONSEXPR_EXPR* expr2)

/** backward derivative evaluation callback
 *
 * The method should compute the partial derivative of expr w.r.t its child at childidx.
 * That is, it should return
 * \f[
 *   \frac{\partial \text{expr}}{\partial \text{child}_{\text{childidx}}}
 * \f]
 *
 * input:
 *  - scip : SCIP main data structure
 *  - expr : expression to be evaluated
 *  - childidx : index of the child
 *  - val : buffer to store the partial derivative w.r.t. the i-th children
 */
#define SCIP_DECL_CONSEXPR_EXPRBWDIFF(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSEXPR_EXPR* expr, \
   int childidx, \
   SCIP_Real* val)

/** forward derivative evaluation callback
 *
 * The method should evaluate the directional derivative of expr.
 * The expr should be interpreted as an operator \f$ \text{expr}(c_1, \ldots, c_n) \f$, where \f$ c_1, \ldots, c_n \f$
 * are the children of the expr.
 * The directional derivative is evaluated at the point
 *   \f$ \text{SCIPgetConsExprExprValue}(c_1), \ldots, \text{SCIPgetConsExprExprValue}(c_n) \f$
 * in the direction given by direction.
 *
 * This method should return
 * \f[
 *    \sum_{i = 1}^n \frac{\partial \text{expr}}{\partial c_i} D_u c_i,
 * \f]
 * where \f$ u \f$ is the direction and \f$ D_u c_i \f$ is the directional derivative of the i-th child,
 * which can be accessed via SCIPgetConsExprExprDot.
 *
 * See Differentiation methods in cons_expr.h for more details.
 *
 * input:
 *  - scip : SCIP main data structure
 *  - expr : expression to be evaluated
 *  - dot : buffer to store derivative value
 *  - direction : direction of the derivative (useful only for var expressions)
 *
 *  TODO: think whether we actually need to pass direction. Right now, the direction is being set
 *  to the var expressions in SCIPcomputeConsExprHessianDir and it is not used anywhere else.
 *  If we remove direction, update documentation accordingly
 */
#define SCIP_DECL_CONSEXPR_EXPRFWDIFF(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSEXPR_EXPR* expr, \
   SCIP_Real* dot, \
   SCIP_SOL* direction)

/** derivative evaluation callback for hessian directions (backward over forward)
 *
 * The method computes the total derivative, w.r.t its children, of the partial derivative of expr w.r.t childidx
 * Equivalently, it computes the partial derivative w.r.t childidx of the total derivative
 *
 * The expr should be interpreted as an operator \f$ \text{expr}(c_1, \ldots, c_n) \f$, where \f$ c_1, \ldots, c_n \f$
 * are the children of the expr.
 * The directional derivative is evaluated at the point
 *   \f$ \text{SCIPgetConsExprExprValue}(c_1), \ldots, \text{SCIPgetConsExprExprValue}(c_n) \f$
 * in the direction given by direction.
 *
 * This method should return
 * \f[
 *    \sum_{i = 1}^n \frac{\partial^2 \text{expr}}{\partial c_i} \partial c_{\text{childidx}} D_u c_i,
 * \f]
 *
 * where \f$ u \f$ is the direction and \f$ D_u c_i \f$ is the directional derivative of the i-th child,
 * which can be accessed via SCIPgetConsExprExprDot.
 *
 * Thus, if \f$ n = 1 \f$ (i.e. if expr represents a univariate operator), the method should return
 * \f[
 *    \text{expr}^{\prime \prime}}(\text{SCIPgetConsExprExprValue}(c))  D_u c.
 * \f]
 *
 * See Differentiation methods in cons_expr.h for more details.
 *
 * input:
 *  - scip : SCIP main data structure
 *  - expr : expression to be evaluated
 *  - childidx : index of the child
 *  - bardot : buffer to store derivative value
 *  - direction : direction of the derivative (useful only for var expressions)
 */
#define SCIP_DECL_CONSEXPR_EXPRBWFWDIFF(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSEXPR_EXPR* expr, \
   int childidx, \
   SCIP_Real* bardot, \
   SCIP_SOL* direction)

/** expression (point-) evaluation callback
 *
 * The method evaluates an expression by taking the values of its children into account.
 * We might extend this later to store (optionally) also information for
 * gradient and Hessian computations.
 *
 * input:
 *  - scip : SCIP main data structure
 *  - expr : expression to be evaluated
 *  - val : buffer where to store value
 *  - sol : solution that is evaluated (can be NULL)
 */
#define SCIP_DECL_CONSEXPR_EXPREVAL(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSEXPR_EXPR* expr, \
   SCIP_Real* val, \
   SCIP_SOL* sol)

/** expression (interval-) evaluation callback
 *
 * The method evaluates an expression by taking the intervals of its children into account.
 *
 * input:
 *  - scip : SCIP main data structure
 *  - interval : buffer where to store interval
 *  - expr : expression to be evaluated
 *  - intevalvar : callback to be called when interval evaluating a variable
 *  - intevalvardata : data to be passed to intevalvar callback
 */
#define SCIP_DECL_CONSEXPR_EXPRINTEVAL(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSEXPR_EXPR* expr, \
   SCIP_INTERVAL* interval, \
   SCIP_DECL_CONSEXPR_INTEVALVAR((*intevalvar)), \
   void* intevalvardata)

/** expression under/overestimation callback
 *
 * The method tries to compute a linear under- or overestimator that is as tight as possible
 * at a given point by using auxiliary variables stored in all children.
 * If the value of the estimator in the solution is smaller (larger) than targetvalue
 * when underestimating (overestimating), then no estimator needs to be computed.
 * Note, that targetvalue can be infinite if any estimator will be accepted.
 * If successful, it shall store the coefficient of the i-th child in entry coefs[i] and
 * the constant part in \par constant.
 * The callback shall set branchcand[i] to FALSE if branching in the i-th child would not
 * improve the estimator. That is, branchcand[i] will be initialized to TRUE for all children.
 *
 * input:
 *  - scip : SCIP main data structure
 *  - expr : expression
 *  - sol  : solution at which to estimate (NULL for the LP solution)
 *  - overestimate : whether the expression needs to be over- or underestimated
 *  - targetvalue : a value that the estimator shall exceed, can be +/-infinity
 *  - coefs : array to store coefficients of estimator
 *  - constant : buffer to store constant part of estimator
 *  - islocal : buffer to store whether estimator is valid locally only
 *  - success : buffer to indicate whether an estimator could be computed
 *  - branchcand: array to indicate which children (not) to consider for branching
 */
#define SCIP_DECL_CONSEXPR_EXPRESTIMATE(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSHDLR* conshdlr, \
   SCIP_CONSEXPR_EXPR* expr, \
   SCIP_SOL* sol, \
   SCIP_Bool overestimate, \
   SCIP_Real targetvalue, \
   SCIP_Real* coefs, \
   SCIP_Real* constant, \
   SCIP_Bool* islocal, \
   SCIP_Bool* success, \
   SCIP_Bool* branchcand)

/** expression simplify callback
 *
 * the method receives the expression to be simplified and a pointer to store the simplified expression
 *
 * input:
 *  - scip           : SCIP main data structure
 *  - consexprhdlr   : expression constraint handler
 *  - expr           : expression to simplify
 * output:
 *  - simplifiedexpr : the simplified expression
 */
#define SCIP_DECL_CONSEXPR_EXPRSIMPLIFY(x) SCIP_RETCODE x (\
   SCIP*                 scip,               \
   SCIP_CONSHDLR*        conshdlr,           \
   SCIP_CONSEXPR_EXPR*   expr,               \
   SCIP_CONSEXPR_EXPR**  simplifiedexpr)

/** expression callback for reverse propagation
 *
 * The method propagates given bounds over the children of an expression.
 * The tighter interval should be passed to the corresponding child expression by using
 * SCIPtightenConsExprExprInterval().
 *
 * input:
 *  - scip : SCIP main data structure
 *  - conshdlr: expr constraint handler
 *  - expr : expression
 *  - bounds : the bounds on the expression that should be propagated
 *  - infeasible: buffer to store whether an expression's bounds were propagated to an empty interval
 *  - nreductions : buffer to store the number of interval reductions of all children
 */
#define SCIP_DECL_CONSEXPR_EXPRREVERSEPROP(x) SCIP_RETCODE x (\
   SCIP* scip, \
   SCIP_CONSHDLR* conshdlr, \
   SCIP_CONSEXPR_EXPR* expr, \
   SCIP_INTERVAL bounds, \
   SCIP_Bool* infeasible, \
   int* nreductions )

/** separation initialization method of an expression handler (called during CONSINITLP)
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - conshdlr        : expression constraint handler
 *  - cons            : expression constraint
 *  - expr            : expression
 *  - overestimate    : whether the expression needs to be overestimated
 *  - underestimate   : whether the expression needs to be underestimated
 *
 *  output:
 *  - infeasible      : pointer to store whether an infeasibility was detected while building the LP
 */
#define SCIP_DECL_CONSEXPR_EXPRINITSEPA(x) SCIP_RETCODE x (\
      SCIP* scip, \
      SCIP_CONSHDLR* conshdlr, \
      SCIP_CONS* cons, \
      SCIP_CONSEXPR_EXPR* expr, \
      SCIP_Bool overestimate, \
      SCIP_Bool underestimate, \
      SCIP_Bool* infeasible)

/** separation deinitialization method of an expression handler (called during CONSEXITSOL)
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - expr            : expression
 */
#define SCIP_DECL_CONSEXPR_EXPREXITSEPA(x) SCIP_RETCODE x (\
      SCIP* scip, \
      SCIP_CONSEXPR_EXPR* expr)

typedef struct SCIP_ConsExpr_ExprHdlr     SCIP_CONSEXPR_EXPRHDLR;     /**< expression handler */
typedef struct SCIP_ConsExpr_ExprHdlrData SCIP_CONSEXPR_EXPRHDLRDATA; /**< expression handler data */

/** @} */  /* expression handler */



/** @name expression iterator
 * @{
 */

/** maximal number of iterators that can be active on an expression graph concurrently
 *
 * How often an expression graph iteration can be started within an active iteration, plus one.
 */
#define SCIP_CONSEXPRITERATOR_MAXNACTIVE 5

/** stages of expression DFS iteration */
#define SCIP_CONSEXPRITERATOR_ENTEREXPR     1u /**< an expression is visited the first time (before any of its children are visited) */
#define SCIP_CONSEXPRITERATOR_VISITINGCHILD 2u /**< a child of an expression is to be visited */
#define SCIP_CONSEXPRITERATOR_VISITEDCHILD  4u /**< a child of an expression has been visited */
#define SCIP_CONSEXPRITERATOR_LEAVEEXPR     8u /**< an expression is to be left (all of its children have been processed) */
#define SCIP_CONSEXPRITERATOR_ALLSTAGES     (SCIP_CONSEXPRITERATOR_ENTEREXPR | SCIP_CONSEXPRITERATOR_VISITINGCHILD | SCIP_CONSEXPRITERATOR_VISITEDCHILD | SCIP_CONSEXPRITERATOR_LEAVEEXPR)

/** type to represent stage of DFS iterator */
typedef unsigned int SCIP_CONSEXPRITERATOR_STAGE;

/** user data storage type for expression iteration */
typedef union
{
   SCIP_Real             realval;            /**< a floating-point value */
   int                   intval;             /**< an integer value */
   int                   intvals[2];         /**< two integer values */
   unsigned int          uintval;            /**< an unsigned integer value */
   void*                 ptrval;             /**< a pointer */
} SCIP_CONSEXPRITERATOR_USERDATA;

/** mode for expression iterator */
typedef enum
{
   SCIP_CONSEXPRITERATOR_RTOPOLOGIC,         /**< reverse topological order */
   SCIP_CONSEXPRITERATOR_BFS,                /**< breadth-first search */
   SCIP_CONSEXPRITERATOR_DFS                 /**< depth-first search */
} SCIP_CONSEXPRITERATOR_TYPE;

typedef struct SCIP_ConsExpr_Expr_IterData SCIP_CONSEXPR_EXPR_ITERDATA; /**< expression tree iterator data for a specific expression */
typedef struct SCIP_ConsExpr_Iterator      SCIP_CONSEXPR_ITERATOR;      /**< expression tree iterator */

/** @} */

/** @name expression printing
 * @{
 */

#define SCIP_CONSEXPR_PRINTDOT_EXPRSTRING   0x1u /**< print the math. function that the expression represents (e.g., "c0+c1") */
#define SCIP_CONSEXPR_PRINTDOT_EXPRHDLR     0x2u /**< print expression handler name */
#define SCIP_CONSEXPR_PRINTDOT_NUSES        0x4u /**< print number of uses (reference counting) */
#define SCIP_CONSEXPR_PRINTDOT_NLOCKS       0x8u /**< print number of locks */
#define SCIP_CONSEXPR_PRINTDOT_EVALVALUE   0x10u /**< print evaluation value */
#define SCIP_CONSEXPR_PRINTDOT_EVALTAG     0x30u /**< print evaluation value and tag */
#define SCIP_CONSEXPR_PRINTDOT_ACTIVITY    0x40u /**< print activity value */
#define SCIP_CONSEXPR_PRINTDOT_ACTIVITYTAG 0xC0u /**< print activity value and corresponding tag */

/** print everything */
#define SCIP_CONSEXPR_PRINTDOT_ALL SCIP_CONSEXPR_PRINTDOT_EXPRSTRING | SCIP_CONSEXPR_PRINTDOT_EXPRHDLR | SCIP_CONSEXPR_PRINTDOT_NUSES | SCIP_CONSEXPR_PRINTDOT_NLOCKS | SCIP_CONSEXPR_PRINTDOT_EVALTAG | SCIP_CONSEXPR_PRINTDOT_ACTIVITYTAG


typedef unsigned int                      SCIP_CONSEXPR_PRINTDOT_WHAT; /**< type for printdot bitflags */
typedef struct SCIP_ConsExpr_PrintDotData SCIP_CONSEXPR_PRINTDOTDATA;  /**< printing a dot file data */

/** @} */


#endif /* SCIP_TYPE_EXPR_H_ */
