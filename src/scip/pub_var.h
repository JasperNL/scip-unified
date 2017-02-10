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

/**@file   pub_var.h
 * @ingroup PUBLICCOREAPI
 * @brief  public methods for problem variables
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_PUB_VAR_H__
#define __SCIP_PUB_VAR_H__


#include "scip/def.h"
#include "scip/type_retcode.h"
#include "scip/type_misc.h"
#include "scip/type_history.h"
#include "scip/type_var.h"
#include "scip/type_implics.h"
#include "scip/type_cons.h"

#ifdef NDEBUG
#include "scip/struct_var.h"
#include "scip/implics.h"
#include "scip/history.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * methods for variables 
 */

/**@addtogroup PublicVariableMethods
 *
 * @{
 */

/** gets number of locks for rounding down */
EXTERN
int SCIPvarGetNLocksDown(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets number of locks for rounding up */
EXTERN
int SCIPvarGetNLocksUp(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** is it possible, to round variable down and stay feasible? */
EXTERN
SCIP_Bool SCIPvarMayRoundDown(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** is it possible, to round variable up and stay feasible? */
EXTERN
SCIP_Bool SCIPvarMayRoundUp(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** compares the index of two variables, only active or negated variables are allowed, if a variable
 *  is negated then the index of the corresponding active variable is taken, returns -1 if first is
 *  smaller than, and +1 if first is greater than second variable index; returns 0 if both indices
 *  are equal, which means both variables are equal
 */
EXTERN
int SCIPvarCompareActiveAndNegated(
   SCIP_VAR*             var1,               /**< first problem variable */
   SCIP_VAR*             var2                /**< second problem variable */
   );

/** comparison method for sorting active and negated variables by non-decreasing index, active and negated 
 *  variables are handled as the same variables
 */
EXTERN
SCIP_DECL_SORTPTRCOMP(SCIPvarCompActiveAndNegated);

/** compares the index of two variables, returns -1 if first is smaller than, and +1 if first is greater than second
 *  variable index; returns 0 if both indices are equal, which means both variables are equal
 */
EXTERN
int SCIPvarCompare(
   SCIP_VAR*             var1,               /**< first problem variable */
   SCIP_VAR*             var2                /**< second problem variable */
   );

/** comparison method for sorting variables by non-decreasing index */
EXTERN
SCIP_DECL_SORTPTRCOMP(SCIPvarComp);

/** comparison method for sorting variables by non-decreasing objective coefficient */
EXTERN
SCIP_DECL_SORTPTRCOMP(SCIPvarCompObj);

/** hash key retrieval function for variables */
EXTERN
SCIP_DECL_HASHGETKEY(SCIPvarGetHashkey);

/** returns TRUE iff the indices of both variables are equal */
EXTERN
SCIP_DECL_HASHKEYEQ(SCIPvarIsHashkeyEq);

/** returns the hash value of the key */
EXTERN
SCIP_DECL_HASHKEYVAL(SCIPvarGetHashkeyVal);


/** gets corresponding active, fixed, or multi-aggregated problem variables of given variables,
 *  @note the content of the given array will/might change
 */
EXTERN
void SCIPvarsGetProbvar(
   SCIP_VAR**            vars,               /**< array of problem variables */
   int                   nvars               /**< number of variables */
   );

/** gets corresponding active, fixed, or multi-aggregated problem variable of a variable */
EXTERN
SCIP_VAR* SCIPvarGetProbvar(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets corresponding active, fixed, or multi-aggregated problem variables of binary variables and
 *  updates the given negation status of each variable
 */
EXTERN
SCIP_RETCODE SCIPvarsGetProbvarBinary(
   SCIP_VAR***           vars,               /**< pointer to binary problem variables */
   SCIP_Bool**           negatedarr,         /**< pointer to corresponding array to update the negation status */
   int                   nvars               /**< number of variables and values in vars and negated array */
   );

/** gets corresponding active, fixed, or multi-aggregated problem variable of a binary variable and
 *  updates the given negation status
 */
EXTERN
SCIP_RETCODE SCIPvarGetProbvarBinary(
   SCIP_VAR**            var,                /**< pointer to binary problem variable */
   SCIP_Bool*            negated             /**< pointer to update the negation status */
   );

/** transforms given variable, boundtype and bound to the corresponding active, fixed, or multi-aggregated variable
 *  values
 */
EXTERN
SCIP_RETCODE SCIPvarGetProbvarBound(
   SCIP_VAR**            var,                /**< pointer to problem variable */
   SCIP_Real*            bound,              /**< pointer to bound value to transform */
   SCIP_BOUNDTYPE*       boundtype           /**< pointer to type of bound: lower or upper bound */
   );

/** transforms given variable and domain hole to the corresponding active, fixed, or multi-aggregated variable
 *  values
 */
EXTERN
SCIP_RETCODE SCIPvarGetProbvarHole(
   SCIP_VAR**            var,                /**< pointer to problem variable */
   SCIP_Real*            left,               /**< pointer to left bound of open interval in hole to transform */
   SCIP_Real*            right               /**< pointer to right bound of open interval in hole to transform */
   );

/** retransforms given variable, scalar and constant to the corresponding original variable, scalar
 *  and constant, if possible; if the retransformation is impossible, NULL is returned as variable
 */
EXTERN
SCIP_RETCODE SCIPvarGetOrigvarSum(
   SCIP_VAR**            var,                /**< pointer to problem variable x in sum a*x + c */
   SCIP_Real*            scalar,             /**< pointer to scalar a in sum a*x + c */
   SCIP_Real*            constant            /**< pointer to constant c in sum a*x + c */
   );

/** returns whether the given variable is the direct counterpart of an original problem variable */
EXTERN
SCIP_Bool SCIPvarIsTransformedOrigvar(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the number of times, a bound of the variable was changed in given direction due to branching */
EXTERN
SCIP_Longint SCIPvarGetNBranchings(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the number of times, a bound of the variable was changed in given direction due to branching
 *  in the current run
 */
EXTERN
SCIP_Longint SCIPvarGetNBranchingsCurrentRun(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the number of inferences branching on this variable in given direction triggered */
EXTERN
SCIP_Real SCIPvarGetInferenceSum(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the number of inferences branching on this variable in given direction triggered
 *  in the current run
 */
EXTERN
SCIP_Real SCIPvarGetInferenceSumCurrentRun(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the number of cutoffs branching on this variable in given direction produced */
EXTERN
SCIP_Real SCIPvarGetCutoffSum(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the number of cutoffs branching on this variable in given direction produced in the current run */
EXTERN
SCIP_Real SCIPvarGetCutoffSumCurrentRun(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the average depth of bound changes in given direction due to branching on the variable */
EXTERN
SCIP_Real SCIPvarGetAvgBranchdepth(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns the average depth of bound changes in given direction due to branching on the variable
 *  in the current run
 */
EXTERN
SCIP_Real SCIPvarGetAvgBranchdepthCurrentRun(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   );

/** returns whether there is an implication x == varfixing -> y <= b or y >= b in the implication graph;
 *  implications that are represented as cliques in the clique table are not regarded (use SCIPvarsHaveCommonClique());
 *  both variables must be active, variable x must be binary
 */
EXTERN
SCIP_Bool SCIPvarHasImplic(
   SCIP_VAR*             var,                /**< problem variable x */
   SCIP_Bool             varfixing,          /**< FALSE if y should be searched in implications for x == 0, TRUE for x == 1 */
   SCIP_VAR*             implvar,            /**< variable y to search for */
   SCIP_BOUNDTYPE        impltype            /**< type of implication y <=/>= b to search for */
   );

/** returns whether there is an implication x == varfixing -> y == implvarfixing in the implication graph;
 *  implications that are represented as cliques in the clique table are not regarded (use SCIPvarsHaveCommonClique());
 *  both variables must be active binary variables
 */
EXTERN
SCIP_Bool SCIPvarHasBinaryImplic(
   SCIP_VAR*             var,                /**< problem variable x */
   SCIP_Bool             varfixing,          /**< FALSE if y should be searched in implications for x == 0, TRUE for x == 1 */
   SCIP_VAR*             implvar,            /**< variable y to search for */
   SCIP_Bool             implvarfixing       /**< value of the implied variable to search for */
   );

/** returns whether there is a clique that contains both given variable/value pairs;
 *  the variables must be active binary variables;
 *  if regardimplics is FALSE, only the cliques in the clique table are looked at;
 *  if regardimplics is TRUE, both the cliques and the implications of the implication graph are regarded
 */
EXTERN
SCIP_Bool SCIPvarsHaveCommonClique(
   SCIP_VAR*             var1,               /**< first variable */
   SCIP_Bool             value1,             /**< value of first variable */
   SCIP_VAR*             var2,               /**< second variable */
   SCIP_Bool             value2,             /**< value of second variable */
   SCIP_Bool             regardimplics       /**< should the implication graph also be searched for a clique? */
   );

/** gets corresponding objective value of active, fixed, or multi-aggregated problem variable of given variable
 *  e.g. obj(x) = 1 this method returns for ~x the value -1
 */
EXTERN
SCIP_RETCODE SCIPvarGetAggregatedObj(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real*            aggrobj             /**< pointer to store the aggregated objective value */
   );

/** sets the initial flag of a variable; only possible for original or loose variables */
EXTERN
SCIP_RETCODE SCIPvarSetInitial(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Bool             initial             /**< initial flag */
   );

/** sets the removable flag of a variable; only possible for original or loose variables */
EXTERN
SCIP_RETCODE SCIPvarSetRemovable(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Bool             removable           /**< removable flag */
   );

/** returns the name of the variable
 *
 *  @note to change the name of a variable, use SCIPchgVarName() from scip.h
 */
EXTERN
const char* SCIPvarGetName(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets number of times, the variable is currently captured */
EXTERN
int SCIPvarGetNUses(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the user data of the variable */
EXTERN
SCIP_VARDATA* SCIPvarGetData(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** sets the user data for the variable */
EXTERN
void SCIPvarSetData(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_VARDATA*         vardata             /**< user variable data */
   );

/** sets method to free user data for the original variable */
EXTERN
void SCIPvarSetDelorigData(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_DECL_VARDELORIG  ((*vardelorig))     /**< frees user data of original variable */
   );

/** sets method to transform user data of the variable */
EXTERN
void SCIPvarSetTransData(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_DECL_VARTRANS    ((*vartrans))       /**< creates transformed user data by transforming original user data */
   );

/** sets method to free transformed user data for the variable */
EXTERN
void SCIPvarSetDeltransData(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_DECL_VARDELTRANS ((*vardeltrans))    /**< frees user data of transformed variable */
   );

/** sets method to copy this variable into sub-SCIPs */
EXTERN
void SCIPvarSetCopyData(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_DECL_VARCOPY     ((*varcopy))        /**< copy method of the variable */
   );

/** gets status of variable */
EXTERN
SCIP_VARSTATUS SCIPvarGetStatus(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns whether the variable belongs to the original problem */
EXTERN
SCIP_Bool SCIPvarIsOriginal(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns whether the variable belongs to the transformed problem */
EXTERN
SCIP_Bool SCIPvarIsTransformed(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns whether the variable was created by negation of a different variable */
EXTERN
SCIP_Bool SCIPvarIsNegated(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets type of variable */
EXTERN
SCIP_VARTYPE SCIPvarGetType(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns TRUE if the variable is of binary type; this is the case if:
 *  (1) variable type is binary
 *  (2) variable type is integer or implicit integer and 
 *      (i)  the lazy lower bound or the global lower bound is greater than or equal to zero
 *      (ii) the lazy upper bound or the global upper bound is less than or equal to one
 */
EXTERN
SCIP_Bool SCIPvarIsBinary(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns whether variable is of integral type (binary, integer, or implicit integer) */
EXTERN
SCIP_Bool SCIPvarIsIntegral(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns whether variable's column should be present in the initial root LP */
EXTERN
SCIP_Bool SCIPvarIsInitial(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns whether variable's column is removable from the LP (due to aging or cleanup) */
EXTERN
SCIP_Bool SCIPvarIsRemovable(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns whether the variable was deleted from the problem */
EXTERN
SCIP_Bool SCIPvarIsDeleted(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** marks the variable to be deletable, i.e., it may be deleted completely from the problem;
 *  method can only be called before the variable is added to the problem by SCIPaddVar() or SCIPaddPricedVar()
 */
EXTERN
void SCIPvarMarkDeletable(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** marks the variable to be not deletable from the problem */
EXTERN
void SCIPvarMarkNotDeletable(
   SCIP_VAR*             var
   );

/** returns whether variable is allowed to be deleted completely from the problem */
EXTERN
SCIP_Bool SCIPvarIsDeletable(
   SCIP_VAR*             var
   );

/** returns whether variable is an active (neither fixed nor aggregated) variable */
EXTERN
SCIP_Bool SCIPvarIsActive(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets unique index of variable */
EXTERN
int SCIPvarGetIndex(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets position of variable in problem, or -1 if variable is not active */
EXTERN
int SCIPvarGetProbindex(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets transformed variable of ORIGINAL variable */
EXTERN
SCIP_VAR* SCIPvarGetTransVar(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets column of COLUMN variable */
EXTERN
SCIP_COL* SCIPvarGetCol(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns whether the variable is a COLUMN variable that is member of the current LP */
EXTERN
SCIP_Bool SCIPvarIsInLP(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets aggregation variable y of an aggregated variable x = a*y + c */
EXTERN
SCIP_VAR* SCIPvarGetAggrVar(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets aggregation scalar a of an aggregated variable x = a*y + c */
EXTERN
SCIP_Real SCIPvarGetAggrScalar(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets aggregation constant c of an aggregated variable x = a*y + c */
EXTERN
SCIP_Real SCIPvarGetAggrConstant(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets number n of aggregation variables of a multi aggregated variable x = a0*y0 + ... + a(n-1)*y(n-1) + c */
EXTERN
int SCIPvarGetMultaggrNVars(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets vector of aggregation variables y of a multi aggregated variable x = a0*y0 + ... + a(n-1)*y(n-1) + c */
EXTERN
SCIP_VAR** SCIPvarGetMultaggrVars(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets vector of aggregation scalars a of a multi aggregated variable x = a0*y0 + ... + a(n-1)*y(n-1) + c */
EXTERN
SCIP_Real* SCIPvarGetMultaggrScalars(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets aggregation constant c of a multi aggregated variable x = a0*y0 + ... + a(n-1)*y(n-1) + c */
EXTERN
SCIP_Real SCIPvarGetMultaggrConstant(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets the negation of the given variable; may return NULL, if no negation is existing yet */
EXTERN
SCIP_VAR* SCIPvarGetNegatedVar(
   SCIP_VAR*             var                 /**< negated problem variable */
   );

/** gets the negation variable x of a negated variable x' = offset - x */
EXTERN
SCIP_VAR* SCIPvarGetNegationVar(
   SCIP_VAR*             var                 /**< negated problem variable */
   );

/** gets the negation offset of a negated variable x' = offset - x */
EXTERN
SCIP_Real SCIPvarGetNegationConstant(
   SCIP_VAR*             var                 /**< negated problem variable */
   );

/** gets objective function value of variable */
EXTERN
SCIP_Real SCIPvarGetObj(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets the unchanged objective function value of variable (ignoring temproray changes performed in probing mode) */
EXTERN
SCIP_Real SCIPvarGetUnchangedObj(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets original lower bound of original problem variable (i.e. the bound set in problem creation) */
EXTERN
SCIP_Real SCIPvarGetLbOriginal(
   SCIP_VAR*             var                 /**< original problem variable */
   );

/** gets original upper bound of original problem variable (i.e. the bound set in problem creation) */
EXTERN
SCIP_Real SCIPvarGetUbOriginal(
   SCIP_VAR*             var                 /**< original problem variable */
   );

/** gets the original hole list of an original variable */
EXTERN
SCIP_HOLELIST* SCIPvarGetHolelistOriginal(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets global lower bound of variable */
EXTERN
SCIP_Real SCIPvarGetLbGlobal(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets global upper bound of variable */
EXTERN
SCIP_Real SCIPvarGetUbGlobal(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets the global hole list of an active variable */
EXTERN
SCIP_HOLELIST* SCIPvarGetHolelistGlobal(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets best global bound of variable with respect to the objective function */
EXTERN
SCIP_Real SCIPvarGetBestBoundGlobal(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets worst global bound of variable with respect to the objective function */
EXTERN
SCIP_Real SCIPvarGetWorstBoundGlobal(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets current lower bound of variable */
EXTERN
SCIP_Real SCIPvarGetLbLocal(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets current upper bound of variable */
EXTERN
SCIP_Real SCIPvarGetUbLocal(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets the current hole list of an active variable */
EXTERN
SCIP_HOLELIST* SCIPvarGetHolelistLocal(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets best local bound of variable with respect to the objective function */
EXTERN
SCIP_Real SCIPvarGetBestBoundLocal(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets worst local bound of variable with respect to the objective function */
EXTERN
SCIP_Real SCIPvarGetWorstBoundLocal(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets type (lower or upper) of best bound of variable with respect to the objective function */
EXTERN
SCIP_BOUNDTYPE SCIPvarGetBestBoundType(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets type (lower or upper) of worst bound of variable with respect to the objective function */
EXTERN
SCIP_BOUNDTYPE SCIPvarGetWorstBoundType(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets lazy lower bound of variable */
EXTERN
SCIP_Real SCIPvarGetLbLazy(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets lazy upper bound of variable */
EXTERN
SCIP_Real SCIPvarGetUbLazy(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets the branch factor of the variable; this value can be used in the branching methods to scale the score
 *  values of the variables; higher factor leads to a higher probability that this variable is chosen for branching
 */
EXTERN
SCIP_Real SCIPvarGetBranchFactor(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets the branch priority of the variable; variables with higher priority should always be preferred to variables
 *  with lower priority
 */
EXTERN
int SCIPvarGetBranchPriority(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets the preferred branch direction of the variable (downwards, upwards, or auto) */
EXTERN
SCIP_BRANCHDIR SCIPvarGetBranchDirection(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets number of variable lower bounds x >= b_i*z_i + d_i of given variable x */
EXTERN
int SCIPvarGetNVlbs(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets array with bounding variables z_i in variable lower bounds x >= b_i*z_i + d_i of given variable x;
 *  the variable bounds are sorted by increasing variable index of the bounding variable z_i (see SCIPvarGetIndex())
 */
EXTERN
SCIP_VAR** SCIPvarGetVlbVars(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets array with bounding coefficients b_i in variable lower bounds x >= b_i*z_i + d_i of given variable x */
EXTERN
SCIP_Real* SCIPvarGetVlbCoefs(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets array with bounding constants d_i in variable lower bounds x >= b_i*z_i + d_i of given variable x */
EXTERN
SCIP_Real* SCIPvarGetVlbConstants(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets number of variable upper bounds x <= b_i*z_i + d_i of given variable x */
EXTERN
int SCIPvarGetNVubs(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets array with bounding variables z_i in variable upper bounds x <= b_i*z_i + d_i of given variable x;
 *  the variable bounds are sorted by increasing variable index of the bounding variable z_i (see SCIPvarGetIndex())
 */
EXTERN
SCIP_VAR** SCIPvarGetVubVars(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets array with bounding coefficients b_i in variable upper bounds x <= b_i*z_i + d_i of given variable x */
EXTERN
SCIP_Real* SCIPvarGetVubCoefs(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets array with bounding constants d_i in variable upper bounds x <= b_i*z_i + d_i of given variable x */
EXTERN
SCIP_Real* SCIPvarGetVubConstants(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets number of implications  y <= b or y >= b for x == 0 or x == 1 of given active problem variable x, 
 *  there are no implications for nonbinary variable x
 */
EXTERN
int SCIPvarGetNImpls(
   SCIP_VAR*             var,                /**< active problem variable */
   SCIP_Bool             varfixing           /**< FALSE for implications for x == 0, TRUE for x == 1 */
   );

/** gets array with implication variables y of implications  y <= b or y >= b for x == 0 or x == 1 of given active
 *  problem variable x, there are no implications for nonbinary variable x;
 *  the implications are sorted such that implications with binary implied variables precede the ones with non-binary
 *  implied variables, and as a second criteria, the implied variables are sorted by increasing variable index
 *  (see SCIPvarGetIndex())
 */
EXTERN
SCIP_VAR** SCIPvarGetImplVars(
   SCIP_VAR*             var,                /**< active problem variable */
   SCIP_Bool             varfixing           /**< FALSE for implications for x == 0, TRUE for x == 1 */
   );

/** gets array with implication types of implications  y <= b or y >= b for x == 0 or x == 1 of given active problem
 *  variable x (SCIP_BOUNDTYPE_UPPER if y <= b, SCIP_BOUNDTYPE_LOWER if y >= b), 
 *  there are no implications for nonbinary variable x
 */
EXTERN
SCIP_BOUNDTYPE* SCIPvarGetImplTypes(
   SCIP_VAR*             var,                /**< active problem variable */
   SCIP_Bool             varfixing           /**< FALSE for implications for x == 0, TRUE for x == 1 */
   );

/** gets array with implication bounds b of implications  y <= b or y >= b for x == 0 or x == 1 of given active problem
 *  variable x, there are no implications for nonbinary variable x
 */
EXTERN
SCIP_Real* SCIPvarGetImplBounds(
   SCIP_VAR*             var,                /**< active problem variable */
   SCIP_Bool             varfixing           /**< FALSE for implications for x == 0, TRUE for x == 1 */
   );

/** Gets array with unique ids of implications  y <= b or y >= b for x == 0 or x == 1 of given active problem variable x,
 *  there are no implications for nonbinary variable x.
 *  If an implication is a shortcut, i.e., it was added as part of the transitive closure of another implication,
 *  its id is negative, otherwise it is nonnegative.
 */
EXTERN
int* SCIPvarGetImplIds(
   SCIP_VAR*             var,                /**< active problem variable */
   SCIP_Bool             varfixing           /**< FALSE for implications for x == 0, TRUE for x == 1 */
   );

/** gets number of cliques, the active variable is contained in */
EXTERN
int SCIPvarGetNCliques(
   SCIP_VAR*             var,                /**< active problem variable */
   SCIP_Bool             varfixing           /**< FALSE for cliques containing x == 0, TRUE for x == 1 */
   );

/** gets array of cliques, the active variable is contained in */
EXTERN
SCIP_CLIQUE** SCIPvarGetCliques(
   SCIP_VAR*             var,                /**< active problem variable */
   SCIP_Bool             varfixing           /**< FALSE for cliques containing x == 0, TRUE for x == 1 */
   );

/** gets primal LP solution value of variable */
EXTERN
SCIP_Real SCIPvarGetLPSol(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets primal NLP solution value of variable */
EXTERN
SCIP_Real SCIPvarGetNLPSol(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** return lower bound change info at requested position */
EXTERN
SCIP_BDCHGINFO* SCIPvarGetBdchgInfoLb(
   SCIP_VAR*             var,                /**< problem variable */
   int                   pos                 /**< requested position */
   );

/** gets the number of lower bound change info array */
EXTERN
int SCIPvarGetNBdchgInfosLb(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** return upper bound change info at requested position */
EXTERN
SCIP_BDCHGINFO* SCIPvarGetBdchgInfoUb(
   SCIP_VAR*             var,                /**< problem variable */
   int                   pos                 /**< requested position */
   );

/** gets the number upper bound change info array */
EXTERN
int SCIPvarGetNBdchgInfosUb(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the value based history for the variable */
EXTERN
SCIP_VALUEHISTORY* SCIPvarGetValuehistory(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the index of the connected component of the clique graph that the variable belongs to, or -1 if not computed */
EXTERN
int SCIPvarGetCliqueComponentIdx(
   SCIP_VAR*             var                 /**< problem variable */
   );

#ifdef NDEBUG

/* In optimized mode, the function calls are overwritten by defines to reduce the number of function calls and
 * speed up the algorithms.
 */

#define SCIPvarGetName(var)             (var)->name
#define SCIPvarGetNUses(var)            (var)->nuses
#define SCIPvarGetData(var)             (var)->vardata
#define SCIPvarSetData(var,vdata)       (var)->vardata = (vdata)
#define SCIPvarSetDelorigData(var,func) (var)->vardelorig = (func)
#define SCIPvarSetTransData(var,func)   (var)->vartrans = (func)
#define SCIPvarSetDeltransData(var,func) (var)->vardeltrans = (func)
#define SCIPvarGetStatus(var)           (SCIP_VARSTATUS)((var)->varstatus)
#define SCIPvarIsOriginal(var)          ((var)->varstatus == SCIP_VARSTATUS_ORIGINAL \
      || ((var)->varstatus == SCIP_VARSTATUS_NEGATED && (var)->negatedvar->varstatus == SCIP_VARSTATUS_ORIGINAL))
#define SCIPvarIsTransformed(var)       ((var)->varstatus != SCIP_VARSTATUS_ORIGINAL \
      && ((var)->varstatus != SCIP_VARSTATUS_NEGATED || (var)->negatedvar->varstatus != SCIP_VARSTATUS_ORIGINAL))
#define SCIPvarIsNegated(var)           ((var)->varstatus == SCIP_VARSTATUS_NEGATED)
#define SCIPvarGetType(var)             ((SCIP_VARTYPE)((var)->vartype))
#define SCIPvarIsBinary(var)            ((var)->vartype == SCIP_VARTYPE_BINARY || \
      ((var)->vartype != SCIP_VARTYPE_CONTINUOUS && MAX((var)->glbdom.lb, (var)->lazylb) >= 0.0  && MIN((var)->glbdom.ub, (var)->lazyub) <= 1.0))
#define SCIPvarIsIntegral(var)          ((var)->vartype != SCIP_VARTYPE_CONTINUOUS)
#define SCIPvarIsInitial(var)           (var)->initial
#define SCIPvarIsRemovable(var)         (var)->removable
#define SCIPvarIsDeleted(var)           (var)->deleted
#define SCIPvarMarkDeletable(var)       (var)->deletable = TRUE
#define SCIPvarMarkNotDeletable(var)    (var)->deletable = FALSE
#define SCIPvarIsDeletable(var)         (var)->deletable
#define SCIPvarIsActive(var)            ((var)->probindex >= 0)
#define SCIPvarGetIndex(var)            (var)->index
#define SCIPvarGetProbindex(var)        (var)->probindex
#define SCIPvarGetTransVar(var)         (var)->data.original.transvar
#define SCIPvarGetCol(var)              (var)->data.col
#define SCIPvarIsInLP(var)              ((var)->varstatus == SCIP_VARSTATUS_COLUMN && SCIPcolIsInLP((var)->data.col))
/* use different name for var - otherwise we have clash with the var at the end */
#define SCIPvarGetAggrVar(war)          (war)->data.aggregate.var
#define SCIPvarGetAggrScalar(var)       (var)->data.aggregate.scalar
#define SCIPvarGetAggrConstant(var)     (var)->data.aggregate.constant
#define SCIPvarGetMultaggrNVars(var)    (var)->data.multaggr.nvars
#define SCIPvarGetMultaggrVars(var)     (var)->data.multaggr.vars
#define SCIPvarGetMultaggrScalars(var)  (var)->data.multaggr.scalars
#define SCIPvarGetMultaggrConstant(var) (var)->data.multaggr.constant
#define SCIPvarGetNegatedVar(var)       (var)->negatedvar
#define SCIPvarGetNegationVar(var)      (var)->negatedvar
#define SCIPvarGetNegationConstant(var) (var)->data.negate.constant
#define SCIPvarGetObj(var)              (var)->obj
#define SCIPvarGetLbOriginal(var)       ((var)->varstatus == SCIP_VARSTATUS_ORIGINAL \
      ? (var)->data.original.origdom.lb                                 \
      : (var)->data.negate.constant - (var)->negatedvar->data.original.origdom.ub)
#define SCIPvarGetUbOriginal(var)       ((var)->varstatus == SCIP_VARSTATUS_ORIGINAL \
      ? (var)->data.original.origdom.ub                                 \
      : (var)->data.negate.constant - (var)->negatedvar->data.original.origdom.lb)
#define SCIPvarGetHolelistOriginal(var) ((var)->varstatus == SCIP_VARSTATUS_ORIGINAL \
      ? (var)->data.original.origdom.holelist                           \
      : NULL)
#define SCIPvarGetLbGlobal(var)         (var)->glbdom.lb
#define SCIPvarGetUbGlobal(var)         (var)->glbdom.ub
#define SCIPvarGetHolelistGlobal(var)   (var)->glbdom.holelist
#define SCIPvarGetBestBoundGlobal(var)  ((var)->obj >= 0.0 ? (var)->glbdom.lb : (var)->glbdom.ub)
#define SCIPvarGetWorstBoundGlobal(var) ((var)->obj >= 0.0 ? (var)->glbdom.ub : (var)->glbdom.lb)
#define SCIPvarGetLbLocal(var)          (var)->locdom.lb
#define SCIPvarGetUbLocal(var)          (var)->locdom.ub
#define SCIPvarGetHolelistLocal(var)    (var)->locdom.holelist
#define SCIPvarGetBestBoundLocal(var)   ((var)->obj >= 0.0 ? (var)->locdom.lb : (var)->locdom.ub)
#define SCIPvarGetWorstBoundLocal(var)  ((var)->obj >= 0.0 ? (var)->locdom.ub : (var)->locdom.lb)
#define SCIPvarGetBestBoundType(var)    ((var)->obj >= 0.0 ? SCIP_BOUNDTYPE_LOWER : SCIP_BOUNDTYPE_UPPER)
#define SCIPvarGetWorstBoundType(var)   ((var)->obj >= 0.0 ? SCIP_BOUNDTYPE_UPPER : SCIP_BOUNDTYPE_LOWER)
#define SCIPvarGetLbLazy(var)           (var)->lazylb
#define SCIPvarGetUbLazy(var)           (var)->lazyub
#define SCIPvarGetBranchFactor(var)     (var)->branchfactor
#define SCIPvarGetBranchPriority(var)   (var)->branchpriority
#define SCIPvarGetBranchDirection(var)  (var)->branchdirection
#define SCIPvarGetNVlbs(var)            (SCIPvboundsGetNVbds((var)->vlbs))
#define SCIPvarGetVlbVars(var)          (SCIPvboundsGetVars((var)->vlbs))
#define SCIPvarGetVlbCoefs(var)         (SCIPvboundsGetCoefs((var)->vlbs))
#define SCIPvarGetVlbConstants(var)     (SCIPvboundsGetConstants((var)->vlbs))
#define SCIPvarGetNVubs(var)            (SCIPvboundsGetNVbds((var)->vubs))
#define SCIPvarGetVubVars(var)          (SCIPvboundsGetVars((var)->vubs))
#define SCIPvarGetVubCoefs(var)         (SCIPvboundsGetCoefs((var)->vubs))
#define SCIPvarGetVubConstants(var)     (SCIPvboundsGetConstants((var)->vubs))
#define SCIPvarGetNImpls(var, fix)      (SCIPimplicsGetNImpls((var)->implics, fix))
#define SCIPvarGetImplVars(var, fix)    (SCIPimplicsGetVars((var)->implics, fix))
#define SCIPvarGetImplTypes(var, fix)   (SCIPimplicsGetTypes((var)->implics, fix))
#define SCIPvarGetImplBounds(var, fix)  (SCIPimplicsGetBounds((var)->implics, fix))
#define SCIPvarGetImplIds(var, fix)     (SCIPimplicsGetIds((var)->implics, fix))
#define SCIPvarGetNCliques(var, fix)    (SCIPcliquelistGetNCliques((var)->cliquelist, fix))
#define SCIPvarGetCliques(var, fix)     (SCIPcliquelistGetCliques((var)->cliquelist, fix))
#define SCIPvarGetLPSol(var)            ((var)->varstatus == SCIP_VARSTATUS_COLUMN ? SCIPcolGetPrimsol((var)->data.col) : SCIPvarGetLPSol_rec(var))
#define SCIPvarGetNLPSol(var)           (((var)->varstatus == SCIP_VARSTATUS_COLUMN || ((var)->varstatus == SCIP_VARSTATUS_LOOSE)) ? (var)->nlpsol : SCIPvarGetNLPSol_rec(var))
#define SCIPvarGetBdchgInfoLb(var, pos)   (&((var)->lbchginfos[pos]))
#define SCIPvarGetNBdchgInfosLb(var)      ((var)->nlbchginfos)
#define SCIPvarGetBdchgInfoUb(var, pos)   (&((var)->ubchginfos[pos]))
#define SCIPvarGetNBdchgInfosUb(var)      ((var)->nubchginfos)
#define SCIPvarGetValuehistory(var)       (var)->valuehistory
#define SCIPvarGetCliqueComponentIdx(var) ((var)->clqcomponentidx)

#endif

/** gets primal LP solution value of variable */
EXTERN
SCIP_Real SCIPvarGetLPSol_rec(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets primal NLP solution value of variable */
EXTERN
SCIP_Real SCIPvarGetNLPSol_rec(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets pseudo solution value of variable at current node */
EXTERN
SCIP_Real SCIPvarGetPseudoSol(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** gets current LP or pseudo solution value of variable */
EXTERN
SCIP_Real SCIPvarGetSol(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Bool             getlpval            /**< should the LP solution value be returned? */
   );

/** returns the solution of the variable in the last root node's relaxation, if the root relaxation is not yet
 *  completely solved, zero is returned
 */
EXTERN
SCIP_Real SCIPvarGetRootSol(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the best solution (w.r.t. root reduced cost propagation) of the variable in the root node's relaxation, if
 *  the root relaxation is not yet completely solved, zero is returned
 */
EXTERN
SCIP_Real SCIPvarGetBestRootSol(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the best reduced costs (w.r.t. root reduced cost propagation) of the variable in the root node's relaxation,
 *  if the root relaxation is not yet completely solved, or the variable was no column of the root LP, SCIP_INVALID is
 *  returned
 */
EXTERN
SCIP_Real SCIPvarGetBestRootRedcost(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the best objective value (w.r.t. root reduced cost propagation) of the root LP which belongs the root
 *  reduced cost which is accessible via SCIPvarGetRootRedcost() or the variable was no column of the root LP,
 *  SCIP_INVALID is returned
 */
EXTERN
SCIP_Real SCIPvarGetBestRootLPObjval(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** set the given solution as the best root solution w.r.t. root reduced cost propagation in the variables */
EXTERN
void SCIPvarSetBestRootSol(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             rootsol,            /**< root solution value */
   SCIP_Real             rootredcost,        /**< root reduced cost */
   SCIP_Real             rootlpobjval        /**< objective value of the root LP */
   );

/** returns a weighted average solution value of the variable in all feasible primal solutions found so far */
EXTERN
SCIP_Real SCIPvarGetAvgSol(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the bound change information for the last lower bound change on given active problem variable before or
 *  after the bound change with the given index was applied;
 *  returns NULL, if no change to the lower bound was applied up to this point of time
 */
EXTERN
SCIP_BDCHGINFO* SCIPvarGetLbchgInfo(
   SCIP_VAR*             var,                /**< active problem variable */
   SCIP_BDCHGIDX*        bdchgidx,           /**< bound change index representing time on path to current node */
   SCIP_Bool             after               /**< should the bound change with given index be included? */
   );

/** returns the bound change information for the last upper bound change on given active problem variable before or
 *  after the bound change with the given index was applied;
 *  returns NULL, if no change to the upper bound was applied up to this point of time
 */
EXTERN
SCIP_BDCHGINFO* SCIPvarGetUbchgInfo(
   SCIP_VAR*             var,                /**< active problem variable */
   SCIP_BDCHGIDX*        bdchgidx,           /**< bound change index representing time on path to current node */
   SCIP_Bool             after               /**< should the bound change with given index be included? */
   );

/** returns the bound change information for the last lower or upper bound change on given active problem variable
 *  before or after the bound change with the given index was applied;
 *  returns NULL, if no change to the lower/upper bound was applied up to this point of time
 */
EXTERN
SCIP_BDCHGINFO* SCIPvarGetBdchgInfo(
   SCIP_VAR*             var,                /**< active problem variable */
   SCIP_BOUNDTYPE        boundtype,          /**< type of bound: lower or upper bound */
   SCIP_BDCHGIDX*        bdchgidx,           /**< bound change index representing time on path to current node */
   SCIP_Bool             after               /**< should the bound change with given index be included? */
   );

/** returns lower bound of variable directly before or after the bound change given by the bound change index
 *  was applied
 *
 *  @deprecated Please use SCIPgetVarLbAtIndex()
 */
EXTERN
SCIP_Real SCIPvarGetLbAtIndex(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BDCHGIDX*        bdchgidx,           /**< bound change index representing time on path to current node */
   SCIP_Bool             after               /**< should the bound change with given index be included? */
   );

/** returns upper bound of variable directly before or after the bound change given by the bound change index
 *  was applied
 *
 *  @deprecated Please use SCIPgetVarUbAtIndex()
 */
EXTERN
SCIP_Real SCIPvarGetUbAtIndex(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BDCHGIDX*        bdchgidx,           /**< bound change index representing time on path to current node */
   SCIP_Bool             after               /**< should the bound change with given index be included? */
   );

/** returns lower or upper bound of variable directly before or after the bound change given by the bound change index
 *  was applied
 *
 *  @deprecated Please use SCIPgetVarBdAtIndex()
 */
EXTERN
SCIP_Real SCIPvarGetBdAtIndex(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BOUNDTYPE        boundtype,          /**< type of bound: lower or upper bound */
   SCIP_BDCHGIDX*        bdchgidx,           /**< bound change index representing time on path to current node */
   SCIP_Bool             after               /**< should the bound change with given index be included? */
   );

/** returns whether the binary variable was fixed at the time given by the bound change index
 *
 *  @deprecated Please use SCIPgetVarWasFixedAtIndex()
 */
EXTERN
SCIP_Bool SCIPvarWasFixedAtIndex(
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BDCHGIDX*        bdchgidx,           /**< bound change index representing time on path to current node */
   SCIP_Bool             after               /**< should the bound change with given index be included? */
   );

/** returns the last bound change index, at which the bounds of the given variable were tightened */
EXTERN
SCIP_BDCHGIDX* SCIPvarGetLastBdchgIndex(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns the last depth level, at which the bounds of the given variable were tightened;
 *  returns -2, if the variable's bounds are still the global bounds
 *  returns -1, if the variable was fixed in presolving
 */
EXTERN
int SCIPvarGetLastBdchgDepth(
   SCIP_VAR*             var                 /**< problem variable */
   );

/** returns whether the first binary variable was fixed earlier than the second one;
 *  returns FALSE, if the first variable is not fixed, and returns TRUE, if the first variable is fixed, but the
 *  second one is not fixed
 */
EXTERN
SCIP_Bool SCIPvarWasFixedEarlier(
   SCIP_VAR*             var1,               /**< first binary variable */
   SCIP_VAR*             var2                /**< second binary variable */
   );

/**
 * @name Public SCIP_BDCHGIDX Methods
 *
 * @{
 */

/** returns whether first bound change index belongs to an earlier applied bound change than second one;
 *  if a bound change index is NULL, the bound change index represents the current time, i.e. the time after the
 *  last bound change was applied to the current node
 */
EXTERN
SCIP_Bool SCIPbdchgidxIsEarlier(
   SCIP_BDCHGIDX*        bdchgidx1,          /**< first bound change index, or NULL */
   SCIP_BDCHGIDX*        bdchgidx2           /**< second bound change index, or NULL */
   );

/** returns whether first bound change index belongs to an earlier applied bound change than second one */
EXTERN
SCIP_Bool SCIPbdchgidxIsEarlierNonNull(
   SCIP_BDCHGIDX*        bdchgidx1,          /**< first bound change index */
   SCIP_BDCHGIDX*        bdchgidx2           /**< second bound change index */
   );

/**@} */

/**
 * @name Public SCIP_BDCHGINFO Methods
 *
 * @{
 */

/** returns old bound that was overwritten for given bound change information */
EXTERN
SCIP_Real SCIPbdchginfoGetOldbound(
   SCIP_BDCHGINFO*       bdchginfo           /**< bound change information */
   );

/** returns new bound installed for given bound change information */
EXTERN
SCIP_Real SCIPbdchginfoGetNewbound(
   SCIP_BDCHGINFO*       bdchginfo           /**< bound change information */
   );

/** returns variable that belongs to the given bound change information */
EXTERN
SCIP_VAR* SCIPbdchginfoGetVar(
   SCIP_BDCHGINFO*       bdchginfo           /**< bound change information */
   );

/** returns whether the bound change information belongs to a branching decision or a deduction */
EXTERN
SCIP_BOUNDCHGTYPE SCIPbdchginfoGetChgtype(
   SCIP_BDCHGINFO*       bdchginfo           /**< bound change information */
   );

/** returns whether the bound change information belongs to a lower or upper bound change */
EXTERN
SCIP_BOUNDTYPE SCIPbdchginfoGetBoundtype(
   SCIP_BDCHGINFO*       bdchginfo           /**< bound change information */
   );

/** returns depth level of given bound change information */
EXTERN
int SCIPbdchginfoGetDepth(
   SCIP_BDCHGINFO*       bdchginfo           /**< bound change information */
   );

/** returns bound change position in its depth level of given bound change information */
EXTERN
int SCIPbdchginfoGetPos(
   SCIP_BDCHGINFO*       bdchginfo           /**< bound change information */
   );

/** returns bound change index of given bound change information */
EXTERN
SCIP_BDCHGIDX* SCIPbdchginfoGetIdx(
   SCIP_BDCHGINFO*       bdchginfo           /**< bound change information */
   );

/** returns inference variable of given bound change information */
EXTERN
SCIP_VAR* SCIPbdchginfoGetInferVar(
   SCIP_BDCHGINFO*       bdchginfo           /**< bound change information */
   );

/** returns inference constraint of given bound change information */
EXTERN
SCIP_CONS* SCIPbdchginfoGetInferCons(
   SCIP_BDCHGINFO*       bdchginfo           /**< bound change information */
   );

/** returns inference propagator of given bound change information, or NULL if no propagator was responsible */
EXTERN
SCIP_PROP* SCIPbdchginfoGetInferProp(
   SCIP_BDCHGINFO*       bdchginfo           /**< bound change information */
   );

/** returns inference user information of given bound change information */
EXTERN
int SCIPbdchginfoGetInferInfo(
   SCIP_BDCHGINFO*       bdchginfo           /**< bound change information */
   );

/** returns inference bound of inference variable of given bound change information */
EXTERN
SCIP_BOUNDTYPE SCIPbdchginfoGetInferBoundtype(
   SCIP_BDCHGINFO*       bdchginfo           /**< bound change information */
   );

/** returns whether the bound change information belongs to a redundant bound change */
EXTERN
SCIP_Bool SCIPbdchginfoIsRedundant(
   SCIP_BDCHGINFO*       bdchginfo           /**< bound change information */
   );

/** returns whether the bound change has an inference reason (constraint or propagator), that can be resolved */
EXTERN
SCIP_Bool SCIPbdchginfoHasInferenceReason(
   SCIP_BDCHGINFO*       bdchginfo           /**< bound change information */
   );

/** for two bound change informations belonging to the same variable and bound, returns whether the first bound change
 *  has a tighter new bound as the second bound change
 */
EXTERN
SCIP_Bool SCIPbdchginfoIsTighter(
   SCIP_BDCHGINFO*       bdchginfo1,         /**< first bound change information */
   SCIP_BDCHGINFO*       bdchginfo2          /**< second bound change information */
   );

/**@} */

/**
 * @name Public SCIP_BOUNDCHG Methods
 *
 * @{
 */

/** returns the new value of the bound in the bound change data */
EXTERN
SCIP_Real SCIPboundchgGetNewbound(
   SCIP_BOUNDCHG*        boundchg            /**< bound change data */
   );

/** returns the variable of the bound change in the bound change data */
EXTERN
SCIP_VAR* SCIPboundchgGetVar(
   SCIP_BOUNDCHG*        boundchg            /**< bound change data */
   );

/** returns the bound change type of the bound change in the bound change data */
EXTERN
SCIP_BOUNDCHGTYPE SCIPboundchgGetBoundchgtype(
   SCIP_BOUNDCHG*        boundchg            /**< bound change data */
   );

/** returns the bound type of the bound change in the bound change data */
EXTERN
SCIP_BOUNDTYPE SCIPboundchgGetBoundtype(
   SCIP_BOUNDCHG*        boundchg            /**< bound change data */
   );

/** returns whether the bound change is redundant due to a more global bound that is at least as strong */
EXTERN
SCIP_Bool SCIPboundchgIsRedundant(
   SCIP_BOUNDCHG*        boundchg            /**< bound change data */
   );

/** @} */

/**
 * @name Public SCIP_DOMCHG Methods
 *
 * @{
 */

/** returns the number of bound changes in the domain change data */
EXTERN
int SCIPdomchgGetNBoundchgs(
   SCIP_DOMCHG*          domchg              /**< domain change data */
   );

/** returns a particular bound change in the domain change data */
EXTERN
SCIP_BOUNDCHG* SCIPdomchgGetBoundchg(
   SCIP_DOMCHG*          domchg,             /**< domain change data */
   int                   pos                 /**< position of the bound change in the domain change data */
   );

/**@} */

/**
 * @name Public SCIP_HOLELIST Methods
 *
 * @{
 */

/** returns left bound of open interval in hole */
EXTERN
SCIP_Real SCIPholelistGetLeft(
   SCIP_HOLELIST*        holelist            /**< hole list pointer to hole of interest */
   );

/** returns right bound of open interval in hole */
EXTERN
SCIP_Real SCIPholelistGetRight(
   SCIP_HOLELIST*        holelist            /**< hole list pointer to hole of interest */
   );

/** returns next hole in list or NULL */
EXTERN
SCIP_HOLELIST* SCIPholelistGetNext(
   SCIP_HOLELIST*        holelist            /**< hole list pointer to hole of interest */
   );

/**@} */

#ifdef NDEBUG

/* In optimized mode, the function calls are overwritten by defines to reduce the number of function calls and
 * speed up the algorithms.
 */

#define SCIPbdchgidxIsEarlierNonNull(idx1,idx2)                         \
   ((idx1)->depth < (idx2)->depth || ((idx1)->depth == (idx2)->depth && (idx1)->pos < (idx2)->pos))
#define SCIPbdchgidxIsEarlier(idx1,idx2)                                \
   ((idx1) != NULL && ((idx2) == NULL || SCIPbdchgidxIsEarlierNonNull(idx1, idx2)))
#define SCIPbdchginfoGetOldbound(bdchginfo)       (bdchginfo)->oldbound
#define SCIPbdchginfoGetNewbound(bdchginfo)       (bdchginfo)->newbound
#define SCIPbdchginfoGetVar(bdchginfo)            (bdchginfo)->var
#define SCIPbdchginfoGetChgtype(bdchginfo)        (SCIP_BOUNDCHGTYPE)((bdchginfo)->boundchgtype)
#define SCIPbdchginfoGetBoundtype(bdchginfo)      (SCIP_BOUNDTYPE)((bdchginfo)->boundtype)
#define SCIPbdchginfoGetDepth(bdchginfo)          (bdchginfo)->bdchgidx.depth
#define SCIPbdchginfoGetPos(bdchginfo)            (bdchginfo)->bdchgidx.pos
#define SCIPbdchginfoGetIdx(bdchginfo)            (&(bdchginfo)->bdchgidx)
#define SCIPbdchginfoGetInferVar(bdchginfo)       (bdchginfo)->inferencedata.var
#define SCIPbdchginfoGetInferCons(bdchginfo)      (bdchginfo)->inferencedata.reason.cons
#define SCIPbdchginfoGetInferProp(bdchginfo)      (bdchginfo)->inferencedata.reason.prop
#define SCIPbdchginfoGetInferInfo(bdchginfo)      (bdchginfo)->inferencedata.info
#define SCIPbdchginfoGetInferBoundtype(bdchginfo) (SCIP_BOUNDTYPE)((bdchginfo)->inferboundtype)
#define SCIPbdchginfoIsRedundant(bdchginfo)       (bdchginfo)->redundant
#define SCIPbdchginfoHasInferenceReason(bdchginfo)                      \
   (((bdchginfo)->boundchgtype == SCIP_BOUNDCHGTYPE_CONSINFER)          \
      || ((bdchginfo)->boundchgtype == SCIP_BOUNDCHGTYPE_PROPINFER && (bdchginfo)->inferencedata.reason.prop != NULL))
#define SCIPbdchginfoIsTighter(bdchginfo1,bdchginfo2) ((bdchginfo1)->boundtype == SCIP_BOUNDTYPE_LOWER \
      ? (bdchginfo1)->newbound > bdchginfo2->newbound : (bdchginfo1)->newbound < bdchginfo2->newbound)
#define SCIPboundchgGetNewbound(boundchg)      ((boundchg)->newbound)
#define SCIPboundchgGetVar(boundchg)           ((boundchg)->var)
#define SCIPboundchgGetBoundchgtype(boundchg)  ((SCIP_BOUNDCHGTYPE)((boundchg)->boundchgtype))
#define SCIPboundchgGetBoundtype(boundchg)     ((SCIP_BOUNDTYPE)((boundchg)->boundtype))
#define SCIPboundchgIsRedundant(boundchg)      ((boundchg)->redundant)
#define SCIPdomchgGetNBoundchgs(domchg)        ((domchg) != NULL ? (domchg)->domchgbound.nboundchgs : 0)
#define SCIPdomchgGetBoundchg(domchg, pos)     (&(domchg)->domchgbound.boundchgs[pos])
#define SCIPholelistGetLeft(holelist)          ((holelist)->hole.left) 
#define SCIPholelistGetRight(holelist)         ((holelist)->hole.right)
#define SCIPholelistGetNext(holelist)          ((holelist)->next)

#endif

/**@} */

#ifdef __cplusplus
}
#endif

#endif
